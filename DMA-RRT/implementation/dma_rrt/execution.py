"""
    Execution loop of CL-RRT: Algorithm 2 in the DMA-RRT paper (step 3)

    The robot moves while the tree keeps growing. In each cycle of replan_interval:
    1. Commit to the path up to the first node the robot cannot avoid passing during this cycle.
       That node becomes the root, and only its descendants stay in the tree.
    2. Grow the tree from there.
    3. Select the best path, and check it again with the lazy check.
    4. Move the robot along the committed path for replan_interval.

    The robot stops at the end of its path, and every node is safe to stop at because the model
    has no acceleration limit and the map is static. This is the safe terminal state of eq. 7-8.
"""

from __future__ import annotations

from typing import Callable, List, NamedTuple, Tuple
import numpy as np

from dma_rrt.cl_rrt import CLRRT
from dma_rrt.collision import CollisionChecker
from dma_rrt.dynamics import PurePursuit
from dma_rrt.tree import Node


class CycleResult(NamedTuple):
    executed_states: np.ndarray  # states the robot went through in this cycle size: (N, 3)
    num_lazy_removed: int  # paths removed by the lazy check in this cycle
    is_safety_stop: bool  # the committed path became unsafe and the robot stopped
    is_path_changed: bool  # the robot's plan changed in this cycle
    is_turning: bool  # the robot was stopped with nowhere to go and turned in place


class CLRRTExecutor:
    def __init__(
        self,
        planner: CLRRT,
        controller: PurePursuit,
        collision_checker: CollisionChecker,
        start_state: np.ndarray,
        goal_pos: np.ndarray,
        replan_interval: float = 1.0,
        samples_per_cycle: int = 30,
        turn_rate: float = 1.0,
        min_tree_growth: int = 2,
        can_turn: bool = True,
        stop_at_goal: bool = True,
    ) -> None:
        """
        :param planner: planner to grow the tree
        :param controller: closed loop model of the planner
        :param collision_checker: collision checker of the planner
        :param start_state: start state (x, y, yaw)
        :param goal_pos: center of the goal region
        :param replan_interval: time of a cycle, delta t in Algorithm 2 [s]
        :param samples_per_cycle: samples to grow the tree in a cycle, instead of a time budget to be reproducible
        :param turn_rate: yaw rate when turning in place [rad/s]
        :param min_tree_growth: a stopped robot whose tree gets fewer new nodes than this in a cycle turns in place.
            0 only turns when the tree has nothing but the root, which is what we first implemented
        :param can_turn: False never turns in place, as in the paper
        :param stop_at_goal: stop moving at the first state in the goal region. A robot sharing its plan has to
            keep to the plan's timing instead, so it needs False
        """
        self._planner = planner
        self._controller = controller
        self._collision_checker = collision_checker
        self.goal_pos = np.asarray(goal_pos, dtype=float)
        self._stop_at_goal = stop_at_goal
        self._dt = controller.dt
        self.steps_per_cycle = int(round(replan_interval / self._dt))
        self.samples_per_cycle = samples_per_cycle
        self._turn_rate = turn_rate
        self._min_tree_growth = min_tree_growth
        self._can_turn = can_turn

        self.state = np.asarray(start_state, dtype=float)
        self.time = 0.0
        self._planner.reset(self.state)
        self.committed_states = np.empty((0, 3))  # states up to the root of the tree
        self.plan_nodes: List[Node] = [self._planner.tree.root]  # from the root to the end of the plan
        self.plan_selector: Callable[[], Tuple[Node, int]] = self._select_plan

    def start_at(self, time: float) -> None:
        """
        Stay stopped until the given time, for a robot whose first cycle starts later than the others
        """
        self.time = time
        self._planner.reset(self.state, time)
        self.plan_nodes = [self._planner.tree.root]

    def state_at(self, time: float) -> np.ndarray:
        """
        :param time: time in the current cycle [s]
        :return: state of the robot at that time. A robot turning in place only gets its new yaw at the end of the cycle
        """
        num_steps = min(int(round((time - self.time) / self._dt)), len(self.committed_states))
        return self.committed_states[num_steps - 1] if num_steps > 0 else self.state

    def is_in_goal(self) -> bool:
        return self._planner.is_in_goal(Node(self.state, None, 0.0), self.goal_pos)

    def future_states(self) -> np.ndarray:
        """
        States the robot is going to go through, if the plan does not change
        :return: states size: (N, 3)
        """
        segments = [self.committed_states] + [node.trajectory for node in self.plan_nodes[1:]]
        return np.vstack(segments)

    def update(self, can_change_plan: bool = True) -> CycleResult:
        """
        One cycle of Algorithm 2
        :param can_change_plan: False keeps the current plan and only grows the tree, for a robot without the token
        """
        self.begin_cycle()
        self.grow(self.samples_per_cycle)
        return self.end_cycle(can_change_plan)

    def begin_cycle(self) -> None:
        """
        Start of a cycle: fix what the robot does in this cycle, the tree grows from where it ends up
        """
        self._is_safety_stop = self._stop_if_committed_path_is_unsafe()
        self._num_lazy_removed = self._commit()
        self._tree_size = len(self._planner.tree)

    def grow(self, num_samples: int) -> None:
        for _ in range(num_samples):
            self._planner.step(self.goal_pos)

    def end_cycle(self, can_change_plan: bool = True) -> CycleResult:
        """
        End of a cycle: select the plan and move the robot to where it is at this time
        :param can_change_plan: False keeps the current plan, for a robot without the token
        """
        is_safety_stop = self._is_safety_stop
        num_lazy_removed = self._num_lazy_removed
        tree_growth = len(self._planner.tree) - self._tree_size

        is_path_changed = False
        if can_change_plan:
            num_removed, is_path_changed = self.change_plan()
            num_lazy_removed += num_removed

        # a tree that keeps growing means the robot is only waiting for a moving obstacle to pass or for the token
        is_stopped = len(self.committed_states) == 0 and len(self.plan_nodes) == 1
        is_turning = (
            self._can_turn and is_stopped and (len(self._planner.tree) == 1 or tree_growth < self._min_tree_growth)
        )
        executed_states = self._turn_in_place() if is_turning else self._move()
        return CycleResult(executed_states, num_lazy_removed, is_safety_stop, is_path_changed, is_turning)

    def change_plan(self) -> Tuple[int, bool]:
        """
        Take the best path as the plan. It starts at the root, so this can also be done during a cycle
        :return: number of paths removed by the lazy check, and if the plan changed
        """
        best_node, num_removed = self.plan_selector()
        plan_nodes = self._planner.tree.nodes_to(best_node)
        is_path_changed = plan_nodes[-1] is not self.plan_nodes[-1]
        self.plan_nodes = plan_nodes
        return num_removed, is_path_changed

    def _select_plan(self) -> Tuple[Node, int]:
        """
        Select the best path that passes the lazy check
        If the plan would end at the root but the robot cannot stop there, keep the current plan:
        the other robots planned on the robot passing the root, not on it stopping there
        :return: end of the plan, and the number of paths removed by the lazy check
        """
        num_removed = 0
        while True:
            best_node = self._planner.best_node(self.goal_pos)
            root = self._planner.tree.root
            if best_node is root and len(self.plan_nodes) > 1 and self.plan_nodes[-1] in self._planner.tree.nodes:
                root_time = self._planner.node_time(root)
                if not self._collision_checker.is_stop_safe(root.state[:2], root_time):
                    best_node = self.plan_nodes[-1]
            if self._planner.lazy_check(best_node):
                return best_node, num_removed
            num_removed += 1

    def _stop_if_committed_path_is_unsafe(self) -> bool:
        """
        The safety action of Algorithm 2. The committed path cannot be changed, so stop here and start a new tree
        """
        times = self._planner.trajectory_times(self.time, len(self.committed_states))
        is_unsafe = (
            self._collision_checker.find_first_unsafe(self.committed_states[:, :2], times) is not None
        )
        if not is_unsafe:
            return False

        self._planner.reset(self.state, self.time)
        self.committed_states = np.empty((0, 3))
        self.plan_nodes = [self._planner.tree.root]
        return True

    def _commit(self) -> int:
        """
        Extend the committed path until it covers this cycle. The robot cannot avoid the nodes on it anymore,
        so a node that became unsafe since the last cycle is removed instead, and the robot stops before it
        :return: number of paths removed
        """
        num_removed = 0
        while len(self.committed_states) < self.steps_per_cycle and len(self.plan_nodes) > 1:
            next_node = self.plan_nodes[1]
            times = self._planner.trajectory_times(
                self.time + len(self.committed_states) * self._dt, len(next_node.trajectory)
            )
            if self._collision_checker.find_first_unsafe(next_node.trajectory[:, :2], times) is not None:
                self._planner.tree.remove_subtree(next_node)
                self.plan_nodes = self.plan_nodes[:1]
                num_removed += 1
                break
            self.committed_states = np.vstack((self.committed_states, next_node.trajectory))
            self.plan_nodes = self.plan_nodes[1:]
        self._planner.tree.reroot(self.plan_nodes[0])
        # new nodes are only committed in the next cycle, so a robot arriving at the root early waits there until then
        num_steps_at_root = max(len(self.committed_states), self.steps_per_cycle)
        self._planner.root_time = self.time + num_steps_at_root * self._dt
        return num_removed

    def _turn_in_place(self) -> np.ndarray:
        """
        Not in the paper. A stopped robot can have almost no feasible path because of its heading, like when
        it faces a pillar or a wall, and its tree would stop growing. A skid-steer robot can turn in place,
        and a circular robot occupies the same space while turning, so this is always safe.
        """
        model = self._controller.model
        delta_v = self._turn_rate * model.wheel_base
        states = []
        state = self.state
        for _ in range(self.steps_per_cycle):
            state = model.step(state, 0.0, delta_v, self._dt)
            states.append(state)
        self.state = state
        self.time += self.steps_per_cycle * self._dt
        self._planner.reset(self.state, self.time)
        self.plan_nodes = [self._planner.tree.root]
        return np.array(states)

    def _move(self) -> np.ndarray:
        """
        Move along the committed path, the robot stays stopped when the path ends
        """
        executed_states = self.committed_states[: self.steps_per_cycle]
        for i, state in enumerate(executed_states):
            if self._stop_at_goal and self._planner.is_in_goal(Node(state, None, 0.0), self.goal_pos):
                executed_states = executed_states[: i + 1]
                break
        self.committed_states = self.committed_states[len(executed_states) :]
        if len(executed_states) > 0:
            self.state = executed_states[-1]
        self.time += self.steps_per_cycle * self._dt
        return executed_states
