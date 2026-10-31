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

from typing import List, NamedTuple
import numpy as np

from cl_rrt import CLRRT
from collision import CollisionChecker
from dynamics import PurePursuit
from tree import Node


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
        """
        self._planner = planner
        self._controller = controller
        self._collision_checker = collision_checker
        self._goal_pos = np.asarray(goal_pos, dtype=float)
        self._dt = controller.dt
        self._steps_per_cycle = int(round(replan_interval / self._dt))
        self._samples_per_cycle = samples_per_cycle
        self._turn_rate = turn_rate

        self.state = np.asarray(start_state, dtype=float)
        self.time = 0.0
        self._planner.reset(self.state)
        self.committed_states = np.empty((0, 3))  # states up to the root of the tree
        self.plan_nodes: List[Node] = [self._planner.tree.root]  # from the root to the end of the plan

    def is_in_goal(self) -> bool:
        return self._planner.is_in_goal(Node(self.state, None, 0.0), self._goal_pos)

    def future_states(self) -> np.ndarray:
        """
        States the robot is going to go through, if the plan does not change
        :return: states size: (N, 3)
        """
        segments = [self.committed_states] + [node.trajectory for node in self.plan_nodes[1:]]
        return np.vstack(segments)

    def update(self) -> CycleResult:
        """
        One cycle of Algorithm 2
        """
        is_safety_stop = self._stop_if_committed_path_is_unsafe()
        num_lazy_removed = self._commit()

        for _ in range(self._samples_per_cycle):
            self._planner.step(self._goal_pos)

        while True:
            best_node = self._planner.best_node(self._goal_pos)
            if self._planner.lazy_check(best_node):
                break
            num_lazy_removed += 1
        plan_nodes = self._planner.tree.nodes_to(best_node)
        is_path_changed = plan_nodes[-1] is not self.plan_nodes[-1]
        self.plan_nodes = plan_nodes

        is_turning = len(self.committed_states) == 0 and len(self.plan_nodes) == 1
        executed_states = self._turn_in_place() if is_turning else self._move()
        return CycleResult(executed_states, num_lazy_removed, is_safety_stop, is_path_changed, is_turning)

    def _stop_if_committed_path_is_unsafe(self) -> bool:
        """
        The safety action of Algorithm 2. The committed path cannot be changed, so stop here and start a new tree
        """
        is_unsafe = (
            self._collision_checker.find_first_unsafe(self.committed_states[:, :2]) is not None
        )
        if not is_unsafe:
            return False

        self._planner.reset(self.state)
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
        while len(self.committed_states) < self._steps_per_cycle and len(self.plan_nodes) > 1:
            next_node = self.plan_nodes[1]
            if self._collision_checker.find_first_unsafe(next_node.trajectory[:, :2]) is not None:
                self._planner.tree.remove_subtree(next_node)
                self.plan_nodes = self.plan_nodes[:1]
                num_removed += 1
                break
            self.committed_states = np.vstack((self.committed_states, next_node.trajectory))
            self.plan_nodes = self.plan_nodes[1:]
        self._planner.tree.reroot(self.plan_nodes[0])
        return num_removed

    def _turn_in_place(self) -> np.ndarray:
        """
        Not in the paper. A stopped robot can have no feasible path at all because of its heading, like when
        it faces a pillar, and the tree would never grow again. A skid-steer robot can turn in place,
        and a circular robot occupies the same space while turning, so this is always safe.
        """
        model = self._controller.model
        delta_v = self._turn_rate * model.wheel_base
        states = []
        state = self.state
        for _ in range(self._steps_per_cycle):
            state = model.step(state, 0.0, delta_v, self._dt)
            states.append(state)
        self.state = state
        self._planner.reset(self.state)
        self.plan_nodes = [self._planner.tree.root]
        self.time += self._steps_per_cycle * self._dt
        return np.array(states)

    def _move(self) -> np.ndarray:
        """
        Move along the committed path, the robot stays stopped when the path ends
        """
        executed_states = self.committed_states[: self._steps_per_cycle]
        for i, state in enumerate(executed_states):
            if self._planner.is_in_goal(Node(state, None, 0.0), self._goal_pos):
                executed_states = executed_states[: i + 1]
                break
        self.committed_states = self.committed_states[len(executed_states) :]
        if len(executed_states) > 0:
            self.state = executed_states[-1]
        self.time += self._steps_per_cycle * self._dt
        return executed_states
