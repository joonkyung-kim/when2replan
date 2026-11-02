"""
    A robot of the team: the individual and interaction components of Algorithms 4-5 in the DMA-RRT paper (step 5)

    All robots plan with their own tree and only know the others from messages. A robot rebuilds
    another robot's trajectory by simulating its waypoints, and keeps it as a moving obstacle.

    The paper's message has waypoints and the winner of the token. Ours also has the state and the time at the
    start of the waypoints. The paper's robots can derive them, but ours can turn in place, which changes the state.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional
import numpy as np

from dma_rrt.cl_rrt import CLRRT
from dma_rrt.cl_rrt_paper import PaperCLRRT
from dma_rrt.collision import CollisionChecker, MovingObstacle, SIXTEEN_PILLAR_MAP
from dma_rrt.dynamics import PurePursuit
from dma_rrt.execution import CLRRTExecutor, CycleResult
from dma_rrt.tree import Node


class StopNode(NamedTuple):
    """
    A node where a robot can be asked to stop (Sec. 6.1 of the paper)
    """

    time: float  # time the robot arrives at the node [s]
    position: np.ndarray
    cost: float  # finish time estimate of the robot if it stops here [s]


class PlanMessage(NamedTuple):
    sender: int
    start_state: np.ndarray
    start_time: float
    from_pos: np.ndarray  # where the reference to the first waypoint starts
    waypoints: List[np.ndarray]
    winner: int  # next token holder
    stop_nodes: List[StopNode] = []  # emergency stop nodes of the plan (Cooperative DMA-RRT)
    end_cost: float = 0.0  # finish time estimate of the plan (Cooperative DMA-RRT)
    estop_target: Optional[int] = None  # robot that has to stop early (Cooperative DMA-RRT)
    estop_time: float = 0.0  # time of the node where it has to stop [s]


class Agent:
    def __init__(
        self,
        agent_id: int,
        start_state: np.ndarray,
        goals: List[np.ndarray],
        controller: PurePursuit,
        map_path: str = SIXTEEN_PILLAR_MAP,
        robot_radius: float = 1.0,
        samples_per_cycle: int = 30,
        select_ignoring_others: bool = False,
        grow_ignoring_others: bool = False,
        paper_tree: bool = False,
        can_turn: bool = True,
        commit_one_cycle: bool = False,
        seed: int = 0,
    ) -> None:
        """
        :param agent_id: id of this robot in the team
        :param start_state: start state (x, y, yaw), the robot is stopped there
        :param goals: goals to visit in order, it goes back to the first goal after the last one
        :param controller: closed loop model, known to all robots
        :param map_path: map yaml file, relative to the repo root
        :param robot_radius: radius of this robot and the others [m]
        :param samples_per_cycle: samples to grow the tree in a cycle
        :param select_ignoring_others: select paths without the other robots (Cooperative DMA-RRT)
        :param grow_ignoring_others: grow the tree without the other robots (Algorithm 6 of the paper)
        :param paper_tree: grow the tree as in Algorithm 1 (PaperCLRRT), instead of our CLRRT
        :param can_turn: False never turns in place, as in the paper
        :param commit_one_cycle: commit only up to the state at t + delta t, as in Algorithm 2
        :param seed: seed of this robot's planner
        """
        self.agent_id = agent_id
        self.goals = [np.asarray(goal, dtype=float) for goal in goals]
        self.num_goals_reached = 0
        self.waiting_time = 0.0  # time stopped at the end of the plan [s]
        self._goal_index = 0
        self._controller = controller
        self._robot_radius = robot_radius

        self.checker = CollisionChecker(map_path, robot_radius=robot_radius)
        planner_class = PaperCLRRT if paper_tree else CLRRT
        self.planner = planner_class(
            self.checker,
            controller,
            select_ignoring_others=select_ignoring_others,
            grow_ignoring_others=grow_ignoring_others,
            seed=seed,
        )
        self.bids: Dict[int, float] = {}  # latest bid of each other robot
        self.estop_target: Optional[int] = None  # robot this robot asked to stop in this cycle
        self._rng = np.random.default_rng(seed + 1)
        self.executor = CLRRTExecutor(
            self.planner,
            controller,
            self.checker,
            start_state,
            self.goals[0],
            samples_per_cycle=samples_per_cycle,
            can_turn=can_turn,
            stop_at_goal=False,
            commit_one_cycle=commit_one_cycle,
        )
        self._other_robots: Dict[int, MovingObstacle] = {}

    @property
    def goal_pos(self) -> np.ndarray:
        return self.executor.goal_pos

    def add_other_robot(self, agent_id: int, start_state: np.ndarray) -> None:
        """
        Register another robot, which is stopped at its start until its first plan
        """
        obstacle = MovingObstacle(
            np.asarray(start_state, dtype=float)[np.newaxis], 0.0, self._controller.dt, self._robot_radius
        )
        self._other_robots[agent_id] = obstacle
        self.checker.moving_obstacles = list(self._other_robots.values())
        self.bids[agent_id] = 0.0

    def compute_bid(self, add_waiting_time: bool = False, allow_negative: bool = False) -> float:
        """
        Potential path improvement, eq. 5: the cost of the current plan minus the cost of the best path in the tree.
        The tree already exists, so this needs no extra planning (line 12 of Algorithm 4)
        The current plan is in the tree as well, so the improvement is not negative. best_node leaves out the root,
        which is the current plan of a stopped robot, so this has to be stated here
        :param add_waiting_time: not in the paper. Add the time this robot has been stopped at the end of its plan
        :param allow_negative: keep negative improvements, which starve stopped robots. This is what we first
            implemented, kept to compare
        """
        best_node = self.planner.best_node(self.goal_pos)
        current_cost = self.planner.cost_estimate(self.executor.plan_nodes[-1], self.goal_pos)
        bid = current_cost - self.planner.cost_estimate(best_node, self.goal_pos)
        if not allow_negative:
            bid = max(bid, 0.0)
        return bid + self.waiting_time if add_waiting_time else bid

    def receive_bid(self, agent_id: int, bid: float, root_time: float = 0.0) -> None:
        """
        Keep the bid to use when this robot passes the token (line 10 of Algorithm 5)
        :param root_time: time of the sender's root, only used by Cooperative DMA-RRT
        """
        self.bids[agent_id] = bid

    def receive_root_time(self, agent_id: int, root_time: float) -> None:
        """
        Only used by Cooperative DMA-RRT
        """

    def select_winner(self) -> int:
        """
        Select the robot with the best bid, at random for a tie (line 7 of Algorithm 4)
        """
        best_bid = max(self.bids.values())
        candidates = [agent_id for agent_id, bid in self.bids.items() if bid == best_bid]
        return int(self._rng.choice(candidates))

    def update(self, has_token: bool) -> CycleResult:
        """
        One cycle: grow the tree, and change the plan only with the token (line 5 of Algorithm 4)
        """
        self.executor.begin_cycle()
        self.executor.grow(self.executor.samples_per_cycle)
        return self.end_cycle(has_token)

    def end_cycle(self, has_token: bool) -> CycleResult:
        """
        The end of a cycle, also used when the tree grows between messages (Asynchronous DMA-RRT)
        """
        result = self.executor.end_cycle(can_change_plan=has_token)
        if len(self.executor.committed_states) == 0 and len(self.executor.plan_nodes) == 1:
            self.waiting_time += self.executor.steps_per_cycle * self._controller.dt
        else:
            self.waiting_time = 0.0
        for state in np.vstack([result.executed_states.reshape(-1, 3), self.executor.state]):
            if self.planner.is_in_goal(Node(state, None, 0.0), self.goal_pos):
                self.num_goals_reached += 1
                self._goal_index = (self._goal_index + 1) % len(self.goals)
                self.executor.goal_pos = self.goals[self._goal_index]
                break
        return result

    def current_plan(self) -> MovingObstacle:
        """
        What this robot is going to do if its plan does not change, from the current time on
        """
        executor = self.executor
        dt = self._controller.dt
        committed = executor.committed_states[:, :2]
        end_of_committed = committed[-1] if len(committed) > 0 else executor.state[:2]
        # the plan starts at the root time, and the robot waits until then after the committed path
        num_waiting = max(int(round((self.planner.root_time - executor.time) / dt)) - len(committed), 0)
        segments = [
            executor.state[:2][np.newaxis],
            committed,
            np.repeat(end_of_committed[np.newaxis], num_waiting, axis=0),
        ]
        segments += [node.trajectory[:, :2] for node in executor.plan_nodes[1:]]
        return MovingObstacle(np.vstack(segments), executor.time, dt, self._robot_radius)

    def believed_positions(self, agent_id: int, times: np.ndarray) -> np.ndarray:
        """
        Where this robot believes another robot is at the given times
        :return: positions size: (N, 2)
        """
        return self._other_robots[agent_id].positions_at(times)

    def make_plan_message(self, winner: int) -> PlanMessage:
        """
        Message to broadcast after changing the plan (line 8 of Algorithm 4)
        """
        root = self.executor.plan_nodes[0]
        return PlanMessage(
            self.agent_id,
            root.state,
            self.planner.root_time,
            root.waypoint,
            [node.waypoint for node in self.executor.plan_nodes[1:]],
            winner,
        )

    def receive_plan(self, message: PlanMessage) -> None:
        """
        Simulate the other robot's trajectory along its waypoints and update the constraints (line 4 of Algorithm 5)
        """
        states = self.planner.simulate_waypoints(
            message.start_state, message.waypoints, from_pos=message.from_pos
        )
        self._other_robots[message.sender].replace_from(message.start_time, states)
