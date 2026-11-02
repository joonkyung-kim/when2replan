"""
    Closed-loop RRT: tree expansion of Algorithm 1 in the DMA-RRT paper (step 2)

    A node is a waypoint on the reference. The trajectory to a node is the closed loop simulation
    tracking the line from the parent's waypoint to the node's waypoint, so a path can be reproduced
    from the initial state and its waypoints only.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional
import numpy as np

from dma_rrt.collision import CollisionChecker
from dma_rrt.dynamics import PurePursuit
from dma_rrt.tree import Node, Tree


NO_GOAL_PENALTY = 1000.0  # [s] larger than the cost of any path


class CLRRTAttempt(NamedTuple):
    nearest_node: Node
    waypoint: np.ndarray
    trajectory: np.ndarray  # simulated states size: (N, 3)
    collision_pos: Optional[np.ndarray]  # first unsafe position on the trajectory, None if safe
    new_node: Optional[Node]  # None if this attempt failed


class CLRRTStep(NamedTuple):
    sample_pos: np.ndarray
    attempts: List[CLRRTAttempt]  # the closest node first, stops at the first success
    new_node: Optional[Node]  # None if all attempts failed


class CLRRT:
    def __init__(
        self,
        collision_checker: CollisionChecker,
        controller: PurePursuit,
        step_size: float = 2.0,
        goal_radius: float = 1.0,
        goal_sample_rate: float = 0.05,
        num_nearest: int = 5,
        time_to_go_weight: float = 2.0,
        select_ignoring_others: bool = False,
        grow_ignoring_others: bool = False,
        seed: int = 0,
    ) -> None:
        """
        :param collision_checker: collision checker of the map to plan on
        :param controller: closed loop model used to simulate the trajectories
        :param step_size: max length of the reference line of a new node [m]
        :param goal_radius: radius of the goal region [m]
        :param goal_sample_rate: probability of sampling the goal itself
        :param num_nearest: number of the closest nodes to try for a sample. With only the closest node,
            a node that cannot go anywhere because of its heading keeps taking the samples around it
        :param time_to_go_weight: weight of the straight line time to the goal in the cost estimate
        :param select_ignoring_others: select paths without the moving obstacles. The caller has to check them
        :param grow_ignoring_others: grow the tree without the moving obstacles, as in line 4 of Algorithm 6
        :param seed: seed of the planner's own random generator
        """
        self._collision_checker = collision_checker
        self._controller = controller
        self._step_size = step_size
        self._goal_radius = goal_radius
        self._goal_sample_rate = goal_sample_rate
        self._num_nearest = num_nearest
        self._time_to_go_weight = time_to_go_weight
        self._select_ignoring_others = select_ignoring_others
        self._grow_ignoring_others = grow_ignoring_others
        self._rng = np.random.default_rng(seed)
        self.tree: Tree = None
        self.root_time = 0.0

    def plan(
        self,
        start_state: np.ndarray,
        goal_pos: np.ndarray,
        max_iter: int = 5000,
        stop_at_goal: bool = True,
    ) -> Optional[Node]:
        """
        Grow a new tree from start_state and return the lowest cost node in the goal region
        :param start_state: start state (x, y, yaw)
        :param goal_pos: center of the goal region
        :param max_iter: max number of samples
        :param stop_at_goal: stop at the first node in the goal region, otherwise use all samples
        :return: node, None if no node reached the goal region
        """
        goal_pos = np.asarray(goal_pos, dtype=float)
        self.reset(start_state)

        best_node: Optional[Node] = None
        for _ in range(max_iter):
            node = self.step(goal_pos).new_node
            if node is None or not self.is_in_goal(node, goal_pos):
                continue

            if best_node is None or node.cost < best_node.cost:
                best_node = node
            if stop_at_goal:
                break
        return best_node

    def reset(self, start_state: np.ndarray, start_time: float = 0.0) -> None:
        """
        Start a new tree from start_state
        :param start_time: time the robot is at start_state [s]
        """
        self.tree = Tree(start_state)
        self.tree.root.waypoint = self.tree.root.state[:2]
        self.root_time = start_time

    def node_time(self, node: Node) -> float:
        """
        Time the robot arrives at node. Only the root's time is stored, so a robot waiting at the root
        delays all the paths after it
        """
        return self.root_time + node.cost - self.tree.root.cost

    def trajectory_times(self, start_time: float, num_states: int) -> np.ndarray:
        """
        Times of the states of a trajectory starting at start_time, which does not include the state at start_time
        """
        return start_time + self._controller.dt * np.arange(1, num_states + 1)

    def step(self, goal_pos: np.ndarray) -> CLRRTStep:
        """
        One expansion of the tree (Algorithm 1), with the closest nodes in order until one succeeds
        """
        sample_pos = self._sample(goal_pos)
        attempts: List[CLRRTAttempt] = []
        for nearest_node in self.tree.k_nearest(sample_pos, self._num_nearest):
            waypoint = self._get_waypoint(nearest_node, sample_pos)
            if waypoint is None:
                continue

            attempt = self._extend(nearest_node, waypoint)
            attempts.append(attempt)
            if attempt.new_node is not None:
                return CLRRTStep(sample_pos, attempts, attempt.new_node)
        return CLRRTStep(sample_pos, attempts, None)

    def is_in_goal(self, node: Node, goal_pos: np.ndarray) -> bool:
        return np.linalg.norm(node.state[:2] - goal_pos) <= self._goal_radius

    def best_node(self, goal_pos: np.ndarray) -> Node:
        """
        Get the end of the best path in the tree (line 12 of Algorithm 2)
        A path reaching the goal is always preferred, as in Sec. 8.2 of the paper. Otherwise use the estimated cost.
        A path has to end where the robot can stay safe for all future time (eq. 7).
        The root is only returned when there is no other node to end at, or the robot would never leave it.
        """
        positions = self.tree.positions
        costs = np.array([node.cost for node in self.tree.nodes])
        times = self.root_time + costs - self.tree.root.cost
        if self._select_ignoring_others:
            is_stop_safe = np.ones(len(positions), dtype=bool)
        else:
            is_stop_safe = self._collision_checker.stop_safe_mask(positions, times)
        dists = np.linalg.norm(positions - goal_pos, axis=1)

        is_candidate = is_stop_safe & (dists <= self._goal_radius)
        scores = costs
        if not is_candidate.any():
            is_candidate = is_stop_safe.copy()
            is_candidate[0] = False  # the root
            if not is_candidate.any():
                return self.tree.root
            scores = costs + self._time_to_go_weight * dists / self._controller.speed
        indices = np.flatnonzero(is_candidate)
        return self.tree.nodes[indices[np.argmin(scores[indices])]]

    def cost_estimate(self, node: Node, goal_pos: np.ndarray) -> float:
        """
        Estimated travel time to the goal through node, comparable with the selection of best_node
        The straight line is weighted because it is a lower bound of the time to go, which would make
        any path look worse than a shorter part of it.
        A path not reaching the goal has a penalty, since a path reaching the goal is always preferred. Without it,
        the optimistic straight line makes a stopped robot bid less than 0 for a winding path to its goal,
        and it never gets the token. Between two such paths, the penalty cancels out.
        """
        if self.is_in_goal(node, goal_pos):
            return node.cost
        time_to_go = np.linalg.norm(node.state[:2] - goal_pos) / self._controller.speed
        return node.cost + self._time_to_go_weight * time_to_go + NO_GOAL_PENALTY

    def finish_time_estimate(self, node: Node, goal_pos: np.ndarray) -> float:
        """
        cost_estimate as a time on the clock shared by the robots, since the costs of two trees are not comparable
        """
        return self.cost_estimate(node, goal_pos) - self.tree.root.cost + self.root_time

    def lazy_check(self, node: Node) -> bool:
        """
        Simulate the path to node again against the current constraints (Algorithm 3)
        The first node with an unsafe trajectory is removed with all its children
        :return: True if the whole path is still safe
        """
        for path_node in self.tree.nodes_to(node)[1:]:
            parent = path_node.parent
            trajectory, _ = self._controller.simulate_segment(
                parent.state, parent.waypoint, path_node.waypoint
            )
            times = None if self._select_ignoring_others else self.trajectory_times(self.node_time(parent), len(trajectory))
            if self._collision_checker.find_first_unsafe(trajectory[:, :2], times) is not None:
                self.tree.remove_subtree(path_node)
                return False
        return True

    def path_trajectory(self, node: Node) -> np.ndarray:
        """
        Get the whole trajectory from the root to node, including the root's state
        :return: states size: (N, 3)
        """
        nodes = self.tree.nodes_to(node)
        return np.vstack([nodes[0].state[np.newaxis]] + [n.trajectory for n in nodes[1:]])

    def simulate_waypoints(
        self,
        start_state: np.ndarray,
        waypoints: List[np.ndarray],
        from_pos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Reproduce a trajectory from the initial state and the waypoints
        :param start_state: initial state
        :param waypoints: waypoints after the initial state
        :param from_pos: where the reference starts, the position of start_state by default.
            A robot starting from a node has to use that node's waypoint
        :return: states including start_state size: (N, 3)
        """
        states = [np.asarray(start_state, dtype=float)[np.newaxis]]
        current = states[0][0]
        if from_pos is None:
            from_pos = current[:2]
        for waypoint in waypoints:
            segment, _ = self._controller.simulate_segment(current, from_pos, waypoint)
            states.append(segment)
            current = segment[-1]
            from_pos = waypoint
        return np.vstack(states)

    def _sample(self, goal_pos: np.ndarray) -> np.ndarray:
        """
        Sample from the free space as in the paper
        """
        if self._rng.random() < self._goal_sample_rate:
            return goal_pos

        x_min, x_max, y_min, y_max = self._collision_checker.area_limits
        while True:
            pos = self._rng.uniform([x_min, y_min], [x_max, y_max])
            if self._collision_checker.is_free(pos):
                return pos

    def _get_waypoint(self, nearest_node: Node, sample_pos: np.ndarray) -> Optional[np.ndarray]:
        """
        Get the new waypoint from the nearest node's waypoint toward sample_pos, at most step_size away
        :return: new waypoint, None if sample_pos is already the nearest node's waypoint
        """
        direction = sample_pos - nearest_node.waypoint
        dist = np.linalg.norm(direction)
        if dist == 0.0:
            return None
        return nearest_node.waypoint + min(dist, self._step_size) * direction / dist

    def _extend(self, nearest_node: Node, waypoint: np.ndarray) -> CLRRTAttempt:
        """
        Simulate the closed loop system along the line to waypoint, and add a node if the whole trajectory is feasible
        """
        trajectory, is_reached = self._controller.simulate_segment(
            nearest_node.state, nearest_node.waypoint, waypoint
        )
        times = None if self._grow_ignoring_others else self.trajectory_times(self.node_time(nearest_node), len(trajectory))
        collision_pos = self._collision_checker.find_first_unsafe(trajectory[:, :2], times)
        # a node needs at least one state of its own
        if collision_pos is not None or not is_reached or len(trajectory) == 0:
            return CLRRTAttempt(nearest_node, waypoint, trajectory, collision_pos, None)

        edge_cost = len(trajectory) * self._controller.dt  # travel time as in the paper
        new_node = self.tree.add_node(trajectory[-1], nearest_node, edge_cost)
        new_node.waypoint = waypoint
        new_node.trajectory = trajectory
        return CLRRTAttempt(nearest_node, waypoint, trajectory, None, new_node)
