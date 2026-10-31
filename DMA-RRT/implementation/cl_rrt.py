"""
    Closed-loop RRT: tree expansion of Algorithm 1 in the DMA-RRT paper (step 2)

    A node is a waypoint on the reference. The trajectory to a node is the closed loop simulation
    tracking the line from the parent's waypoint to the node's waypoint, so a path can be reproduced
    from the initial state and its waypoints only.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional
import numpy as np

from collision import CollisionChecker
from dynamics import PurePursuit
from tree import Node, Tree


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
        :param seed: seed of the planner's own random generator
        """
        self._collision_checker = collision_checker
        self._controller = controller
        self._step_size = step_size
        self._goal_radius = goal_radius
        self._goal_sample_rate = goal_sample_rate
        self._num_nearest = num_nearest
        self._time_to_go_weight = time_to_go_weight
        self._rng = np.random.default_rng(seed)
        self.tree: Tree = None

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

    def reset(self, start_state: np.ndarray) -> None:
        """
        Start a new tree from start_state
        """
        self.tree = Tree(start_state)
        self.tree.root.waypoint = self.tree.root.state[:2]

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
        The root is only returned when the tree has no other node, or the robot would never leave it.
        """
        goal_nodes = [node for node in self.tree.nodes if self.is_in_goal(node, goal_pos)]
        if len(goal_nodes) > 0:
            return min(goal_nodes, key=lambda node: node.cost)

        if len(self.tree) == 1:
            return self.tree.root
        return min(self.tree.nodes[1:], key=lambda node: self.cost_estimate(node, goal_pos))

    def cost_estimate(self, node: Node, goal_pos: np.ndarray) -> float:
        """
        Estimated travel time to the goal through node
        The straight line is weighted because it is a lower bound of the time to go, which would make
        any path look worse than a shorter part of it
        """
        if self.is_in_goal(node, goal_pos):
            return node.cost
        time_to_go = np.linalg.norm(node.state[:2] - goal_pos) / self._controller.speed
        return node.cost + self._time_to_go_weight * time_to_go

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
            if self._collision_checker.find_first_unsafe(trajectory[:, :2]) is not None:
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

    def simulate_waypoints(self, start_state: np.ndarray, waypoints: List[np.ndarray]) -> np.ndarray:
        """
        Reproduce a trajectory from the initial state and the waypoints
        :param start_state: initial state, the reference starts from its position
        :param waypoints: waypoints after the initial state
        :return: states including start_state size: (N, 3)
        """
        states = [np.asarray(start_state, dtype=float)[np.newaxis]]
        current = states[0][0]
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
        collision_pos = self._collision_checker.find_first_unsafe(trajectory[:, :2])
        # a node needs at least one state of its own
        if collision_pos is not None or not is_reached or len(trajectory) == 0:
            return CLRRTAttempt(nearest_node, waypoint, trajectory, collision_pos, None)

        edge_cost = len(trajectory) * self._controller.dt  # travel time as in the paper
        new_node = self.tree.add_node(trajectory[-1], nearest_node, edge_cost)
        new_node.waypoint = waypoint
        new_node.trajectory = trajectory
        return CLRRTAttempt(nearest_node, waypoint, trajectory, None, new_node)
