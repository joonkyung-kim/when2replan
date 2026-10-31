"""
    Basic RRT on a 2D map with static obstacles (step 1)
"""

from __future__ import annotations

from typing import NamedTuple, Optional
import numpy as np

from collision import CollisionChecker
from tree import Node, Tree


class RRTStep(NamedTuple):
    sample_pos: np.ndarray
    nearest_node: Node
    steer_pos: Optional[np.ndarray]  # None if sample_pos is already in the tree
    collision_pos: Optional[np.ndarray]  # first unsafe point on the edge, None if safe
    new_node: Optional[Node]  # None if nothing was added to the tree


class RRT:
    def __init__(
        self,
        collision_checker: CollisionChecker,
        step_size: float = 0.5,
        goal_radius: float = 0.5,
        goal_sample_rate: float = 0.05,
        seed: int = 0,
    ) -> None:
        """
        :param collision_checker: collision checker of the map to plan on
        :param step_size: max distance between a node and its parent [m]
        :param goal_radius: radius of the goal region [m]
        :param goal_sample_rate: probability of sampling the goal itself
        :param seed: seed of the planner's own random generator
        """
        self._collision_checker = collision_checker
        self._area_limits = collision_checker.area_limits
        self._step_size = step_size
        self._goal_radius = goal_radius
        self._goal_sample_rate = goal_sample_rate
        self._rng = np.random.default_rng(seed)
        self.tree: Tree = None

    def plan(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        max_iter: int = 5000,
        stop_at_goal: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Grow a new tree from start_pos and return the lowest cost path to the goal region
        :param start_pos: start position
        :param goal_pos: center of the goal region
        :param max_iter: max number of samples
        :param stop_at_goal: stop at the first node in the goal region, otherwise use all samples
        :return: path size: (N, 2), None if no node reached the goal region
        """
        goal_pos = np.asarray(goal_pos, dtype=float)
        self.reset(start_pos)

        best_node: Optional[Node] = None
        for _ in range(max_iter):
            node = self.step(goal_pos).new_node
            if node is None or not self.is_in_goal(node, goal_pos):
                continue

            if best_node is None or node.cost < best_node.cost:
                best_node = node
            if stop_at_goal:
                break

        if best_node is None:
            return None
        return self.tree.path_to(best_node)

    def reset(self, start_pos: np.ndarray) -> None:
        """
        Start a new tree from start_pos
        """
        self.tree = Tree(start_pos)

    def step(self, goal_pos: np.ndarray) -> RRTStep:
        """
        One iteration of RRT: sample, find the nearest node, steer, check collision, and add a node
        """
        sample_pos = self._sample(goal_pos)
        nearest_node = self.tree.nearest(sample_pos)

        steer_pos = self._steer(nearest_node, sample_pos)
        if steer_pos is None:
            return RRTStep(sample_pos, nearest_node, None, None, None)

        collision_pos = self._collision_checker.find_edge_collision(
            nearest_node.state, steer_pos
        )
        if collision_pos is not None:
            return RRTStep(sample_pos, nearest_node, steer_pos, collision_pos, None)

        edge_cost = np.linalg.norm(steer_pos - nearest_node.state)
        new_node = self.tree.add_node(steer_pos, nearest_node, edge_cost)
        return RRTStep(sample_pos, nearest_node, steer_pos, None, new_node)

    def is_in_goal(self, node: Node, goal_pos: np.ndarray) -> bool:
        return np.linalg.norm(node.state - goal_pos) <= self._goal_radius

    def _sample(self, goal_pos: np.ndarray) -> np.ndarray:
        if self._rng.random() < self._goal_sample_rate:
            return goal_pos

        x_min, x_max, y_min, y_max = self._area_limits
        return self._rng.uniform([x_min, y_min], [x_max, y_max])

    def _steer(self, nearest_node: Node, sample_pos: np.ndarray) -> Optional[np.ndarray]:
        """
        Move from nearest_node toward sample_pos, at most step_size away
        :return: new position, None if sample_pos is already in the tree
        """
        direction = sample_pos - nearest_node.state
        dist = np.linalg.norm(direction)
        if dist == 0.0:
            return None

        return nearest_node.state + min(dist, self._step_size) * direction / dist
