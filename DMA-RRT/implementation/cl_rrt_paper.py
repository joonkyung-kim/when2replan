"""
    CL-RRT following Algorithm 1 of the DMA-RRT paper as written

    Differences from CLRRT in cl_rrt.py:
    - samples are uniform in the free space, the goal is not sampled on purpose
    - only the nearest node is tried
    - the reference goes all the way to the sample, its length is not limited
    - the simulation stops at the first unsafe state, and the part before it is kept as a new node

    The paper does not specify the following, so they are decided here:
    - The node of a stopped simulation is the point on the reference next to the last safe state.
      The robot turns differently when the reference ends there, so the trajectory to that node is
      simulated again, and the node is dropped if that trajectory is unsafe. This keeps the property
      that a path is reproduced from its waypoints.
    - The while loop of Algorithm 1 has no time limit, but the robot can circle a sample it cannot
      reach forever. The simulation gives up after a time limit and no node is added.
"""

from __future__ import annotations

from typing import List
import numpy as np

from cl_rrt import CLRRT, CLRRTAttempt
from collision import CollisionChecker
from dynamics import PurePursuit
from tree import Node


class PaperCLRRT(CLRRT):
    def __init__(
        self,
        collision_checker: CollisionChecker,
        controller: PurePursuit,
        goal_radius: float = 1.0,
        goal_sample_rate: float = 0.0,
        num_nearest: int = 1,
        seed: int = 0,
    ) -> None:
        """
        goal_sample_rate and num_nearest are only changed from the paper's values to compare their effects
        """
        super().__init__(
            collision_checker,
            controller,
            step_size=np.inf,
            goal_radius=goal_radius,
            goal_sample_rate=goal_sample_rate,
            num_nearest=num_nearest,
            seed=seed,
        )

    def _extend(self, nearest_node: Node, waypoint: np.ndarray) -> CLRRTAttempt:
        """
        Lines 3-15 of Algorithm 1
        """
        controller = self._controller
        states: List[np.ndarray] = []
        current = nearest_node.state
        start_time = self.node_time(nearest_node)
        is_reached = False
        unsafe_state = None
        for step in range(controller.max_segment_steps(nearest_node.waypoint, waypoint)):
            target_pos, is_reached = controller.lookahead_point(
                current[:2], nearest_node.waypoint, waypoint
            )
            if is_reached:
                break
            v, delta_v = controller.control(current, target_pos)
            next_state = controller.model.step(current, v, delta_v, controller.dt)
            time = np.array([start_time + (step + 1) * controller.dt])
            if self._collision_checker.find_first_unsafe(next_state[:2][np.newaxis], time) is not None:
                unsafe_state = next_state
                break
            states.append(next_state)
            current = next_state

        if is_reached and len(states) > 0:
            return self._add_node(nearest_node, waypoint, np.array(states))

        failed_states = list(states)
        collision_pos = None
        if unsafe_state is not None:
            failed_states.append(unsafe_state)
            collision_pos = unsafe_state[:2]
        failed = CLRRTAttempt(
            nearest_node, waypoint, np.array(failed_states).reshape(-1, 3), collision_pos, None
        )
        if unsafe_state is None or len(states) == 0:
            return failed

        # stopped by an obstacle: keep the part before it
        final_waypoint = self._get_final_reference(nearest_node.waypoint, waypoint, states[-1][:2])
        if final_waypoint is None:
            return failed
        trajectory, is_reached = controller.simulate_segment(
            nearest_node.state, nearest_node.waypoint, final_waypoint
        )
        times = self.trajectory_times(start_time, len(trajectory))
        is_unsafe = self._collision_checker.find_first_unsafe(trajectory[:, :2], times) is not None
        if is_unsafe or not is_reached or len(trajectory) == 0:
            return failed
        return self._add_node(nearest_node, final_waypoint, trajectory)

    def _get_final_reference(
        self, from_pos: np.ndarray, to_pos: np.ndarray, position: np.ndarray
    ):
        """
        Get the point on the line from from_pos to to_pos next to position, None if it is behind from_pos
        """
        direction = (to_pos - from_pos) / np.linalg.norm(to_pos - from_pos)
        progress = np.dot(position - from_pos, direction)
        if progress <= 0.0:
            return None
        return from_pos + progress * direction

    def _add_node(
        self, nearest_node: Node, waypoint: np.ndarray, trajectory: np.ndarray
    ) -> CLRRTAttempt:
        edge_cost = len(trajectory) * self._controller.dt
        new_node = self.tree.add_node(trajectory[-1], nearest_node, edge_cost)
        new_node.waypoint = waypoint
        new_node.trajectory = trajectory
        return CLRRTAttempt(nearest_node, waypoint, trajectory, None, new_node)
