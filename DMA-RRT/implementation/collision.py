"""
    Collision check of the robot against the static obstacles of a map
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from matplotlib.colors import ListedColormap

from navigation_stack_py.utils import MapHandler

REPO_DIR = Path(__file__).resolve().parents[2]
SIXTEEN_PILLAR_MAP = "maps/sixteen_pillar/sixteen_pillar/map.yaml"


class MovingObstacle:
    """
    Time-parameterized obstacle of Sec. 7.1. It stays at the first state before its start,
    and stops at the last state after its end, which is the safe terminal state of eq. 7-8
    """

    def __init__(self, states: np.ndarray, start_time: float, dt: float, radius: float) -> None:
        """
        :param states: states at start_time, start_time + dt, ... size: (N, 2 or more)
        :param start_time: time of the first state [s]
        :param dt: time between the states [s]
        :param radius: radius of the obstacle [m]
        """
        self.positions = states[:, :2]
        self.start_time = start_time
        self.dt = dt
        self.radius = radius

    def _indices(self, times: np.ndarray) -> np.ndarray:
        indices = np.round((times - self.start_time) / self.dt).astype(int)
        return np.clip(indices, 0, len(self.positions) - 1)

    def positions_at(self, times: np.ndarray) -> np.ndarray:
        """
        :param times: times size: (N,)
        :return: positions at the times size: (N, 2)
        """
        return self.positions[self._indices(times)]

    def positions_from(self, time: float) -> np.ndarray:
        """
        All positions from time on, including where it stops
        :return: positions size: (N, 2)
        """
        return self.positions[self._indices(np.array([time]))[0] :]


class CollisionChecker:
    def __init__(
        self,
        map_path: str = SIXTEEN_PILLAR_MAP,
        robot_radius: float = 1.0,
        inflation_radius: float = 0.1,
    ) -> None:
        """
        The robot is treated as a point that has to stay safe_dist away from the obstacles.
        The map has no wall on its border, so only the robot's center is kept in the map.
        :param map_path: map yaml file, relative to the repo root
        :param robot_radius: radius of the robot [m]
        :param inflation_radius: margin added to robot_radius [m]
        """
        self._map = MapHandler()
        self._map.load_map_from_file(str(REPO_DIR / map_path))
        self.safe_dist = robot_radius + inflation_radius
        self.check_interval = self._map.get_resolution()
        self.area_limits = self._map.get_area_limits()
        self.circles: List[Tuple[np.ndarray, float]] = []  # obstacles not in the map: (center, radius)
        self.moving_obstacles: List[MovingObstacle] = []

    def add_circle(self, center: np.ndarray, radius: float) -> None:
        """
        Add a circular obstacle that is not in the map
        """
        self.circles.append((np.asarray(center, dtype=float), radius))

    def obstacle_dist(self, pos: np.ndarray) -> float:
        """
        Distance from pos to the nearest obstacle, 0 or less inside an obstacle
        NOTE: MapHandler clips a position out of the map to the map border
        """
        return float(self.obstacle_dists(np.asarray(pos, dtype=float)[np.newaxis])[0])

    def obstacle_dists(self, points: np.ndarray) -> np.ndarray:
        """
        Distances from points size: (N, 2) to the nearest obstacle
        """
        dists = self._map.get_obs_dist_array(points)
        for center, radius in self.circles:
            dists = np.minimum(dists, np.linalg.norm(points - center, axis=1) - radius)
        return dists

    def moving_obstacle_dists(self, points: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Distances to the nearest moving obstacle at the given times
        :param points: positions size: (N, 2)
        :param times: time of each position size: (N,)
        :return: distances size: (N,), inf without moving obstacles
        """
        dists = np.full(len(points), np.inf)
        for obstacle in self.moving_obstacles:
            obstacle_dists = np.linalg.norm(points - obstacle.positions_at(times), axis=1)
            dists = np.minimum(dists, obstacle_dists - obstacle.radius)
        return dists

    def is_stop_safe(self, pos: np.ndarray, time: float) -> bool:
        """
        Check if the robot can stop at pos from time on and stay safe for all future time (eq. 7)
        A moving obstacle can come to this position later, even if it is free when the robot arrives
        """
        for obstacle in self.moving_obstacles:
            dists = np.linalg.norm(obstacle.positions_from(time) - pos, axis=1)
            if dists.min() - obstacle.radius < self.safe_dist:
                return False
        return True

    def is_free(self, pos: np.ndarray) -> bool:
        """
        Check against the obstacles that do not move
        """
        return self.obstacle_dist(pos) >= self.safe_dist

    def edge_points(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """
        Points to check on an edge, including both ends
        :return: points size: (N, 2)
        """
        dist = np.linalg.norm(to_pos - from_pos)
        num_points = int(np.ceil(dist / self.check_interval)) + 1
        return np.linspace(from_pos, to_pos, num_points)

    def find_edge_collision(
        self, from_pos: np.ndarray, to_pos: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Find the first unsafe point on an edge
        :return: first unsafe point from from_pos, None if the whole edge is safe
        """
        return self.find_first_unsafe(self.edge_points(from_pos, to_pos))

    def find_first_unsafe(
        self, points: np.ndarray, times: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Find the first point too close to an obstacle or out of the map
        :param points: positions size: (N, 2)
        :param times: time of each position size: (N,), moving obstacles are only checked with this
        :return: first unsafe point, None if all points are safe
        """
        if len(points) == 0:
            return None

        x_min, x_max, y_min, y_max = self.area_limits
        is_out = (
            (points[:, 0] < x_min)
            | (points[:, 0] > x_max)
            | (points[:, 1] < y_min)
            | (points[:, 1] > y_max)
        )
        is_unsafe = is_out | (self.obstacle_dists(points) < self.safe_dist)
        if times is not None:
            is_unsafe |= self.moving_obstacle_dists(points, times) < self.safe_dist
        unsafe_indices = np.flatnonzero(is_unsafe)
        if len(unsafe_indices) == 0:
            return None
        return points[unsafe_indices[0]]

    def draw(self, ax) -> None:
        """
        Draw the obstacles in dark and the region the robot's center must not enter in light red
        """
        layers = np.zeros(self._map.get_image_size(), dtype=int)
        layers[self._map.get_map_as_np("sdf") < self.safe_dist] = 1
        layers[self._map.get_map_as_np("occupancy") > 0] = 2
        ax.imshow(
            layers.T,
            origin="lower",
            extent=self.area_limits,
            cmap=ListedColormap(["white", "mistyrose", "dimgray"]),
            vmin=0,
            vmax=2,
            interpolation="nearest",
        )
