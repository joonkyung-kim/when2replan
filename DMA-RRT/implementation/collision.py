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

    def is_free(self, pos: np.ndarray) -> bool:
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

    def find_first_unsafe(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the first point too close to an obstacle or out of the map
        :param points: positions size: (N, 2)
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
