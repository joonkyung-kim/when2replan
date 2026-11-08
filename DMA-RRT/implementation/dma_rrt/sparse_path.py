"""
    Sparse waypoint graph from a dense when2replan Dijkstra path, like the plan of an RRT

    The Dijkstra path of when2replan is a chain of grid cells (0.1 m apart). Here it is reduced to a few waypoints
    joined by straight edges that are checked against the obstacles, and stored in the Tree of an RRT plan.
    A local planner such as DWA searches its sub-goal along the points of the reference path, so the sparse path is
    resampled back to the grid spacing before it is tracked.
"""

from __future__ import annotations

from typing import Callable
import numpy as np

from dma_rrt.tree import Tree


def path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def rdp(path: np.ndarray, eps: float) -> np.ndarray:
    """
    Ramer-Douglas-Peucker: only geometric, the obstacles are not checked
    :param path: dense path size: (N, 2)
    :param eps: maximum distance of a removed point from the simplified path [m]
    :return: indices of the kept points
    """
    keep = np.zeros(len(path), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(path) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = path[i], path[j]
        ab = b - a
        seg = path[i + 1 : j] - a
        norm = np.linalg.norm(ab)
        if norm < 1e-9:
            dists = np.linalg.norm(seg, axis=1)
        else:
            dists = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / norm
        k = int(np.argmax(dists))
        if dists[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return np.flatnonzero(keep)


def line_of_sight(
    path: np.ndarray, is_edge_safe: Callable[[np.ndarray, np.ndarray], bool], max_edge: float
) -> np.ndarray:
    """
    Greedy shortcut: from each kept point, jump to the farthest point of the path that a safe straight edge
    not longer than max_edge reaches. Neighbouring points of the path are always connected
    :param path: dense path size: (N, 2)
    :param is_edge_safe: True if the straight edge between two positions is collision free
    :param max_edge: maximum edge length, like the steering distance of an RRT [m]
    :return: indices of the kept points
    """
    keep = [0]
    i = 0
    while i < len(path) - 1:
        best = i + 1
        for j in range(len(path) - 1, i + 1, -1):
            if np.linalg.norm(path[j] - path[i]) > max_edge:
                continue
            if is_edge_safe(path[i], path[j]):
                best = j
                break
        keep.append(best)
        i = best
    return np.array(keep)


def map_edge_checker(cost_map, safe_dist: float) -> Callable[[np.ndarray, np.ndarray], bool]:
    """
    Edge check on a when2replan MapHandler, e.g. the static map with the scan the global planner uses
    :param safe_dist: robot radius + inflation radius [m]
    """
    step = cost_map.get_resolution()

    def is_edge_safe(a: np.ndarray, b: np.ndarray) -> bool:
        num_points = int(np.ceil(np.linalg.norm(b - a) / step)) + 1
        points = np.linspace(a, b, num_points)
        return bool(np.all(cost_map.get_obs_dist_array(points) >= safe_dist))

    return is_edge_safe


def max_deviation(dense: np.ndarray, sparse: np.ndarray) -> float:
    """
    Maximum distance from a point of the dense path to the sparse polyline [m]
    """
    worst = 0.0
    for p in dense:
        best = np.inf
        for a, b in zip(sparse[:-1], sparse[1:]):
            ab = b - a
            t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
            best = min(best, np.linalg.norm(p - (a + t * ab)))
        worst = max(worst, best)
    return worst


def resample(sparse: np.ndarray, spacing: float) -> np.ndarray:
    """
    Points along the sparse polyline, spacing apart, with every waypoint kept
    :return: dense path size: (M, 2)
    """
    points = [sparse[0]]
    for a, b in zip(sparse[:-1], sparse[1:]):
        num = max(int(np.ceil(np.linalg.norm(b - a) / spacing)), 1)
        t = np.linspace(0.0, 1.0, num + 1)[1:, np.newaxis]
        points.extend(a + t * (b - a))
    return np.array(points)


def to_tree(sparse: np.ndarray, speed: float) -> Tree:
    """
    Sparse waypoints as a chain in the tree of an RRT plan. The state is (x, y, yaw) with the yaw of the edge,
    and the cost is the travel time at a constant speed, without the time to turn. node.trajectory stays None,
    since the robot tracking this path is not the closed loop model of CL-RRT
    :param speed: speed along the edges [m/s]
    """
    yaw = np.arctan2(sparse[1, 1] - sparse[0, 1], sparse[1, 0] - sparse[0, 0]) if len(sparse) > 1 else 0.0
    tree = Tree(np.array([sparse[0, 0], sparse[0, 1], yaw]))
    node = tree.root
    for a, b in zip(sparse[:-1], sparse[1:]):
        yaw = np.arctan2(b[1] - a[1], b[0] - a[0])
        node = tree.add_node(np.array([b[0], b[1], yaw]), node, np.linalg.norm(b - a) / speed)
        node.waypoint = b
    return tree
