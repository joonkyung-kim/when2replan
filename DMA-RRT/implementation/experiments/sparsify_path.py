"""
    Can a when2replan Dijkstra path be represented as a sparse waypoint graph like an RRT plan?

    The dense Dijkstra path is reduced in two ways and checked against the obstacles:
    RDP (only geometric) and a line-of-sight shortcut (each edge checked with the planner's safe distance).
    The line-of-sight result is stored in the Tree of an RRT plan. The path is not used for navigation here,
    see demos/demo_sparse_tracking.py for a robot tracking it.

    Run from DMA-RRT/implementation:
    python3 -m experiments.sparsify_path
    python3 -m experiments.sparsify_path --log ../../log/run_multi_robot/DijkstraxDWA/16pillar/4robot/0/rl_based_replan/pkl/rl_based_replan_robot0_data.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from navigation_stack_py.utils import MapHandler
from navigation_stack_py.global_planner.djikstra_planner import DijkstraPlanner

from dma_rrt.collision import CollisionChecker, REPO_DIR, SIXTEEN_PILLAR_MAP
from dma_rrt.sparse_path import line_of_sight, max_deviation, path_length, rdp, to_tree

ROBOT_RADIUS = 1.0
INFLATION_RADIUS = 0.1
SPEED = 2.0  # [m/s] maximum speed of when2replan's DWA, only to turn edge lengths into times
DEFAULT_LOG = str(
    REPO_DIR / "log/run_multi_robot/DijkstraxDWA/16pillar/4robot/0/rl_based_replan/pkl/rl_based_replan_robot0_data.pkl"
)


def plan_on_static_map(start: List[float], goal: List[float]) -> np.ndarray:
    """
    Dijkstra path of when2replan on the static map
    :return: path size: (N, 2)
    """
    static_map = MapHandler()
    static_map.load_map_from_file(str(REPO_DIR / SIXTEEN_PILLAR_MAP))
    params = SimpleNamespace(
        robot_radius=ROBOT_RADIUS, inflation_radius=INFLATION_RADIUS, dijkstra_config={"connectedness": 8}
    )
    planner = DijkstraPlanner(params)
    planner.set_costmap(static_map)
    np.random.seed(0)  # pymap2d breaks ties between equal-cost cells with the numpy global random state
    _, path = planner.make_plan(np.array(start, dtype=float), np.array(goal, dtype=float))
    return path[0]  # make_plan returns size: (1, N, 2)


def logged_replan(log_path: str):
    """
    The replanned path of a multi robot run with the largest detour, and the other robots of that frame
    :return: path size: (N, 2), centers of the other robots, frame index
    """
    import navigation_stack_py.gym_env  # noqa: F401, loads navigation_stack_py.common in the right order for pickle

    with open(log_path, "rb") as f:
        frames = pickle.load(f)
    best = None
    for k, frame in enumerate(frames):
        path = frame.get("global_path")
        if not frame["is_replan"] or path is None or len(path) < 20:
            continue
        ratio = path_length(path) / max(np.linalg.norm(path[-1] - path[0]), 1e-6)
        if best is None or ratio > best[0]:
            best = (ratio, k)
    frame = frames[best[1]]
    others = [other["robot_state"].pos for other in frame.get("other_robots", [])]
    return frame["global_path"], others, best[1]


def run_case(ax, title: str, dense: np.ndarray, circles: List[np.ndarray], max_edge: float, eps: float) -> None:
    checker = CollisionChecker(SIXTEEN_PILLAR_MAP, robot_radius=ROBOT_RADIUS, inflation_radius=INFLATION_RADIUS)
    for center in circles:
        checker.add_circle(np.asarray(center), ROBOT_RADIUS)  # another robot

    def is_edge_safe(a: np.ndarray, b: np.ndarray) -> bool:
        return checker.find_edge_collision(a, b) is None

    def num_unsafe(points: np.ndarray) -> int:
        return sum(not is_edge_safe(a, b) for a, b in zip(points[:-1], points[1:]))

    rdp_points = dense[rdp(dense, eps)]
    los_points = dense[line_of_sight(dense, is_edge_safe, max_edge)]
    tree = to_tree(los_points, SPEED)

    print(f"\n== {title}")
    print(f"{'':26s}{'points':>7s}{'length [m]':>12s}{'max dev [m]':>13s}{'unsafe edges':>14s}{'bytes (f32)':>13s}")
    rows = [
        ("Dijkstra (dense)", dense, "-"),
        (f"RDP eps={eps} m", rdp_points, num_unsafe(rdp_points)),
        (f"LOS max_edge={max_edge} m", los_points, num_unsafe(los_points)),
    ]
    for name, points, unsafe in rows:
        print(
            f"{name:26s}{len(points):7d}{path_length(points):12.2f}{max_deviation(dense, points):13.2f}"
            f"{str(unsafe):>14s}{len(points) * 2 * 4:13d}"
        )
    print(f"LOS as the tree of an RRT plan: {len(tree)} nodes, (x, y, yaw) and time at {SPEED} m/s")
    for node in tree.nodes:
        print("    ({:5.2f}, {:5.2f}, {:5.2f})  t = {:5.2f} s".format(*node.state, node.cost))

    checker.draw(ax)
    for center in circles:
        ax.add_patch(plt.Circle(center, ROBOT_RADIUS, color="tab:red", alpha=0.5))
    ax.plot(dense[:, 0], dense[:, 1], ".", ms=2, color="gray", label=f"Dijkstra ({len(dense)} points)")
    ax.plot(rdp_points[:, 0], rdp_points[:, 1], "-s", ms=4, lw=1, color="tab:orange", label=f"RDP ({len(rdp_points)})")
    ax.plot(los_points[:, 0], los_points[:, 1], "-o", ms=6, lw=2, color="tab:blue", label=f"LOS graph ({len(los_points)})")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_edge", type=float, default=5.0, help="maximum edge length of the LOS graph [m]")
    parser.add_argument("--eps", type=float, default=0.2, help="tolerance of RDP [m]")
    parser.add_argument("--log", type=str, default=DEFAULT_LOG, help="visualized data pkl of a multi robot run")
    parser.add_argument("--save", type=str, default="outputs/when2replan_sparse_path/sparse_path.png")
    args = parser.parse_args()

    cases = [
        ("robot0 (1, 1) -> (24, 24), static map", plan_on_static_map([1.0, 1.0], [24.0, 24.0]), []),
        ("robot4 (12.5, 1) -> (12.5, 24), static map", plan_on_static_map([12.5, 1.0], [12.5, 24.0]), []),
    ]
    log: Optional[str] = args.log if os.path.exists(args.log) else None
    if log is not None:
        path, others, k = logged_replan(log)
        cases.append((f"replan of a 4 robot run, frame {k}", path, others))
    else:
        print(f"No multi robot log at {args.log}, run scripts/run_multi_robot.py for the third case")

    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 6.3))
    for ax, (title, dense, circles) in zip(np.atleast_1d(axes), cases):
        run_case(ax, title, dense, circles, args.max_edge, args.eps)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    fig.savefig(args.save, dpi=90)
    print(f"\nSaved {args.save}")


if __name__ == "__main__":
    main()
