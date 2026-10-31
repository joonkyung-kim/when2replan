"""
    Grow an RRT on a map and draw the tree

    Run from DMA-RRT/implementation:
    python3 demo_rrt.py
    python3 demo_rrt.py --seed 1 --use_all_samples --save rrt.png
    python3 demo_rrt.py --video rrt.mp4
    python3 demo_rrt.py --map maps/white/white/map.yaml  # without obstacles
"""

from __future__ import annotations

import argparse
from typing import List, Optional
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

from collision import CollisionChecker, SIXTEEN_PILLAR_MAP
from rrt import RRT
from tree import Tree


def get_edges(tree: Tree) -> List[np.ndarray]:
    """
    Get the edges in the order they were added to the tree
    :return: list of edges size: (2, 2)
    """
    return [
        np.array([node.parent.state, node.state])
        for node in tree.nodes
        if node.parent is not None
    ]


def save_video(
    fig,
    edge_lines: LineCollection,
    path_line,
    edges: List[np.ndarray],
    path: Optional[np.ndarray],
    video_path: str,
    fps: int,
    duration: float,
) -> None:
    """
    Replay the tree growth in the order of the nodes, then hold the last frame with the path
    :param duration: time to replay the growth [s]
    """
    # H.264 with yuv420p to be playable in VS Code and browsers
    writer = imageio.get_writer(
        video_path, fps=fps, codec="libx264", pixelformat="yuv420p"
    )

    def write_frame() -> None:
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())
        writer.append_data(frame[:, :, :3])

    num_frames = max(int(fps * duration), 1)
    edge_nums = np.linspace(0, len(edges), num_frames + 1, dtype=int)[1:]
    path_line.set_data([], [])
    for edge_num in edge_nums:
        edge_lines.set_segments(edges[:edge_num])
        write_frame()

    if path is not None:
        path_line.set_data(path[:, 0], path[:, 1])
    for _ in range(2 * fps):
        write_frame()

    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default=SIXTEEN_PILLAR_MAP, help="map yaml file, relative to the repo root")
    parser.add_argument("--start", type=float, nargs=2, default=[1.0, 1.0])
    parser.add_argument("--goal", type=float, nargs=2, default=[24.0, 24.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument(
        "--use_all_samples",
        action="store_true",
        help="keep growing after the first node reached the goal region",
    )
    parser.add_argument("--save", type=str, default=None, help="save the figure here instead of showing it")
    parser.add_argument("--video", type=str, default=None, help="save the tree growth as an mp4 here instead of showing it")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=10.0, help="time to replay the tree growth [s]")
    args = parser.parse_args()

    start_pos = np.array(args.start)
    goal_pos = np.array(args.goal)
    goal_radius = 0.5

    collision_checker = CollisionChecker(args.map)
    if not collision_checker.is_free(start_pos):
        parser.error(f"start {args.start} is too close to an obstacle")
    if not collision_checker.is_free(goal_pos):
        parser.error(f"goal {args.goal} is too close to an obstacle")

    planner = RRT(collision_checker, goal_radius=goal_radius, seed=args.seed)
    path = planner.plan(
        start_pos,
        goal_pos,
        max_iter=args.max_iter,
        stop_at_goal=not args.use_all_samples,
    )

    if path is None:
        print(f"nodes: {len(planner.tree)}, goal not reached")
    else:
        length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        print(f"nodes: {len(planner.tree)}, path length: {length:.2f} m")

    edges = get_edges(planner.tree)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    collision_checker.draw(ax)
    edge_lines = LineCollection(edges, colors="gray", linewidths=0.6)
    ax.add_collection(edge_lines)
    (path_line,) = ax.plot([], [], color="tab:blue", linewidth=2.0, label="path")
    if path is not None:
        path_line.set_data(path[:, 0], path[:, 1])
    ax.plot(*start_pos, "o", color="tab:red", label="start")
    ax.add_patch(plt.Circle(goal_pos, goal_radius, color="tab:green", alpha=0.5, label="goal region"))
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    handles, _ = ax.get_legend_handles_labels()
    handles += [Patch(color="dimgray", label="obstacle"), Patch(color="mistyrose", label="too close")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=5, fontsize=8)

    if args.save is not None:
        fig.savefig(args.save, dpi=150)
    if args.video is not None:
        save_video(fig, edge_lines, path_line, edges, path, args.video, args.fps, args.duration)
    if args.save is None and args.video is None:
        plt.show()


if __name__ == "__main__":
    main()
