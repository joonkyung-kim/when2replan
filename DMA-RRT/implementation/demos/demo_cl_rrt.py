"""
    Grow a CL-RRT on a map and draw the tree of simulated trajectories

    With --video, the first samples show the robot simulated along the reference, then the video
    speeds up, and at the end the robot drives along the found path.

    Run from DMA-RRT/implementation:
    python3 -m demos.demo_cl_rrt
    python3 -m demos.demo_cl_rrt --seed 1 --save outputs/cl_rrt.png
    python3 -m demos.demo_cl_rrt --video outputs/cl_rrt.mp4
    python3 -m demos.demo_cl_rrt --paper --video outputs/cl_rrt_paper.mp4  # Algorithm 1 of the paper as written
"""

from __future__ import annotations

import argparse
from typing import List, Optional
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from dma_rrt.cl_rrt import CLRRT, CLRRTAttempt, CLRRTStep
from dma_rrt.cl_rrt_paper import PaperCLRRT
from dma_rrt.collision import CollisionChecker, SIXTEEN_PILLAR_MAP
from dma_rrt.dynamics import PurePursuit, SkidSteerModel
from dma_rrt.tree import Node

REFERENCE_COLOR = "tab:purple"
TRAJECTORY_COLOR = "tab:green"
COLLISION_COLOR = "tab:red"
PATH_COLOR = "tab:blue"
SAMPLE_COLOR = "tab:orange"

EXPLANATION = [
    ("Sample", r"pick $x_{sample}$ in the free space", SAMPLE_COLOR),
    ("Nearest", "try the closest nodes in order", "black"),
    ("Reference", "line from the parent's waypoint, at most step size long", REFERENCE_COLOR),
    ("Simulate", "pure pursuit steers toward the look-ahead point", TRAJECTORY_COLOR),
    ("Check", "every simulated state must be safe", COLLISION_COLOR),
    ("Add", "the new node keeps the waypoint and the trajectory", "black"),
]

PAPER_EXPLANATION = [
    ("Sample", r"pick $x_{sample}$ in the free space", SAMPLE_COLOR),
    ("Nearest", "use only the closest node", "black"),
    ("Reference", "line from the parent's waypoint to the sample", REFERENCE_COLOR),
    ("Simulate", "pure pursuit steers toward the look-ahead point", TRAJECTORY_COLOR),
    ("Check", "stop at the first unsafe state, keep the part before it", COLLISION_COLOR),
    ("Add", "the new node keeps the waypoint and the trajectory", "black"),
]


class CLRRTVideo:
    def __init__(
        self,
        checker: CollisionChecker,
        controller: PurePursuit,
        start_state: np.ndarray,
        goal_pos: np.ndarray,
        goal_radius: float,
        explanation: List,
        video_path: str,
        fps: int,
    ) -> None:
        self._checker = checker
        self._controller = controller
        self._fps = fps
        self._writer = imageio.get_writer(
            video_path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=2
        )
        self._fig = plt.figure(figsize=(12, 6.75), dpi=120)
        self._ax = self._fig.add_axes([0.05, 0.08, 0.5, 0.82])
        panel = self._fig.add_axes([0.58, 0.0, 0.42, 1.0])
        panel.set_xlim(0.0, 1.0)
        panel.set_ylim(0.0, 1.0)
        panel.axis("off")

        ax = self._ax
        checker.draw(ax)
        self._tree_lines = LineCollection([], colors="gray", linewidths=0.9)
        ax.add_collection(self._tree_lines)
        self._path_lines = LineCollection([], colors=PATH_COLOR, linewidths=3.0)
        ax.add_collection(self._path_lines)
        ax.plot(*start_state[:2], "o", color="tab:red", markersize=9, zorder=5)
        ax.text(start_state[0] - 0.7, start_state[1] + 0.5, "start", color="tab:red", zorder=5)
        ax.add_patch(plt.Circle(goal_pos, goal_radius, color="tab:green", alpha=0.35))
        ax.text(goal_pos[0] - 0.7, goal_pos[1] - 1.7, "goal", color="tab:green", zorder=5)
        ax.set_xlim(checker.area_limits[0], checker.area_limits[1])
        ax.set_ylim(checker.area_limits[2], checker.area_limits[3])
        ax.autoscale(False)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        self._title = ax.set_title("", fontsize=12, fontweight="bold")

        (self._sample_mark,) = ax.plot([], [], "x", color=SAMPLE_COLOR, markersize=13, markeredgewidth=3, zorder=6)
        (self._reference_line,) = ax.plot([], [], "--", color=REFERENCE_COLOR, linewidth=2, zorder=4)
        (self._trail,) = ax.plot([], [], "-", color=TRAJECTORY_COLOR, linewidth=2.5, zorder=4)
        (self._lookahead,) = ax.plot([], [], "o", color=REFERENCE_COLOR, markersize=7, zorder=6)
        (self._heading,) = ax.plot([], [], "-", color="black", linewidth=2, zorder=6)
        self._robot = plt.Circle((0.0, 0.0), checker.safe_dist, fill=False, color="black", linewidth=1.5, visible=False, zorder=5)
        ax.add_patch(self._robot)

        panel.text(0.02, 0.945, "CL-RRT, tree expansion", fontsize=14, fontweight="bold")
        self._status = panel.text(0.02, 0.895, "", fontsize=11)
        self._message = panel.text(0.02, 0.85, "", fontsize=11, fontweight="bold")
        for i, (name, text, color) in enumerate(explanation):
            y = 0.76 - i * 0.095
            panel.text(0.02, y, f"{i + 1}. {name}", fontsize=12, fontweight="bold", color=color)
            panel.text(0.05, y - 0.04, text, fontsize=10.5)
        panel.text(
            0.02, 0.135,
            "Circle: robot with the safety margin. Line in the circle: heading.\n"
            "Light red: the robot's center must not enter.",
            fontsize=9.5, color="dimgray", va="top",
        )

    def show_setup(self, seconds: float) -> None:
        self._title.set_text("Start: the robot has a position and a heading")
        self._write(seconds)

    def show_step_slowly(self, sample_num: int, step: CLRRTStep, num_nodes: int) -> None:
        """
        Show the sample and simulate the robot for all attempts of a sample
        """
        self._sample_mark.set_data([step.sample_pos[0]], [step.sample_pos[1]])
        self._status.set_text(f"Sample {sample_num},  nodes in the tree: {num_nodes}")
        self._message.set_text("")
        self._title.set_text("Sample a point")
        self._write(1.2)
        for i, attempt in enumerate(step.attempts):
            self._show_attempt(i, attempt)
        self._clear_robot()
        self._sample_mark.set_data([], [])

    def show_step_fast(self, sample_num: int, step: CLRRTStep, num_nodes: int, seconds: float) -> None:
        self._title.set_text("Fast forward: repeat for more samples")
        self._message.set_text("")
        self._sample_mark.set_data([step.sample_pos[0]], [step.sample_pos[1]])
        for attempt in step.attempts:
            if attempt.new_node is not None:
                self._add_trajectory(attempt)
        self._status.set_text(f"Sample {sample_num},  nodes in the tree: {num_nodes}")
        self._write(seconds)
        self._sample_mark.set_data([], [])

    def show_path(self, trajectory: np.ndarray, cost: float, seconds: float, hold_seconds: float) -> None:
        """
        Drive along the found path
        """
        self._status.set_text(f"Travel time: {cost:.1f} s")
        self._message.set_text("Path found, the robot can follow it as simulated")
        self._title.set_text("Done: the path is a simulated trajectory")
        stride = max(int(len(trajectory) / (seconds * self._fps)), 1)
        for i in range(stride, len(trajectory), stride):
            self._path_lines.set_segments([trajectory[: i + 1, :2]])
            self._set_robot(trajectory[i])
            self._write(1.0 / self._fps)
        self._path_lines.set_segments([trajectory[:, :2]])
        self._set_robot(trajectory[-1])
        self._write(hold_seconds)

    def close(self) -> None:
        self._writer.close()
        plt.close(self._fig)

    def _show_attempt(self, attempt_index: int, attempt: CLRRTAttempt) -> None:
        near = attempt.nearest_node
        order = ["closest", "2nd closest", "3rd closest", "4th closest", "5th closest"][attempt_index]
        self._title.set_text(f"Try the {order} node")
        self._reference_line.set_data(
            [near.waypoint[0], attempt.waypoint[0]], [near.waypoint[1], attempt.waypoint[1]]
        )
        self._trail.set_color(TRAJECTORY_COLOR)
        self._trail.set_data([], [])
        self._lookahead.set_data([], [])
        self._set_robot(near.state)
        self._message.set_text("")
        self._write(1.0)

        states = np.vstack([near.state[np.newaxis], attempt.trajectory])
        end = len(states)
        if attempt.collision_pos is not None:
            hits = np.flatnonzero(np.all(states[:, :2] == attempt.collision_pos, axis=1))
            end = hits[0] + 1
        self._title.set_text("Simulate the robot along the reference")
        for i in range(1, end):
            target, _ = self._controller.lookahead_point(states[i - 1, :2], near.waypoint, attempt.waypoint)
            self._lookahead.set_data([target[0]], [target[1]])
            self._trail.set_data(states[: i + 1, 0], states[: i + 1, 1])
            self._set_robot(states[i])
            self._write(1.0 / self._fps)

        if attempt.new_node is not None:
            self._title.set_text("Whole trajectory is safe: add a node")
            self._message.set_text(f"Added: {len(attempt.trajectory)} states, {attempt.new_node.cost:.1f} s from the start")
            self._message.set_color(TRAJECTORY_COLOR)
            self._write(1.2)
            self._add_trajectory(attempt)
        else:
            self._trail.set_color(COLLISION_COLOR)
            if attempt.collision_pos is not None:
                self._title.set_text("Unsafe state: reject this trajectory")
                self._message.set_text("Rejected: too close to an obstacle or out of the map")
            else:
                self._title.set_text("Could not reach the waypoint in time: reject")
                self._message.set_text("Rejected: the waypoint is too hard to reach from this heading")
            self._message.set_color(COLLISION_COLOR)
            self._write(1.5)
        self._reference_line.set_data([], [])
        self._trail.set_data([], [])
        self._lookahead.set_data([], [])

    def _add_trajectory(self, attempt: CLRRTAttempt) -> None:
        segments = list(self._tree_lines.get_segments())
        segments.append(np.vstack([attempt.nearest_node.state[np.newaxis], attempt.trajectory])[:, :2])
        self._tree_lines.set_segments(segments)

    def _set_robot(self, state: np.ndarray) -> None:
        self._robot.set_visible(True)
        self._robot.center = (state[0], state[1])
        length = self._checker.safe_dist
        self._heading.set_data(
            [state[0], state[0] + length * np.cos(state[2])],
            [state[1], state[1] + length * np.sin(state[2])],
        )

    def _clear_robot(self) -> None:
        self._robot.set_visible(False)
        self._heading.set_data([], [])

    def _write(self, seconds: float) -> None:
        self._fig.canvas.draw()
        frame = np.asarray(self._fig.canvas.buffer_rgba())[:, :, :3]
        for _ in range(max(int(round(seconds * self._fps)), 1)):
            self._writer.append_data(frame)


def draw_result(
    checker: CollisionChecker,
    planner: CLRRT,
    start_state: np.ndarray,
    goal_pos: np.ndarray,
    goal_radius: float,
    goal_node: Optional[Node],
):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    checker.draw(ax)
    segments = [
        np.vstack([node.parent.state[np.newaxis], node.trajectory])[:, :2]
        for node in planner.tree.nodes[1:]
    ]
    ax.add_collection(LineCollection(segments, colors="gray", linewidths=0.6))
    if goal_node is not None:
        trajectory = planner.path_trajectory(goal_node)
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=PATH_COLOR, linewidth=2.0, label="path")
    ax.plot(*start_state[:2], "o", color="tab:red", label="start")
    ax.add_patch(plt.Circle(goal_pos, goal_radius, color="tab:green", alpha=0.5, label="goal region"))
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3, fontsize=8)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default=SIXTEEN_PILLAR_MAP, help="map yaml file, relative to the repo root")
    parser.add_argument("--start", type=float, nargs=3, default=[1.0, 1.0, 0.0], help="x y yaw")
    parser.add_argument("--goal", type=float, nargs=2, default=[24.0, 24.0])
    parser.add_argument("--goal_radius", type=float, default=1.0)
    parser.add_argument("--paper", action="store_true", help="use Algorithm 1 of the paper as written")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--save", type=str, default=None, help="save the figure here instead of showing it")
    parser.add_argument("--video", type=str, default=None, help="save an mp4 here instead of showing the figure")
    parser.add_argument("--slow_samples", type=int, default=6, help="number of samples shown with the simulated robot")
    parser.add_argument("--fast_total_seconds", type=float, default=15.0, help="total time of the fast forward [s]")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    start_state = np.array(args.start)
    goal_pos = np.array(args.goal)

    checker = CollisionChecker(args.map)
    if not checker.is_free(start_state[:2]):
        parser.error(f"start {args.start} is too close to an obstacle")
    if not checker.is_free(goal_pos):
        parser.error(f"goal {args.goal} is too close to an obstacle")

    controller = PurePursuit(SkidSteerModel())
    if args.paper:
        planner = PaperCLRRT(checker, controller, goal_radius=args.goal_radius, seed=args.seed)
    else:
        planner = CLRRT(checker, controller, goal_radius=args.goal_radius, seed=args.seed)
    planner.reset(start_state)

    steps: List[CLRRTStep] = []
    goal_node: Optional[Node] = None
    for _ in range(args.max_iter):
        step = planner.step(goal_pos)
        steps.append(step)
        if step.new_node is not None and planner.is_in_goal(step.new_node, goal_pos):
            goal_node = step.new_node
            break

    if goal_node is None:
        print(f"samples: {len(steps)}, nodes: {len(planner.tree)}, goal not reached")
    else:
        print(f"samples: {len(steps)}, nodes: {len(planner.tree)}, travel time: {goal_node.cost:.1f} s")

    if args.save is not None or args.video is None:
        fig = draw_result(checker, planner, start_state, goal_pos, args.goal_radius, goal_node)
        if args.save is not None:
            fig.savefig(args.save, dpi=150)
        elif args.video is None:
            plt.show()
        plt.close(fig)

    if args.video is not None:
        explanation = PAPER_EXPLANATION if args.paper else EXPLANATION
        video = CLRRTVideo(checker, controller, start_state, goal_pos, args.goal_radius, explanation, args.video, args.fps)
        video.show_setup(2.0)
        num_nodes = 1
        fast_seconds = args.fast_total_seconds / max(len(steps) - args.slow_samples, 1)
        for i, step in enumerate(steps):
            if i < args.slow_samples:
                video.show_step_slowly(i + 1, step, num_nodes)
            else:
                video.show_step_fast(i + 1, step, num_nodes + (step.new_node is not None), fast_seconds)
            num_nodes += step.new_node is not None
        if goal_node is not None:
            video.show_path(planner.path_trajectory(goal_node), goal_node.cost, seconds=8.0, hold_seconds=3.0)
        video.close()


if __name__ == "__main__":
    main()
