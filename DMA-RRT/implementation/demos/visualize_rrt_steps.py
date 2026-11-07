"""
    Explain RRT step by step as a video: sample, nearest, steer, collision check, add, goal check, and path

    The first iterations are slow, then the video speeds up. The first rejected iteration
    and the last iteration, which reaches the goal, are slow again.

    Run from DMA-RRT/implementation:
    python3 -m demos.visualize_rrt_steps --video outputs/1_basic_rrt/rrt_steps.mp4
"""

from __future__ import annotations

import argparse
from typing import List
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from dma_rrt.collision import CollisionChecker, SIXTEEN_PILLAR_MAP
from dma_rrt.rrt import RRT, RRTStep

SAMPLE_COLOR = "tab:orange"
NEAREST_COLOR = "tab:purple"
NEW_COLOR = "tab:green"
COLLISION_COLOR = "tab:red"
ADD_COLOR = "black"
GOAL_COLOR = "tab:cyan"
PATH_COLOR = "tab:blue"

# phase: (title on the map, row title on the panel, formula on the panel, color)
PHASES = {
    "sample": ("1. Sample: pick a random point", "1. Sample", r"$x_{rand} \sim$ uniform(area)", SAMPLE_COLOR),
    "nearest": ("2. Nearest: find the closest node in the tree", "2. Nearest", r"$x_{near} = \arg\min_{x \in T} \|x - x_{rand}\|$", NEAREST_COLOR),
    "steer": ("3. Steer: move from the nearest node toward the sample", "3. Steer", r"$x_{new} = x_{near} + \min(d, \Delta)\,\frac{x_{rand} - x_{near}}{d}$", NEW_COLOR),
    "collision": ("4. Collision check: is the whole edge safe?", "4. Collision", r"dist(obstacle) $\geq r_{safe}$ along the edge ?", COLLISION_COLOR),
    "add": ("5. Add: connect the new node to its parent", "5. Add", r"parent$(x_{new}) = x_{near}$", ADD_COLOR),
    "goal": ("6. Goal check: is the new node in the goal region?", "6. Goal check", r"$\|x_{new} - x_{goal}\| \leq r_{goal}$ ?", GOAL_COLOR),
    "path": ("7. Path: follow the parents from the goal back to the start", "7. Path", "follow parent() back to the root", PATH_COLOR),
}


class StepVisualizer:
    def __init__(
        self,
        collision_checker: CollisionChecker,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        goal_radius: float,
        step_size: float,
        video_path: str,
        fps: int,
    ) -> None:
        self._collision_checker = collision_checker
        self._goal_pos = goal_pos
        self._goal_radius = goal_radius
        self._step_size = step_size
        self._fps = fps
        self._node_states: List[np.ndarray] = [start_pos]
        # macro_block_size=2 keeps the figure size, yuv420p only needs even sizes
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
        collision_checker.draw(ax)
        self._edge_lines = LineCollection([], colors="gray", linewidths=1.0)
        ax.add_collection(self._edge_lines)
        (self._node_points,) = ax.plot([], [], ".", color="dimgray", markersize=5)
        self._path_lines = LineCollection([], colors=PATH_COLOR, linewidths=3.0)
        ax.add_collection(self._path_lines)
        ax.plot(*start_pos, "o", color="tab:red", markersize=9, zorder=5)
        ax.text(start_pos[0] - 0.7, start_pos[1] + 0.5, "start", color="tab:red", zorder=5)
        ax.add_patch(plt.Circle(goal_pos, goal_radius, color="tab:green", alpha=0.35))
        ax.text(goal_pos[0] - 0.7, goal_pos[1] - 1.7, "goal", color="tab:green", zorder=5)
        # keep the map range, circles drawn later must not change it
        ax.set_xlim(collision_checker.area_limits[0], collision_checker.area_limits[1])
        ax.set_ylim(collision_checker.area_limits[2], collision_checker.area_limits[3])
        ax.autoscale(False)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        self._title = ax.set_title("", fontsize=12, fontweight="bold")

        panel.text(0.02, 0.955, "RRT, one iteration", fontsize=14, fontweight="bold")
        self._status = panel.text(0.02, 0.915, "", fontsize=11)
        panel.text(
            0.02, 0.875,
            rf"$\Delta$ = {step_size:.1f} m, $r_{{goal}}$ = {goal_radius:.1f} m, "
            rf"$r_{{safe}}$ = {collision_checker.safe_dist:.1f} m (light red: closer than $r_{{safe}}$)",
            fontsize=9.5, color="dimgray",
        )

        self._highlights = {}
        self._values = {}
        row_height = 0.118
        for i, (phase, (_, row_title, formula, color)) in enumerate(PHASES.items()):
            y = 0.85 - i * row_height
            highlight = plt.Rectangle(
                (0.01, y - row_height + 0.003), 0.96, row_height - 0.006,
                color=color, alpha=0.15, visible=False,
            )
            panel.add_patch(highlight)
            panel.text(0.02, y - 0.035, row_title, fontsize=12, fontweight="bold", color=color, va="center")
            panel.text(0.30, y - 0.035, formula, fontsize=11, va="center")
            self._values[phase] = panel.text(0.02, y - 0.085, "", fontsize=10, va="center")
            self._highlights[phase] = highlight

        self._overlays: List = []

    def show_setup(self, seconds: float) -> None:
        self._title.set_text("Start: the tree has only the start node")
        self._write(seconds)

    def show_iteration(
        self, iter_num: int, step: RRTStep, is_goal_sample: bool, seconds: float
    ) -> None:
        """
        Show all phases of an iteration
        :param seconds: time to show each phase
        """
        self._set_status(iter_num)
        self._reset_values()
        self._show_sample(step, is_goal_sample, seconds)
        self._show_nearest(step, seconds)
        self._show_steer(step, seconds)
        self._show_collision_check(step, seconds)
        if step.new_node is not None:
            self._show_add(step, iter_num, seconds)
            self._show_goal_check(step, seconds)

    def show_iteration_fast(self, iter_num: int, step: RRTStep, seconds: float) -> None:
        """
        Show only the result of an iteration
        """
        self._clear_overlays()
        self._set_phase(None)
        self._reset_values()
        self._title.set_text("Fast forward: repeat steps 1-6")
        near_x, near_y = step.nearest_node.state
        steer_x, steer_y = step.steer_pos
        sample_x, sample_y = step.sample_pos
        self._overlays += self._ax.plot(sample_x, sample_y, "x", color=SAMPLE_COLOR, markersize=8, markeredgewidth=2)
        self._overlays += self._ax.plot([near_x, sample_x], [near_y, sample_y], ":", color=SAMPLE_COLOR)
        if step.new_node is None:
            self._overlays += self._ax.plot([near_x, steer_x], [near_y, steer_y], "-", color=COLLISION_COLOR, linewidth=2)
        else:
            self._add_node(step)
            self._overlays += self._ax.plot(steer_x, steer_y, "o", color=NEW_COLOR, markersize=7)
        self._set_status(iter_num)
        self._write(seconds)

    def show_path(self, path: np.ndarray, total_seconds: float, hold_seconds: float) -> None:
        """
        Trace the path from the goal back to the start
        """
        self._clear_overlays()
        self._set_phase("path")
        self._values["path"].set_text(f"{len(path) - 1} edges, length: {np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)):.2f} m")
        segments = []
        seconds = total_seconds / (len(path) - 1)
        for i in range(len(path) - 1, 0, -1):
            segments.append(np.array([path[i], path[i - 1]]))
            self._path_lines.set_segments(segments)
            self._write(seconds)

        self._title.set_text("Done: path from the start to the goal")
        self._write(hold_seconds)

    def close(self) -> None:
        self._writer.close()
        plt.close(self._fig)

    def _show_sample(self, step: RRTStep, is_goal_sample: bool, seconds: float) -> None:
        self._clear_overlays()
        self._set_phase("sample")
        sample_x, sample_y = step.sample_pos
        self._overlays += self._ax.plot(sample_x, sample_y, "x", color=SAMPLE_COLOR, markersize=13, markeredgewidth=3)
        source = "the goal itself (5% chance)" if is_goal_sample else "uniform random"
        self._values["sample"].set_text(f"$x_{{rand}}$ = ({sample_x:.2f}, {sample_y:.2f}), {source}")
        self._write(seconds)

    def _show_nearest(self, step: RRTStep, seconds: float) -> None:
        self._set_phase("nearest")
        lines = LineCollection(
            [np.array([state, step.sample_pos]) for state in self._node_states],
            colors=SAMPLE_COLOR, linewidths=0.7, linestyles="dashed", alpha=0.35,
        )
        self._ax.add_collection(lines)
        self._overlays.append(lines)
        near_x, near_y = step.nearest_node.state
        sample_x, sample_y = step.sample_pos
        self._overlays += self._ax.plot([near_x, sample_x], [near_y, sample_y], "--", color=NEAREST_COLOR, linewidth=2)
        self._overlays += self._ax.plot(near_x, near_y, "o", color=NEAREST_COLOR, markersize=11)
        dist = np.linalg.norm(step.sample_pos - step.nearest_node.state)
        self._values["nearest"].set_text(f"$x_{{near}}$ = ({near_x:.2f}, {near_y:.2f}), $d$ = {dist:.2f} m, nodes compared: {len(self._node_states)}")
        self._write(seconds)

    def _show_steer(self, step: RRTStep, seconds: float) -> None:
        self._set_phase("steer")
        near_x, near_y = step.nearest_node.state
        steer_x, steer_y = step.steer_pos
        circle = plt.Circle(step.nearest_node.state, self._step_size, fill=False, color=NEW_COLOR, linestyle="--")
        self._ax.add_patch(circle)
        self._overlays.append(circle)
        self._overlays.append(
            self._ax.annotate(
                "", xy=(steer_x, steer_y), xytext=(near_x, near_y),
                arrowprops=dict(arrowstyle="-|>", color=NEW_COLOR, linewidth=2.5),
            )
        )
        self._overlays += self._ax.plot(steer_x, steer_y, "o", color=NEW_COLOR, markersize=9)
        dist = np.linalg.norm(step.sample_pos - step.nearest_node.state)
        if dist > self._step_size:
            reason = rf"$d$ > $\Delta$: stop after {self._step_size:.2f} m"
        else:
            reason = r"$d \leq \Delta$: go to the sample"
        self._values["steer"].set_text(f"$x_{{new}}$ = ({steer_x:.2f}, {steer_y:.2f}), {reason}")
        self._write(seconds)

    def _show_collision_check(self, step: RRTStep, seconds: float) -> None:
        self._set_phase("collision")
        checker = self._collision_checker
        points = checker.edge_points(step.nearest_node.state, step.steer_pos)
        if step.collision_pos is None:
            self._overlays += self._ax.plot(points[:, 0], points[:, 1], ".", color=NEW_COLOR, markersize=3, zorder=4)
            self._values["collision"].set_text(f"{len(points)} points on the edge are all safe")
        else:
            num_safe = np.flatnonzero(np.all(points == step.collision_pos, axis=1))[0]
            safe_points = points[:num_safe]
            self._overlays += self._ax.plot(safe_points[:, 0], safe_points[:, 1], ".", color=NEW_COLOR, markersize=3, zorder=4)
            self._overlays += self._ax.plot(points[num_safe:, 0], points[num_safe:, 1], ".", color=COLLISION_COLOR, markersize=3, zorder=4)
            robot = plt.Circle(step.collision_pos, checker.safe_dist, fill=False, color=COLLISION_COLOR, linewidth=2, zorder=4)
            self._ax.add_patch(robot)
            self._overlays.append(robot)
            self._overlays += self._ax.plot(*step.collision_pos, "X", color=COLLISION_COLOR, markersize=11, zorder=5)
            dist = checker.obstacle_dist(step.collision_pos)
            self._values["collision"].set_text(f"unsafe: dist {dist:.2f} m < {checker.safe_dist:.2f} m. Reject, no new node")
        self._write(seconds)

    def _show_add(self, step: RRTStep, iter_num: int, seconds: float) -> None:
        self._set_phase("add")
        self._add_node(step)
        self._set_status(iter_num)
        self._clear_overlays()
        new_x, new_y = step.new_node.state
        near_x, near_y = step.nearest_node.state
        self._overlays += self._ax.plot([near_x, new_x], [near_y, new_y], "-", color=ADD_COLOR, linewidth=2.5)
        self._overlays += self._ax.plot(new_x, new_y, "o", color=NEW_COLOR, markersize=8)
        self._values["add"].set_text(f"cost: {step.nearest_node.cost:.2f} + {step.new_node.cost - step.nearest_node.cost:.2f} = {step.new_node.cost:.2f} m from the start")
        self._write(seconds)

    def _show_goal_check(self, step: RRTStep, seconds: float) -> None:
        self._set_phase("goal")
        new_x, new_y = step.new_node.state
        self._overlays += self._ax.plot(
            [new_x, self._goal_pos[0]], [new_y, self._goal_pos[1]], ":", color=GOAL_COLOR, linewidth=2
        )
        dist = np.linalg.norm(step.new_node.state - self._goal_pos)
        result = "reached the goal region" if dist <= self._goal_radius else "not yet, next iteration"
        self._values["goal"].set_text(f"distance = {dist:.2f} m, {result}")
        self._write(seconds)

    def _add_node(self, step: RRTStep) -> None:
        segments = list(self._edge_lines.get_segments())
        segments.append(np.array([step.nearest_node.state, step.new_node.state]))
        self._edge_lines.set_segments(segments)
        self._node_states.append(step.new_node.state)
        states = np.array(self._node_states[1:])
        self._node_points.set_data(states[:, 0], states[:, 1])

    def _set_status(self, iter_num: int) -> None:
        self._status.set_text(f"Iteration {iter_num},  nodes in the tree: {len(self._node_states)}")

    def _set_phase(self, phase) -> None:
        for name, highlight in self._highlights.items():
            highlight.set_visible(name == phase)
        if phase is not None:
            self._title.set_text(PHASES[phase][0])

    def _reset_values(self) -> None:
        for value in self._values.values():
            value.set_text("")

    def _clear_overlays(self) -> None:
        for overlay in self._overlays:
            overlay.remove()
        self._overlays = []

    def _write(self, seconds: float) -> None:
        self._fig.canvas.draw()
        frame = np.asarray(self._fig.canvas.buffer_rgba())[:, :, :3]
        for _ in range(max(int(round(seconds * self._fps)), 1)):
            self._writer.append_data(frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="outputs/1_basic_rrt/rrt_steps.mp4")
    parser.add_argument("--map", type=str, default=SIXTEEN_PILLAR_MAP, help="map yaml file, relative to the repo root")
    parser.add_argument("--start", type=float, nargs=2, default=[1.0, 1.0])
    parser.add_argument("--goal", type=float, nargs=2, default=[24.0, 24.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_iter", type=int, default=5000)
    # larger than the planner's default so that the steps are visible on the whole map
    parser.add_argument("--step_size", type=float, default=2.0)
    parser.add_argument("--goal_radius", type=float, default=1.0)
    parser.add_argument("--slow_iters", type=int, default=3, help="number of iterations at the slowest speed")
    parser.add_argument("--medium_iters", type=int, default=6, help="number of iterations at the medium speed")
    parser.add_argument("--slow_seconds", type=float, default=2.5, help="time per phase in the slow iterations [s]")
    parser.add_argument("--medium_seconds", type=float, default=0.5, help="time per phase in the medium iterations [s]")
    parser.add_argument("--fast_total_seconds", type=float, default=15.0, help="total time of the fast forward [s]")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    start_pos = np.array(args.start)
    goal_pos = np.array(args.goal)

    collision_checker = CollisionChecker(args.map)
    if not collision_checker.is_free(start_pos):
        parser.error(f"start {args.start} is too close to an obstacle")
    if not collision_checker.is_free(goal_pos):
        parser.error(f"goal {args.goal} is too close to an obstacle")

    planner = RRT(collision_checker, step_size=args.step_size, goal_radius=args.goal_radius, seed=args.seed)
    planner.reset(start_pos)

    # run the planner first because the last iteration has to be known to slow it down
    steps: List[RRTStep] = []
    goal_node = None
    for _ in range(args.max_iter):
        step = planner.step(goal_pos)
        if step.steer_pos is None:
            continue
        steps.append(step)
        if step.new_node is not None and planner.is_in_goal(step.new_node, goal_pos):
            goal_node = step.new_node
            break

    num_rejected = sum(step.new_node is None for step in steps)
    if goal_node is None:
        print(f"iterations: {len(steps)}, rejected: {num_rejected}, nodes: {len(planner.tree)}, goal not reached")
    else:
        print(f"iterations: {len(steps)}, rejected: {num_rejected}, nodes: {len(planner.tree)}, cost: {goal_node.cost:.2f} m")

    first_rejected = next((i for i, step in enumerate(steps) if step.new_node is None), None)
    num_fast = max(len(steps) - args.slow_iters - args.medium_iters, 1)
    fast_seconds = args.fast_total_seconds / num_fast

    visualizer = StepVisualizer(
        collision_checker, start_pos, goal_pos, args.goal_radius, args.step_size, args.video, args.fps
    )
    visualizer.show_setup(args.slow_seconds)
    for i, step in enumerate(steps):
        is_last = i == len(steps) - 1
        is_goal_sample = np.array_equal(step.sample_pos, goal_pos)
        if i < args.slow_iters or i == first_rejected or is_last:
            visualizer.show_iteration(i + 1, step, is_goal_sample, args.slow_seconds)
        elif i < args.slow_iters + args.medium_iters:
            visualizer.show_iteration(i + 1, step, is_goal_sample, args.medium_seconds)
        else:
            visualizer.show_iteration_fast(i + 1, step, fast_seconds)

    if goal_node is not None:
        visualizer.show_path(planner.tree.path_to(goal_node), total_seconds=4.0, hold_seconds=3.0)
    visualizer.close()


if __name__ == "__main__":
    main()
