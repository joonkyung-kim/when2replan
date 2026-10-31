"""
    Move a robot with the execution loop of CL-RRT: the tree keeps growing while the robot follows its plan

    By default an obstacle appears on the plan, to show the lazy check and the replanning around it.
    This obstacle is not in the paper, use --no_obstacle for the paper's static environment.

    Run from DMA-RRT/implementation:
    python3 demo_execution.py --video execution.mp4
    python3 demo_execution.py --no_obstacle --save execution.png
    python3 demo_execution.py --paper --video execution_paper.mp4  # Algorithm 1 of the paper as written
    python3 demo_execution.py --moving --no_obstacle --video moving.mp4  # robots with fixed plans in the corridors
"""

from __future__ import annotations

import argparse
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from cl_rrt import CLRRT
from cl_rrt_paper import PaperCLRRT
from collision import CollisionChecker, SIXTEEN_PILLAR_MAP
from dynamics import PurePursuit, SkidSteerModel
from execution import CLRRTExecutor, CycleResult
from scenarios import corridor_obstacles

TRAIL_COLOR = "tab:red"
COMMITTED_COLOR = "black"
PLAN_COLOR = "tab:blue"
OBSTACLE_COLOR = "tab:orange"


class ExecutionFigure:
    def __init__(
        self,
        checker: CollisionChecker,
        start_state: np.ndarray,
        goal_pos: np.ndarray,
        goal_radius: float,
    ) -> None:
        self._checker = checker
        self.fig = plt.figure(figsize=(12, 6.75), dpi=120)
        self._ax = self.fig.add_axes([0.05, 0.08, 0.5, 0.82])
        panel = self.fig.add_axes([0.58, 0.0, 0.42, 1.0])
        panel.set_xlim(0.0, 1.0)
        panel.set_ylim(0.0, 1.0)
        panel.axis("off")

        ax = self._ax
        checker.draw(ax)
        self._tree_lines = LineCollection([], colors="silver", linewidths=0.8)
        ax.add_collection(self._tree_lines)
        (self._plan_line,) = ax.plot([], [], "-", color=PLAN_COLOR, linewidth=2.5, zorder=4)
        (self._committed_line,) = ax.plot([], [], "-", color=COMMITTED_COLOR, linewidth=3.5, zorder=5)
        (self._trail,) = ax.plot([], [], "-", color=TRAIL_COLOR, linewidth=1.5, zorder=4)
        (self._heading,) = ax.plot([], [], "-", color="black", linewidth=2, zorder=7)
        self._robot = plt.Circle(start_state[:2], checker.safe_dist, fill=False, color="black", linewidth=1.5, zorder=6)
        ax.add_patch(self._robot)
        ax.plot(*start_state[:2], "o", color="tab:red", markersize=8, zorder=5)
        ax.add_patch(plt.Circle(goal_pos, goal_radius, color="tab:green", alpha=0.35))
        ax.text(goal_pos[0] - 0.7, goal_pos[1] - 1.7, "goal", color="tab:green", zorder=5)
        ax.set_xlim(checker.area_limits[0], checker.area_limits[1])
        ax.set_ylim(checker.area_limits[2], checker.area_limits[3])
        ax.autoscale(False)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        self._title = ax.set_title("", fontsize=12, fontweight="bold")

        panel.text(0.02, 0.945, "CL-RRT, execution loop", fontsize=14, fontweight="bold")
        self._status = panel.text(0.02, 0.895, "", fontsize=11)
        self._stats = panel.text(0.02, 0.855, "", fontsize=10.5)
        self._message = panel.text(0.02, 0.805, "", fontsize=11, fontweight="bold")

        rows = [
            ("Black: committed path", "the robot cannot change it anymore. Its end is the tree's root", COMMITTED_COLOR),
            ("Blue: current plan", "the best path from the root, it can change every cycle", PLAN_COLOR),
            ("Gray: tree", "only the root's descendants, it keeps growing", "gray"),
            ("Red: driven so far", "", TRAIL_COLOR),
        ]
        for i, (name, text, color) in enumerate(rows):
            y = 0.72 - i * 0.075
            panel.text(0.02, y, name, fontsize=11, fontweight="bold", color=color)
            panel.text(0.05, y - 0.035, text, fontsize=10)
        if len(checker.moving_obstacles) > 0:
            panel.text(
                0.02, 0.45, "Orange: moving robot. Its path is red now, green in 10 s",
                fontsize=10.5, fontweight="bold", color=OBSTACLE_COLOR,
            )

        self._moving_circles = []
        self._moving_lines = []
        for obstacle in checker.moving_obstacles:
            circle = plt.Circle(obstacle.positions[0], obstacle.radius, color=OBSTACLE_COLOR, zorder=6)
            ax.add_patch(circle)
            lines = LineCollection([], linewidths=3.0, zorder=3)
            ax.add_collection(lines)
            self._moving_circles.append(circle)
            self._moving_lines.append(lines)
        self.set_moving_obstacles(0.0)
        panel.text(0.02, 0.385, "Each cycle (1 s)", fontsize=12, fontweight="bold")
        steps = [
            "1. Extend the committed path to cover this cycle",
            "2. Move the root there, drop the other branches",
            "3. Grow the tree with new samples",
            "4. Pick the best path, a path to the goal comes first",
            "5. Lazy check: simulate it again against the current",
            "    obstacles. If unsafe, remove it and pick again",
            "6. Move the robot along the committed path",
        ]
        for i, text in enumerate(steps):
            panel.text(0.02, 0.34 - i * 0.042, text, fontsize=10)

    def set_tree(self, executor: CLRRTExecutor, planner: CLRRT) -> None:
        self._tree_lines.set_segments(
            [
                np.vstack([node.parent.state[np.newaxis], node.trajectory])[:, :2]
                for node in planner.tree.nodes[1:]
            ]
        )

    def set_robot(self, state: np.ndarray, trail: np.ndarray, committed: np.ndarray, plan: np.ndarray) -> None:
        self._robot.center = (state[0], state[1])
        length = self._checker.safe_dist
        self._heading.set_data(
            [state[0], state[0] + length * np.cos(state[2])],
            [state[1], state[1] + length * np.sin(state[2])],
        )
        self._trail.set_data(trail[:, 0], trail[:, 1])
        self._committed_line.set_data(committed[:, 0], committed[:, 1])
        self._plan_line.set_data(plan[:, 0], plan[:, 1])

    def set_moving_obstacles(self, time: float) -> None:
        """
        Draw the moving obstacles at time, with their future paths colored by the arrival time as in Fig. 5 of the paper
        """
        for obstacle, circle, lines in zip(self._checker.moving_obstacles, self._moving_circles, self._moving_lines):
            positions = obstacle.positions_from(time)
            circle.center = (positions[0, 0], positions[0, 1])
            if len(positions) < 2:
                lines.set_segments([])
                continue
            lines.set_segments(np.stack([positions[:-1], positions[1:]], axis=1))
            green_ratio = np.minimum(np.arange(len(positions) - 1) * obstacle.dt / 10.0, 1.0)
            lines.set_color(np.column_stack([1.0 - green_ratio, green_ratio, np.zeros_like(green_ratio)]))

    def add_circle(self, center: np.ndarray, radius: float) -> None:
        self._ax.add_patch(plt.Circle(center, radius + self._checker.safe_dist, color="mistyrose", zorder=1))
        self._ax.add_patch(plt.Circle(center, radius, color=OBSTACLE_COLOR, zorder=2))

    def set_text(self, title: str, status: str, stats: str, message: str = "", color: str = "black") -> None:
        self._title.set_text(title)
        self._status.set_text(status)
        self._stats.set_text(stats)
        self._message.set_text(message)
        self._message.set_color(color)


def get_message(result: CycleResult, has_new_obstacle: bool) -> tuple:
    if result.is_safety_stop:
        return "Safety stop: the committed path became unsafe", "tab:red"
    if result.is_turning:
        return "Nowhere to go from this heading: turn in place", "tab:purple"
    if result.num_lazy_removed > 0:
        return f"Lazy check removed {result.num_lazy_removed} unsafe path(s), plan changed", "tab:red"
    if has_new_obstacle:
        return "A new obstacle appeared on the plan", OBSTACLE_COLOR
    if result.is_path_changed:
        return "The tree found a better path: plan changed", PLAN_COLOR
    return "", "black"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default=SIXTEEN_PILLAR_MAP, help="map yaml file, relative to the repo root")
    parser.add_argument("--start", type=float, nargs=3, default=[1.0, 1.0, 0.0], help="x y yaw")
    parser.add_argument("--goal", type=float, nargs=2, default=[24.0, 24.0])
    parser.add_argument("--goal_radius", type=float, default=1.0)
    parser.add_argument("--paper", action="store_true", help="use Algorithm 1 of the paper as written to grow the tree")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_time", type=float, default=300.0, help="give up after this time [s]")
    parser.add_argument("--no_obstacle", action="store_true", help="do not add an obstacle during the run")
    parser.add_argument("--moving", action="store_true", help="add robots with fixed plans driving through the corridors")
    parser.add_argument("--obstacle_time", type=float, default=8.0, help="when the obstacle appears [s]")
    parser.add_argument("--obstacle_ahead", type=float, default=6.0, help="the obstacle appears this far ahead on the plan [s]")
    parser.add_argument("--obstacle_radius", type=float, default=1.5)
    parser.add_argument("--save", type=str, default=None, help="save the last frame here instead of showing it")
    parser.add_argument("--video", type=str, default=None, help="save an mp4 here instead of showing the last frame")
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
    if args.moving:
        checker.moving_obstacles = corridor_obstacles(checker, controller)
    if args.paper:
        planner = PaperCLRRT(checker, controller, goal_radius=args.goal_radius, seed=args.seed)
    else:
        planner = CLRRT(checker, controller, goal_radius=args.goal_radius, seed=args.seed)
    executor = CLRRTExecutor(planner, controller, checker, start_state, goal_pos)

    figure = ExecutionFigure(checker, start_state, goal_pos, args.goal_radius)
    writer = None
    if args.video is not None:
        # macro_block_size=2 keeps the figure size, yuv420p only needs even sizes
        writer = imageio.get_writer(
            args.video, fps=args.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=2
        )

    def write_frames(seconds: float) -> None:
        if writer is None:
            return
        figure.fig.canvas.draw()
        frame = np.asarray(figure.fig.canvas.buffer_rgba())[:, :, :3]
        for _ in range(max(int(round(seconds * args.fps)), 1)):
            writer.append_data(frame)

    trail = start_state[np.newaxis]
    num_changes, num_lazy, num_stops, num_turns = 0, 0, 0, 0
    is_obstacle_added = args.no_obstacle
    while not executor.is_in_goal() and executor.time < args.max_time:
        has_new_obstacle = False
        if not is_obstacle_added and executor.time >= args.obstacle_time:
            future = executor.future_states()
            index = int(args.obstacle_ahead / controller.dt)
            if len(future) > index:
                center = future[index, :2]
                checker.add_circle(center, args.obstacle_radius)
                figure.add_circle(center, args.obstacle_radius)
                is_obstacle_added = True
                has_new_obstacle = True
                figure.set_text(
                    "A new obstacle appears on the plan",
                    f"Time: {executor.time:.0f} s", "", "A new obstacle appeared on the plan", OBSTACLE_COLOR,
                )
                write_frames(2.0)

        cycle_time = executor.time
        result = executor.update()

        num_changes += result.is_path_changed
        num_lazy += result.num_lazy_removed
        num_stops += result.is_safety_stop
        num_turns += result.is_turning
        message, color = get_message(result, has_new_obstacle)
        figure.set_text(
            "The robot moves while the tree keeps growing",
            f"Time: {cycle_time:.0f} s,  nodes in the tree: {len(planner.tree)}",
            f"Plan changes: {num_changes},  lazy removals: {num_lazy},  safety stops: {num_stops},  turns: {num_turns}",
            message,
            color,
        )
        figure.set_tree(executor, planner)

        plan_states = executor.future_states()[len(executor.committed_states) :]
        num_executed = len(result.executed_states)
        whole_committed = np.vstack([result.executed_states, executor.committed_states])
        plan_line = np.vstack([whole_committed[-1:], plan_states])[:, :2]
        # the robot stays still for the rest of the cycle when it has nowhere to go
        state = executor.state
        for i in range(executor.steps_per_cycle):
            if i < num_executed:
                state = result.executed_states[i]
                trail = np.vstack([trail, state])
            figure.set_robot(state, trail, whole_committed[i:], plan_line)
            figure.set_moving_obstacles(cycle_time + (i + 1) * controller.dt)
            write_frames(1.0 / args.fps)

    is_reached = executor.is_in_goal()
    print(
        f"reached: {is_reached}, time: {executor.time:.0f} s, plan changes: {num_changes}, "
        f"lazy check removals: {num_lazy}, safety stops: {num_stops}, turns in place: {num_turns}"
    )
    figure.set_text(
        "Reached the goal" if is_reached else "Did not reach the goal in time",
        f"Time: {executor.time:.0f} s",
        f"Plan changes: {num_changes},  lazy removals: {num_lazy},  safety stops: {num_stops},  turns: {num_turns}",
    )
    figure.set_robot(executor.state, trail, np.empty((0, 3)), np.empty((0, 2)))
    write_frames(3.0)
    if writer is not None:
        writer.close()

    if args.save is not None:
        figure.fig.savefig(args.save, dpi=120)
    elif args.video is None:
        plt.show()
    plt.close(figure.fig)


if __name__ == "__main__":
    main()
