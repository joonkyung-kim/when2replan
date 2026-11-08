"""
    How DMA-RRT keeps time: the shared clock, the root time, and the times of the nodes and edges

    Two robots swap their corners on the sixteen pillar map. Robot 0 is drawn in detail:
    - map: its tree, its plan with the arrival time of each node on the shared clock, the committed part it can no
      longer change, and where robot 0 believes robot 1 is at the times robot 0 reaches its next nodes
    - table: node_time = root_time + node.cost - root.cost for the nodes of the plan
    - timeline: committed part, waiting at the root and the node times of both robots on the shared clock
    - root time over the clock: flat while a robot follows its plan, a step when the root moves on to the next
      node, and the line now + 1 s while a robot waits at its root

    One video frame is one control step (0.1 s), so the video runs in real time at 10 fps.

    Run from DMA-RRT/implementation:
    python3 -m demos.demo_time
    python3 -m demos.demo_time --commit_one_cycle   # Algorithm 2: commit only up to t + delta t
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import imageio
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from dma_rrt.dynamics import PurePursuit, SkidSteerModel
from dma_rrt.scenarios import make_scene
from dma_rrt.team import Team
from dma_rrt.tree import Node

COLORS = ["tab:blue", "tab:orange"]
DARK = ["navy", "saddlebrown"]
FOCUS = 0  # robot drawn in detail
OTHER = 1
NUM_PAIRED_NODES = 3  # next nodes of robot 0 paired with robot 1's believed position at the same time
TIMELINE_AHEAD = 14.0  # [s]


class Snapshot:
    """
    What one robot looks like during one cycle, taken after Team.step
    """

    def __init__(self, team: Team, agent_id: int, cycle_start: float, executed: np.ndarray) -> None:
        agent = team.agents[agent_id]
        executor, planner = agent.executor, agent.planner
        dt = executor._dt
        self.cycle_start = cycle_start
        self.executed = executed.reshape(-1, 3)
        # committed at the start of the cycle = executed in this cycle + what is still committed
        self.committed = np.vstack((self.executed, executor.committed_states))
        self.commit_end = cycle_start + len(self.committed) * dt
        self.root_time = planner.root_time
        self.plan: List[Node] = list(executor.plan_nodes)
        self.root_cost = planner.tree.root.cost
        self.node_times = [planner.node_time(node) for node in self.plan]
        self.tree_edges = [
            np.vstack((node.parent.state[:2], node.trajectory[:, :2]))
            for node in planner.tree.nodes
            if node.parent is not None and node.trajectory is not None and len(node.trajectory) > 0
        ]


class TimeFigure:
    def __init__(self, team: Team, commit_one_cycle: bool, max_time: float) -> None:
        self._team = team
        self._names: Dict[int, str] = {}
        self._root_history = [[], []]  # (clock at the start of a cycle, root time) of each robot
        self.fig = plt.figure(figsize=(16, 9), dpi=100)

        # map
        ax = self.fig.add_axes([0.02, 0.06, 0.44, 0.88])
        checker = team.agents[0].checker
        checker.draw(ax)
        ax.set_xlim(checker.area_limits[0], checker.area_limits[1])
        ax.set_ylim(checker.area_limits[2], checker.area_limits[3])
        ax.set_aspect("equal")
        ax.autoscale(False)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        self._map = ax
        self._tree = LineCollection([], colors="silver", linewidths=0.7, zorder=2)
        ax.add_collection(self._tree)
        self._map_artists = []
        self._map_title = ax.set_title("", fontsize=13, fontweight="bold")

        # text
        panel = self.fig.add_axes([0.48, 0.52, 0.51, 0.46])
        panel.axis("off")
        panel.set_xlim(0, 1)
        panel.set_ylim(0, 1)
        mode = "commit_one_cycle=True: commit up to t + 1 s (Algorithm 2)" if commit_one_cycle else (
            "commit_one_cycle=False: commit whole edges"
        )
        panel.text(0.0, 0.97, "DMA-RRT time: shared clock, root_time, node times", fontsize=15, fontweight="bold")
        panel.text(0.0, 0.905, f"cycle (replan) = 1.0 s, control step dt = 0.1 s, {mode}", fontsize=10.5)
        self._text = panel.text(0.0, 0.86, "", fontsize=10.5, family="monospace", va="top")

        # timeline
        self._timeline = self.fig.add_axes([0.52, 0.29, 0.46, 0.2])
        # root time over the clock
        self._root_ax = self.fig.add_axes([0.52, 0.05, 0.46, 0.18])
        self._max_time = max_time

    def name(self, node: Node) -> str:
        if id(node) not in self._names:
            self._names[id(node)] = f"n{len(self._names)}"
        return self._names[id(node)]

    def new_cycle(self, snapshots: List[Snapshot], holder: int, message) -> None:
        self._snapshots = snapshots
        self._holder = holder
        self._message = message
        for i, snap in enumerate(snapshots):
            self._root_history[i].append((snap.cycle_start, snap.root_time))
        for node in snapshots[FOCUS].plan:
            self.name(node)
        self._tree.set_segments(snapshots[FOCUS].tree_edges)

    def draw(self, step: int, positions: List[np.ndarray], trails: List[np.ndarray]) -> None:
        snaps = self._snapshots
        dt = self._team.agents[0].executor._dt
        now = snaps[0].cycle_start + (step + 1) * dt
        for artist in self._map_artists:
            artist.remove()
        self._map_artists = []
        self._draw_map(now, step, positions, trails)
        self._draw_text(now, step)
        self._draw_timeline(now, step)
        self._draw_root_time(now)
        self._map_title.set_text(f"shared clock t = {now:5.1f} s")

    def _add(self, artist):
        self._map_artists.append(artist)
        return artist

    def _draw_map(self, now: float, step: int, positions: List[np.ndarray], trails: List[np.ndarray]) -> None:
        ax = self._map
        for i, snap in enumerate(self._snapshots):
            color, dark = COLORS[i], DARK[i]
            trail = trails[i]
            self._add(ax.plot(trail[:, 0], trail[:, 1], "-", color=color, lw=1, alpha=0.4, zorder=3)[0])
            remaining = snap.committed[step + 1 :, :2]
            if len(remaining) > 0:
                line = np.vstack((positions[i], remaining))
                self._add(ax.plot(line[:, 0], line[:, 1], "-", color=dark, lw=4.5, zorder=5, solid_capstyle="round")[0])
            for node in snap.plan[1:]:
                traj = np.vstack((node.parent.state[:2], node.trajectory[:, :2]))
                self._add(ax.plot(traj[:, 0], traj[:, 1], "--", color=color, lw=2, zorder=4)[0])
            is_focus = i == FOCUS
            for k, (node, t) in enumerate(zip(snap.plan, snap.node_times)):
                x, y = node.state[:2]
                if k == 0:
                    self._add(ax.plot(x, y, "o", ms=13, mfc="none", mec=dark, mew=2.5, zorder=7)[0])
                label = f"{self.name(node)} {t:.1f}s" if is_focus else f"{t:.1f}s"
                if k == 0:
                    label = ("ROOT " if is_focus else "root ") + label
                self._add(ax.plot(x, y, "s" if is_focus else "^", ms=6, color=dark, zorder=7)[0])
                self._add(ax.text(x + 0.35, y + 0.35, label, fontsize=8.5 if is_focus else 7.5, color=dark,
                                  fontweight="bold" if is_focus else None, zorder=9,
                                  bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75)))
            self._add(ax.add_patch(plt.Circle(positions[i], 1.0, color=color, alpha=0.75, zorder=6)))
            yaw = snap.committed[step, 2] if step < len(snap.committed) else snap.committed[-1, 2] if len(snap.committed) else 0.0
            tip = positions[i] + np.array([np.cos(yaw), np.sin(yaw)])
            self._add(ax.plot([positions[i][0], tip[0]], [positions[i][1], tip[1]], "-", color="black", lw=2, zorder=8)[0])
            self._add(ax.text(positions[i][0], positions[i][1], str(i), ha="center", va="center", fontsize=10,
                              color="white", fontweight="bold", zorder=9))

        # robot 0's belief of robot 1 at the times robot 0 reaches its next nodes
        focus = self._snapshots[FOCUS]
        agent = self._team.agents[FOCUS]
        pairs = [(node, t) for node, t in zip(focus.plan, focus.node_times) if t > now][:NUM_PAIRED_NODES]
        believed_now = agent.believed_positions(OTHER, np.array([now]))[0]
        self._add(ax.add_patch(plt.Circle(believed_now, 1.0, fill=False, ls="-", lw=1.5, color="black", zorder=8)))
        for node, t in pairs:
            other = agent.believed_positions(OTHER, np.array([t]))[0]
            self._add(ax.add_patch(plt.Circle(other, 1.0, fill=False, ls="--", lw=1.3, color=COLORS[OTHER], zorder=6)))
            self._add(ax.plot([node.state[0], other[0]], [node.state[1], other[1]], ":", color="gray", lw=1.2, zorder=5)[0])
            self._add(ax.text(other[0], other[1] - 1.45, f"1 @ {t:.1f}s", ha="center", fontsize=7.5,
                              color=DARK[OTHER], zorder=9))
        legend = [
            plt.Line2D([], [], color=DARK[0], lw=4.5, label="committed (cannot change)"),
            plt.Line2D([], [], color=COLORS[0], lw=2, ls="--", label="plan after the root"),
            plt.Line2D([], [], color="silver", lw=1, label="tree of robot 0"),
            plt.Line2D([], [], marker="o", ms=11, mfc="none", mec=DARK[0], mew=2.5, ls="none", label="root"),
            plt.Line2D([], [], marker="o", ms=11, mfc="none", mec=COLORS[1], ls="none",
                       label="robot 1 as robot 0 believes, at its node times"),
            plt.Line2D([], [], marker="o", ms=11, mfc="none", mec="black", ls="none", label="... at now (same as real)"),
        ]
        self._add(ax.legend(handles=legend, loc="upper left", fontsize=8, framealpha=0.9))

    def _draw_text(self, now: float, step: int) -> None:
        dt = self._team.agents[0].executor._dt
        lines = []
        start = self._snapshots[0].cycle_start
        phase = "begin_cycle: commit, reroot, set root_time" if step == 0 else f"_move: step {step + 1}/10 of the cycle"
        lines.append(f"cycle [{start:.1f}, {start + 1:.1f}) s   {phase}")
        msg = self._message
        lines.append(
            f"token: robot {self._holder} sent PlanMessage(start_state=its root, start_time={msg.start_time:.1f}, "
            f"{len(msg.waypoints)} waypoints)"
        )
        lines.append("")
        for i, snap in enumerate(self._snapshots):
            committed_time = len(snap.committed) * dt
            lines.append(
                f"robot {i}: root_time = cycle start + max(committed, 1 s) = {snap.cycle_start:.1f} + "
                f"max({committed_time:.1f}, 1.0) = {snap.root_time:.1f}"
                + ("   (waits at root)" if committed_time < 1.0 - 1e-9 else "")
            )
        lines.append("")
        focus = self._snapshots[FOCUS]
        lines.append("robot 0 plan   cost [s]   node_time = root_time + cost - root.cost")
        for k, (node, t) in enumerate(zip(focus.plan, focus.node_times)):
            if k >= 8:
                lines.append(f"  ... {len(focus.plan) - k} more")
                break
            name = self.name(node) + (" (root)" if k == 0 else "")
            lines.append(
                f"  {name:10s} {node.cost:7.1f}    {focus.root_time:5.1f} + {node.cost:5.1f} - {focus.root_cost:5.1f} = {t:5.1f}"
            )
        self._text.set_text("\n".join(lines))

    def _draw_timeline(self, now: float, step: int) -> None:
        ax = self._timeline
        ax.clear()
        dt = self._team.agents[0].executor._dt
        for row, (i, snap) in zip([1, 0], enumerate(self._snapshots)):
            ax.barh(row, snap.commit_end - now, left=now, height=0.42, color=DARK[i], alpha=0.85)
            if snap.root_time > snap.commit_end + 1e-9:
                ax.barh(row, snap.root_time - snap.commit_end, left=snap.commit_end, height=0.42, color="lightgray",
                        hatch="//", edgecolor="gray")
            if len(snap.plan) > 1:
                ax.barh(row, snap.node_times[-1] - snap.root_time, left=snap.root_time, height=0.18, color=COLORS[i],
                        alpha=0.5)
            for k, (node, t) in enumerate(zip(snap.plan, snap.node_times)):
                ax.plot(t, row, "D" if k == 0 else "|", color="black" if k == 0 else DARK[i], ms=8 if k == 0 else 14,
                        mew=2)
                if i == FOCUS and k < 9:
                    ax.text(t, row + 0.3, self.name(node), ha="center", fontsize=7.5, color=DARK[i])
        ax.axvline(now, color="red", lw=1.5)
        ax.text(now, 1.62, "now", color="red", ha="center", fontsize=9)
        ax.set_xlim(now - 1.0, now + TIMELINE_AHEAD)
        ax.set_ylim(-0.5, 1.8)
        ax.set_yticks([1, 0], ["robot 0", "robot 1"])
        ax.set_xlabel("shared clock [s]", fontsize=9)
        ax.set_title("committed (dark) | waiting at root (hatched) | root (diamond) and node times (ticks)",
                     fontsize=9.5, loc="left")
        ax.grid(axis="x", alpha=0.3)

    def _draw_root_time(self, now: float) -> None:
        ax = self._root_ax
        ax.clear()
        clock = np.array([0.0, self._max_time])
        ax.plot(clock, clock + 1.0, "--", color="gray", lw=1, label="now + 1 s (earliest root_time)")
        for i in range(2):
            history = np.array(self._root_history[i] + [(now, self._root_history[i][-1][1])])
            ax.step(history[:, 0], history[:, 1], where="post", color=COLORS[i], lw=2, label=f"root_time of robot {i}")
        ax.axvline(now, color="red", lw=1)
        left = max(0.0, now - 15.0)
        ax.set_xlim(left, left + 18.0)
        ax.set_ylim(left, left + 24.0)
        ax.set_xlabel("shared clock [s]", fontsize=9)
        ax.set_ylabel("root_time [s]", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit_one_cycle", action="store_true", help="commit only up to t + delta t (Algorithm 2)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_time", type=float, default=55.0, help="time of the run [s]")
    parser.add_argument("--fps", type=int, default=10, help="10 is real time")
    parser.add_argument("--out_dir", type=str, default="outputs/time_explained")
    parser.add_argument("--frame", type=float, default=None, help="also save the frame at this time as a png")
    args = parser.parse_args()

    controller = PurePursuit(SkidSteerModel())
    start_states, goals, map_path = make_scene("pillar_2")  # (1, 1) <-> (24, 24) on the diagonal
    team = Team(start_states, goals, controller, coordination="merit", map_path=map_path, seed=args.seed,
                commit_one_cycle=args.commit_one_cycle)
    figure = TimeFigure(team, args.commit_one_cycle, args.max_time)

    os.makedirs(args.out_dir, exist_ok=True)
    name = "time_commit_one_cycle" if args.commit_one_cycle else "time_commit_edges"
    video_path = os.path.join(args.out_dir, f"{name}.mp4")
    writer = imageio.get_writer(video_path, fps=args.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=2)

    steps = team.agents[0].executor.steps_per_cycle
    positions = [state[:2].copy() for state in start_states]
    trails = [state[:2][np.newaxis] for state in start_states]
    while team.time < args.max_time - 1e-9:
        cycle_start = team.time
        holder = team.token_holder
        results = team.step()
        message = team.agents[holder].make_plan_message(0)
        snapshots = [Snapshot(team, i, cycle_start, result.executed_states) for i, result in enumerate(results)]
        figure.new_cycle(snapshots, holder, message)
        for k in range(steps):
            for i, result in enumerate(results):
                if k < len(result.executed_states):
                    positions[i] = result.executed_states[k][:2]
                    trails[i] = np.vstack((trails[i], positions[i]))
            figure.draw(k, positions, trails)
            figure.fig.canvas.draw()
            writer.append_data(np.asarray(figure.fig.canvas.buffer_rgba())[:, :, :3])
            now = cycle_start + (k + 1) * controller.dt
            if args.frame is not None and abs(now - args.frame) < 0.5 * controller.dt:
                png = os.path.join(args.out_dir, f"{name}_{args.frame:.1f}s.png")
                figure.fig.savefig(png)
                print(f"Saved {png}")
    writer.close()
    print(f"Saved {video_path}, goals reached: {[agent.num_goals_reached for agent in team.agents]}")


if __name__ == "__main__":
    main()
