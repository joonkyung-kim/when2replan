"""
    A team of robots planning with their own trees and sharing their plans

    Run from DMA-RRT/implementation:
    python3 -m demos.demo_team --video outputs/team.mp4
    python3 -m demos.demo_team --robots 8 --video outputs/team8.mp4
    python3 -m demos.demo_team --open_map --video outputs/team_open.mp4  # ten robots
    python3 -m demos.demo_team --coordination round_robin --video outputs/team_rr.mp4
    python3 -m demos.demo_team --coordination cooperative --video outputs/team_cooperative.mp4
    python3 -m demos.demo_team --scene hallway_2 --coordination cooperative --video outputs/team_hallway.mp4
    python3 -m demos.demo_team --open_map --asynchronous --video outputs/team_async_open.mp4
    python3 -m demos.demo_team --scene four_agents_4 --asynchronous --paper --coordination merit --video outputs/team_paper_four_agents.mp4
    python3 -m demos.demo_team --scene four_agents_4 --asynchronous --paper --paper_strict --coordination merit --video outputs/team_strict_four_agents.mp4
"""

from __future__ import annotations

import argparse
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from dma_rrt.async_team import AsyncTeam
from dma_rrt.dynamics import PurePursuit, SkidSteerModel
from dma_rrt.scenarios import make_scene
from dma_rrt.team import COORDINATIONS, Team

COLORS = ["tab:blue", "tab:orange", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "tab:green", "navy", "darkgoldenrod"]


class TeamFigure:
    def __init__(self, team: Team, coordination: str, asynchronous: bool = False, paper_strict: bool = False) -> None:
        self._team = team
        self._coordination = coordination
        self.fig = plt.figure(figsize=(12, 6.75), dpi=120)
        self._ax = self.fig.add_axes([0.05, 0.08, 0.5, 0.82])
        panel = self.fig.add_axes([0.58, 0.0, 0.42, 1.0])
        panel.set_xlim(0.0, 1.0)
        panel.set_ylim(0.0, 1.0)
        panel.axis("off")

        ax = self._ax
        checker = team.agents[0].checker
        checker.draw(ax)
        self._tree_lines = LineCollection([], colors="silver", linewidths=0.7)
        ax.add_collection(self._tree_lines)
        self._robots, self._headings, self._plans, self._trails, self._goals = [], [], [], [], []
        for agent, color in zip(team.agents, COLORS):
            (plan,) = ax.plot([], [], "-", color=color, linewidth=2.5, zorder=4)
            (trail,) = ax.plot([], [], "-", color=color, linewidth=1.0, alpha=0.4, zorder=3)
            (goal,) = ax.plot([], [], "*", color=color, markersize=16, markeredgecolor="black", zorder=7)
            robot = plt.Circle(agent.executor.state[:2], 1.0, color=color, alpha=0.8, zorder=6)
            ax.add_patch(robot)
            (heading,) = ax.plot([], [], "-", color="black", linewidth=2, zorder=8)
            self._robots.append(robot)
            self._headings.append(heading)
            self._plans.append(plan)
            self._trails.append(trail)
            self._goals.append(goal)
        (self._stop_dots,) = ax.plot([], [], "o", color="dimgray", markersize=5, zorder=5)
        self._token_ring = plt.Circle((0.0, 0.0), 1.45, fill=False, color="tab:red", linewidth=3, zorder=8)
        ax.add_patch(self._token_ring)
        self._token_ring.set_visible(coordination != "none")
        ax.set_xlim(checker.area_limits[0], checker.area_limits[1])
        ax.set_ylim(checker.area_limits[2], checker.area_limits[3])
        ax.autoscale(False)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        self._title = ax.set_title("", fontsize=12, fontweight="bold")

        name = {"merit": "DMA-RRT (token goes to the best bid)", "merit_negative": "DMA-RRT, bids can be negative", "merit_wait": "DMA-RRT, bids add waiting time", "cooperative_paper": "Cooperative (as in the paper)", "cooperative": "Cooperative DMA-RRT (ours)", "cooperative_negative": "Cooperative, negative bids", "cooperative_wait": "Cooperative, bids add wait", "round_robin": "Round-robin token", "none": "No coordination"}[coordination]
        panel.text(0.02, 0.945, f"Team: {name}", fontsize=14, fontweight="bold")
        self._status = panel.text(0.02, 0.895, "", fontsize=11)
        self._rows = []
        for i, (agent, color) in enumerate(zip(team.agents, COLORS)):
            self._rows.append(panel.text(0.02, 0.85 - i * 0.034, "", fontsize=10, color=color, fontweight="bold"))
        if asynchronous and paper_strict:
            lines = [
                "Each robot has its own cycle of 1 s",
                "1. The red ring has the token. It changes its plan 1 s after",
                "    getting it, even in the middle of its own cycle.",
                "2. It sends its waypoints and the robot with the best bid.",
                "3. The others keep growing their trees and send a bid",
                "    every 0.2 s. A new bid replaces the old one.",
                "4. Each robot commits to its plan for 1 s only.",
            ]
        elif asynchronous:
            lines = [
                "Each robot has its own cycle of 1 s",
                "1. The red ring has the token. It changes its plan 1 s after",
                "    getting it, even in the middle of its own cycle.",
                "2. It sends its waypoints and the robot with the best bid.",
                "3. The others keep growing their trees. They send a bid",
                "    when they find a better path, checking every 0.2 s.",
                "4. The bids of a robot that sent a plan go back to 0.",
            ]
        elif coordination.startswith("cooperative"):
            lines = [
                "Each cycle (1 s)",
                "1. The red ring has the token. If its best path is blocked",
                "    by a robot, it compares cutting its own path short",
                "    with asking that robot to stop at a stop node (dots).",
                "2. A robot asked to stop gets the token next.",
                "3. Others must not plan through a stop node after the",
                "    robot could be there, so stop nodes block space.",
            ]
        elif coordination in ("merit", "merit_negative", "merit_wait"):
            lines = [
                "Each cycle (1 s)",
                "1. The red ring has the token. Only this robot changes",
                "    its plan, against the plans the others shared.",
                "2. It sends its waypoints and the robot with the best bid.",
                "3. The others keep their plans and grow their trees.",
                "    Each sends a bid: its current plan's cost minus the",
                "    cost of the best path in its tree.",
            ]
        elif coordination == "round_robin":
            lines = [
                "Each cycle (1 s)",
                "1. The red ring has the token. Only this robot changes",
                "    its plan, against the plans the others shared.",
                "2. It sends its waypoints to the others. They simulate",
                "    them to get its trajectory, an obstacle in time.",
                "3. The others keep their plans and grow their trees.",
                "4. The token goes to the next robot in a fixed order.",
            ]
        else:
            lines = [
                "Each cycle (1 s)",
                "1. All robots change their plans at the same time,",
                "    against the old plans of the others.",
                "2. They share the new plans only afterwards.",
                "    New plans can conflict with each other.",
            ]
        for i, text in enumerate(lines):
            panel.text(
                0.02, 0.50 - i * 0.04, text, fontsize=10.5, fontweight="bold" if i == 0 else "normal"
            )
        panel.text(
            0.02, 0.19,
            "Thick line: current plan. Star: current goal.\nGray: tree of the robot with the token.\nLight red: a robot's center must not enter.",
            fontsize=9.5, color="dimgray", va="top",
        )

    def set_plans(self) -> None:
        team = self._team
        for agent, plan_line, goal in zip(team.agents, self._plans, self._goals):
            positions = agent.current_plan().positions
            plan_line.set_data(positions[:, 0], positions[:, 1])
            goal.set_data([agent.goal_pos[0]], [agent.goal_pos[1]])
        stop_positions = [
            node.position
            for agent in team.agents
            if hasattr(agent, "own_stop_nodes")
            for node in agent.own_stop_nodes
            if node.time >= agent.planner.root_time - 1e-6
        ]
        if len(stop_positions) > 0:
            self._stop_dots.set_data(*np.array(stop_positions).T)
        else:
            self._stop_dots.set_data([], [])
        for agent, row, in zip(team.agents, self._rows):
            tokens = team.token_counts[agent.agent_id]
            row.set_text(f"Robot {agent.agent_id}: {agent.num_goals_reached} goals, had the token {tokens} times")

    def set_tree(self, agent_id: int) -> None:
        planner = self._team.agents[agent_id].planner
        self._tree_lines.set_segments(
            [
                np.vstack([node.parent.state[np.newaxis], node.trajectory])[:, :2]
                for node in planner.tree.nodes[1:]
            ]
        )

    def set_robots(self, states, trails, token_holder: int, time: float) -> None:
        for i, state in enumerate(states):
            self._robots[i].center = (state[0], state[1])
            self._headings[i].set_data(
                [state[0], state[0] + np.cos(state[2])], [state[1], state[1] + np.sin(state[2])]
            )
            self._trails[i].set_data(trails[i][:, 0], trails[i][:, 1])
        self._token_ring.center = (states[token_holder][0], states[token_holder][1])
        self._status.set_text(f"Time: {time:.1f} s")
        if self._coordination != "none":
            if self._team.estop_target is not None:
                self._title.set_text(f"Robot {token_holder} asks robot {self._team.estop_target} to stop early")
            else:
                self._title.set_text(f"Robot {token_holder} has the token")
        else:
            self._title.set_text("All robots change their plans in every cycle")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", type=int, default=4, choices=[2, 4, 6, 8], help="robots on the sixteen pillar map")
    parser.add_argument("--open_map", action="store_true", help="ten robots on the open map, instead of --robots")
    parser.add_argument("--scene", type=str, default=None, help="scene name like hallway_2, instead of --robots")
    parser.add_argument("--coordination", type=str, default="merit", choices=COORDINATIONS)
    parser.add_argument("--asynchronous", action="store_true", help="each robot runs its own cycles (step 9)")
    parser.add_argument("--paper", action="store_true", help="tree of Algorithm 1, each wheel limited on its own (step 10). The rule of turning in place stays, see the README")
    parser.add_argument("--paper_strict", action="store_true", help="with --asynchronous, also commit only up to t + delta t, random first token, no bid reset (step 11)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_time", type=float, default=100.0, help="time of the run [s]")
    parser.add_argument("--save", type=str, default=None, help="save the last frame here instead of showing it")
    parser.add_argument("--video", type=str, default=None, help="save an mp4 here instead of showing the last frame")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    controller = PurePursuit(SkidSteerModel(clip_each_wheel=args.paper))
    scene = args.scene if args.scene is not None else "open_10" if args.open_map else f"pillar_{args.robots}"
    start_states, goals, map_path = make_scene(scene)
    team_class = AsyncTeam if args.asynchronous else Team
    options = dict(paper_strict=True) if args.paper_strict else {}
    if args.paper_strict and not args.asynchronous:
        parser.error("--paper_strict needs --asynchronous")
    team = team_class(
        start_states, goals, controller, coordination=args.coordination, map_path=map_path, seed=args.seed,
        paper_tree=args.paper, **options,
    )
    figure = TeamFigure(team, args.coordination, asynchronous=args.asynchronous, paper_strict=args.paper_strict)
    writer = None
    if args.video is not None:
        # macro_block_size=2 keeps the figure size, yuv420p only needs even sizes
        writer = imageio.get_writer(
            args.video, fps=args.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=2
        )

    trails = [state[:2][np.newaxis] for state in start_states]
    steps = team.agents[0].executor.steps_per_cycle
    min_dist = np.inf
    states = list(start_states)
    while args.asynchronous and team.time < args.max_time:
        team.step()
        figure.set_plans()
        figure.set_tree(team.token_holder)
        states = [agent.executor.state_at(team.time) for agent in team.agents]
        trails = [np.vstack([trail, state[:2]]) for trail, state in zip(trails, states)]
        figure.set_robots(states, trails, team.token_holder, team.time)
        positions = np.array([state[:2] for state in states])
        dists = np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis], axis=2)
        min_dist = min(min_dist, dists[np.triu_indices(len(states), k=1)].min())
        if writer is not None:
            figure.fig.canvas.draw()
            writer.append_data(np.asarray(figure.fig.canvas.buffer_rgba())[:, :, :3])

    while team.time < args.max_time:
        cycle_time = team.time
        token_holder = team.token_holder
        results = team.step()
        figure.set_plans()
        figure.set_tree(token_holder)
        positions = []
        for i in range(steps):
            for j, result in enumerate(results):
                if i < len(result.executed_states):
                    states[j] = result.executed_states[i]
                    trails[j] = np.vstack([trails[j], states[j][:2]])
            positions.append(np.array([state[:2] for state in states]))
            figure.set_robots(states, trails, token_holder, cycle_time + (i + 1) * controller.dt)
            if writer is not None:
                figure.fig.canvas.draw()
                writer.append_data(np.asarray(figure.fig.canvas.buffer_rgba())[:, :, :3])
        for frame in positions:
            for a in range(len(frame)):
                for b in range(a + 1, len(frame)):
                    min_dist = min(min_dist, np.linalg.norm(frame[a] - frame[b]))

    goals_reached = [agent.num_goals_reached for agent in team.agents]
    print(f"goals reached: {goals_reached}, min distance between centers: {min_dist:.3f} m, emergency stops: {team.num_estops}")
    if writer is not None:
        writer.close()

    if args.save is not None:
        figure.fig.savefig(args.save, dpi=120)
    elif args.video is None:
        plt.show()
    plt.close(figure.fig)


if __name__ == "__main__":
    main()
