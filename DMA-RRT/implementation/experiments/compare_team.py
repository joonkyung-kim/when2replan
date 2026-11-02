"""
    Compare DMA-RRT (bids) with the round-robin token and no coordination

    Run from DMA-RRT/implementation:
    python3 -m experiments.compare_team
    python3 -m experiments.compare_team --scenes pillar_8 open_10 --seeds 24 --workers 22
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
import os
from typing import Dict, List, Tuple
import numpy as np

from dma_rrt.dynamics import PurePursuit, SkidSteerModel
from dma_rrt.scenarios import make_scene
from dma_rrt.team import COORDINATIONS, Team

MIN_DIST = 2.0 * 1.0 + 0.1  # two robots of radius 1.0 m and the margin


def count_conflicts(team: Team) -> int:
    """
    Number of pairs of robots whose current plans come closer than MIN_DIST at the same time
    """
    plans = [agent.current_plan() for agent in team.agents]
    dt = plans[0].dt
    num_steps = max(len(plan.positions) for plan in plans)
    times = team.time + dt * np.arange(num_steps)
    positions = [plan.positions_at(times) for plan in plans]
    num_conflicts = 0
    for i in range(len(plans)):
        for j in range(i + 1, len(plans)):
            num_conflicts += np.linalg.norm(positions[i] - positions[j], axis=1).min() < MIN_DIST
    return num_conflicts


def get_positions(step_count: int, state: np.ndarray, executed_states: np.ndarray) -> np.ndarray:
    """
    Position of a robot at each control step of a cycle, it stays at the last position when it does not move
    :return: positions size: (step_count, 2)
    """
    positions = np.full((step_count, 2), state[:2])
    if len(executed_states) > 0:
        positions[:] = executed_states[-1, :2]
        positions[: len(executed_states)] = executed_states[:, :2]
    return positions


def make_team(scene: str, coordination: str, seed: int, controller: PurePursuit, stop_interval: float) -> Team:
    """
    :param scene: name for make_scene
    """
    start_states, goal_lists, map_path = make_scene(scene)
    return Team(
        start_states, goal_lists, controller, coordination=coordination, map_path=map_path, seed=seed,
        stop_interval=stop_interval,
    )


def run_seed(args: Tuple[str, str, int, float, float]) -> Dict:
    """
    One run of a team
    :param args: scene, coordination, seed, time of the run [s], time between emergency stop nodes [s]
    """
    scene, coordination, seed, max_time, stop_interval = args
    controller = PurePursuit(SkidSteerModel())
    team = make_team(scene, coordination, seed, controller, stop_interval)
    agents = team.agents
    num_robots = len(agents)
    steps = agents[0].executor.steps_per_cycle
    result = dict(min_dist=np.inf, conflict_cycles=0, cycles=0, wrong=0, beliefs=0, stops=0, lazy=0)
    while team.time < max_time:
        states = [agent.executor.state for agent in agents]
        times = team.time + controller.dt * np.arange(1, steps + 1)
        # what each robot believes about the others, compared with what they really do in this cycle
        beliefs = [
            (agent, other, agent.believed_positions(other.agent_id, times))
            for agent in agents
            for other in agents
            if other is not agent
        ]
        cycle_results = team.step()
        positions = [
            get_positions(steps, state, cycle_result.executed_states)
            for state, cycle_result in zip(states, cycle_results)
        ]
        for _, other, belief in beliefs:
            # the estop message comes after these beliefs were taken, and a robot can stop right after it
            if other.agent_id == team.estop_target:
                continue
            result["beliefs"] += 1
            result["wrong"] += not np.array_equal(belief, positions[other.agent_id])
        for i in range(num_robots):
            for j in range(i + 1, num_robots):
                dists = np.linalg.norm(positions[i] - positions[j], axis=1)
                result["min_dist"] = min(result["min_dist"], dists.min())
        result["cycles"] += 1
        result["conflict_cycles"] += count_conflicts(team) > 0
        result["stops"] += sum(cycle_result.is_safety_stop for cycle_result in cycle_results)
        result["lazy"] += sum(cycle_result.num_lazy_removed for cycle_result in cycle_results)
    result["goals"] = [agent.num_goals_reached for agent in agents]
    result["tokens"] = team.token_counts
    result["estops"] = team.num_estops
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=str, nargs="+", default=["pillar_2", "pillar_4", "pillar_6", "pillar_8", "open_10", "hallway_2", "four_agents_4"])
    parser.add_argument("--coordinations", type=str, nargs="+", default=list(COORDINATIONS), choices=COORDINATIONS)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--max_time", type=float, default=600.0, help="time of a run, ten minutes as in the paper [s]")
    parser.add_argument("--stop_interval", type=float, default=4.0, help="time between emergency stop nodes, 4 s in the paper [s]")
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() - 2, 1), help="number of processes")
    args = parser.parse_args()

    print(f"| Scene | Coordination | Goals per robot in {args.max_time:.0f} s | 95% interval | Std. dev. between robots | Min distance between centers | Runs under {MIN_DIST} m | Cycles with conflicting plans | Wrong beliefs | Safety stops | Lazy check removals | Emergency stops | Token share, most / least |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    with Pool(args.workers) as pool:
        for scene in args.scenes:
            for coordination in args.coordinations:
                runs = pool.map(
                    run_seed,
                    [(scene, coordination, seed, args.max_time, args.stop_interval) for seed in range(args.seeds)],
                )
                goals = [goal for run in runs for goal in run["goals"]]
                run_means = [np.mean(run["goals"]) for run in runs]
                half_interval = 1.96 * np.std(run_means, ddof=1) / np.sqrt(len(runs))
                min_dists = [run["min_dist"] for run in runs]
                num_close = sum(dist < MIN_DIST for dist in min_dists)
                total = {key: sum(run[key] for run in runs) for key in ("conflict_cycles", "cycles", "wrong", "beliefs", "stops", "lazy", "estops")}
                token_shares = _token_shares(runs)
                print(
                    f"| {scene} | {coordination} | {np.mean(goals):.2f} | [{np.mean(run_means) - half_interval:.2f}, {np.mean(run_means) + half_interval:.2f}] | {np.std(goals):.2f} | {min(min_dists):.3f} m | {num_close}/{args.seeds} "
                    f"| {total['conflict_cycles']}/{total['cycles']} | {total['wrong']}/{total['beliefs']} | {total['stops']} | {total['lazy']} | {total['estops']} | {token_shares} |",
                    flush=True,
                )


def _token_shares(runs: List[Dict]) -> str:
    """
    Median over the runs of the share of the cycles the most and the least frequent token holders had the token
    """
    tokens = np.array([run["tokens"] for run in runs], dtype=float)
    if tokens.sum() == 0:
        return "-"
    shares = tokens / tokens.sum(axis=1, keepdims=True)
    return f"{np.median(shares.max(axis=1)) * 100:.0f}% / {np.median(shares.min(axis=1)) * 100:.0f}%"


if __name__ == "__main__":
    main()
