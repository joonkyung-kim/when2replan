"""
    Compare the synchronous cycles of Team with the robots running their own cycles

    Run from DMA-RRT/implementation:
    python3 -m experiments.compare_async
    python3 -m experiments.compare_async --scenes open_10 --variants same_every async_every async_fast_better
    python3 -m experiments.compare_async --paper_tree --clip_each_wheel --paper_strict --coordinations round_robin merit --variants timed_fast_better
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
import os
from typing import Dict, Tuple
import numpy as np

from dma_rrt.async_team import AsyncTeam
from dma_rrt.dynamics import PurePursuit, SkidSteerModel
from dma_rrt.scenarios import make_scene
from experiments.compare_team import MIN_DIST, count_conflicts

# name: same phase, time between the checks for a bid [s], bids only when better, plan for a cycle with the token
VARIANTS = {
    "same_every": (True, 1.0, False, False),
    "async_every": (False, 1.0, False, False),
    "async_better": (False, 1.0, True, False),
    "async_fast": (False, 0.2, False, False),
    "async_fast_better": (False, 0.2, True, False),
    "timed_every": (False, 1.0, False, True),
    "timed_fast": (False, 0.2, False, True),
    "timed_fast_better": (False, 0.2, True, True),
}
# (coordination, variant)
DEFAULT_RUNS = (
    ("round_robin", "same_every"),
    ("round_robin", "async_every"),
    ("merit", "same_every"),
    ("merit", "async_every"),
    ("merit", "async_better"),
    ("merit", "async_fast"),
    ("merit", "async_fast_better"),
    ("cooperative", "async_fast_better"),
    ("cooperative_paper", "async_fast_better"),
    ("round_robin", "timed_every"),
    ("merit", "timed_every"),
    ("merit", "timed_fast"),
    ("merit", "timed_fast_better"),
    ("cooperative", "timed_fast_better"),
    ("cooperative_paper", "timed_fast_better"),
)


def run_seed(args: Tuple[str, str, str, int, float, bool, bool, bool, float, bool]) -> Dict:
    """
    One run of a team
    :param args: scene, coordination, variant, seed, time of the run [s], tree of Algorithm 1, can turn in place,
        limit each wheel on its own, L1 [m], paper_strict of AsyncTeam
    """
    scene, coordination, variant, seed, max_time, paper_tree, can_turn, clip_each_wheel, lookahead, paper_strict = args
    same_phase, bid_interval, only_when_better, plan_for_a_cycle = VARIANTS[variant]
    controller = PurePursuit(SkidSteerModel(clip_each_wheel=clip_each_wheel), lookahead=lookahead)
    start_states, goals, map_path = make_scene(scene)
    team = AsyncTeam(
        start_states, goals, controller, coordination=coordination, map_path=map_path, seed=seed,
        same_phase=same_phase, bid_interval=bid_interval, bids_only_when_better=only_when_better,
        plan_for_a_cycle=plan_for_a_cycle, paper_tree=paper_tree, can_turn=can_turn, paper_strict=paper_strict,
    )
    agents = team.agents
    pairs = [(agent, other) for agent in agents for other in agents if other is not agent]
    result = dict(min_dist=np.inf, conflicts=0, wrong=0, beliefs=0, stops=0)
    last_positions = np.array(start_states)[:, :2]
    stopped_steps = np.zeros(len(agents), dtype=int)
    longest_stop = np.zeros(len(agents), dtype=int)
    while team.time < max_time:
        next_time = np.array([team.time + controller.dt])
        # what each robot believes about the others at the next step, compared with what they really do
        beliefs = [agent.believed_positions(other.agent_id, next_time)[0] for agent, other in pairs]
        team.step()
        positions = np.array([agent.executor.state_at(team.time)[:2] for agent in agents])
        stopped_steps = np.where((positions == last_positions).all(axis=1), stopped_steps + 1, 0)
        longest_stop = np.maximum(longest_stop, stopped_steps)
        last_positions = positions
        for (_, other), belief in zip(pairs, beliefs):
            result["beliefs"] += 1
            result["wrong"] += not np.array_equal(belief, positions[other.agent_id])
        dists = np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis], axis=2)
        result["min_dist"] = min(result["min_dist"], dists[np.triu_indices(len(agents), k=1)].min())
        if team.is_plan_sent:
            result["conflicts"] += count_conflicts(team) > 0
    result["goals"] = [agent.num_goals_reached for agent in agents]
    result["longest_stop"] = longest_stop.max() * controller.dt
    result["plans"] = team.num_plans
    result["changed_plans"] = team.num_changed_plans
    result["bids"] = team.num_bids
    result["estops"] = team.num_estops
    result["tokens"] = team.token_counts
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=str, nargs="+", default=["pillar_4", "pillar_8", "open_10", "hallway_2", "four_agents_4"])
    parser.add_argument("--coordinations", type=str, nargs="+", default=None, help="instead of the default runs, all variants of these")
    parser.add_argument("--variants", type=str, nargs="+", default=None, choices=list(VARIANTS))
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--max_time", type=float, default=600.0, help="time of a run, ten minutes as in the paper [s]")
    parser.add_argument("--paper_tree", action="store_true", help="grow the trees as in Algorithm 1")
    parser.add_argument("--no_turning", action="store_true", help="never use the rule of turning in place, as in the paper")
    parser.add_argument("--clip_each_wheel", action="store_true", help="limit each wheel on its own, as stated in Sec. 7.1")
    parser.add_argument("--lookahead", type=float, default=1.5, help="L1, not given in the paper [m]")
    parser.add_argument("--paper_strict", action="store_true", help="commit only up to t + delta t, random first token, no bid reset. Only merit and round_robin (step 11)")
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() - 2, 1), help="number of processes")
    args = parser.parse_args()

    runs_to_do = list(DEFAULT_RUNS)
    if args.coordinations is not None or args.variants is not None:
        runs_to_do = [
            (coordination, variant)
            for coordination in (args.coordinations or ["merit"])
            for variant in (args.variants or list(VARIANTS))
        ]
    print("| Scene | Coordination | Variant | Goals per robot | 95% interval | Min distance between centers | Runs under limit | Conflicts after plans | Wrong beliefs | Plans per s | Plans that changed | Bids per s, per robot | Emergency stops | Token share, most / least | Longest stop in a run, median / max |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    with Pool(args.workers) as pool:
        for scene in args.scenes:
            for coordination, variant in runs_to_do:
                runs = pool.map(
                    run_seed,
                    [
                        (scene, coordination, variant, seed, args.max_time, args.paper_tree, not args.no_turning, args.clip_each_wheel, args.lookahead, args.paper_strict)
                        for seed in range(args.seeds)
                    ],
                )
                run_means = [np.mean(run["goals"]) for run in runs]
                half_interval = 1.96 * np.std(run_means, ddof=1) / np.sqrt(len(runs))
                min_dists = [run["min_dist"] for run in runs]
                num_robots = len(runs[0]["goals"])
                total = {key: sum(run[key] for run in runs) for key in ("conflicts", "wrong", "beliefs", "plans", "changed_plans", "bids", "estops")}
                shares = np.array([run["tokens"] for run in runs], dtype=float)
                shares = shares / shares.sum(axis=1, keepdims=True)
                run_time = len(runs) * args.max_time
                print(
                    f"| {scene} | {coordination} | {variant} | {np.mean(run_means):.2f} | [{np.mean(run_means) - half_interval:.2f}, {np.mean(run_means) + half_interval:.2f}] "
                    f"| {min(min_dists):.3f} m | {sum(dist < MIN_DIST for dist in min_dists)}/{args.seeds} | {total['conflicts']}/{total['plans']} | {total['wrong']}/{total['beliefs']} "
                    f"| {total['plans'] / run_time:.2f} | {100 * total['changed_plans'] / total['plans']:.0f}% | {total['bids'] / run_time / num_robots:.2f} | {total['estops']} "
                    f"| {np.median(shares.max(axis=1)) * 100:.0f}% / {np.median(shares.min(axis=1)) * 100:.0f}% "
                    f"| {np.median([run['longest_stop'] for run in runs]):.0f} / {max(run['longest_stop'] for run in runs):.0f} s |",
                    flush=True,
                )


if __name__ == "__main__":
    main()
