"""
    Compare CL-RRT as written in the paper (PaperCLRRT) with our version (CLRRT) and the effect of each difference

    Run from DMA-RRT/implementation:
    python3 compare_cl_rrt.py
"""

from __future__ import annotations

import argparse
import time
import numpy as np

from cl_rrt import CLRRT
from cl_rrt_paper import PaperCLRRT
from collision import CollisionChecker
from dynamics import PurePursuit, SkidSteerModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--max_iter", type=int, default=5000)
    args = parser.parse_args()

    checker = CollisionChecker()
    controller = PurePursuit(SkidSteerModel())
    start_state = np.array([1.0, 1.0, 0.0])
    goal_pos = np.array([24.0, 24.0])

    variants = {
        "paper as written": lambda seed: PaperCLRRT(checker, controller, seed=seed),
        "paper + goal sampling 5%": lambda seed: PaperCLRRT(checker, controller, goal_sample_rate=0.05, seed=seed),
        "paper + k=5": lambda seed: PaperCLRRT(checker, controller, num_nearest=5, seed=seed),
        "paper + k=5 + goal sampling 5%": lambda seed: PaperCLRRT(checker, controller, goal_sample_rate=0.05, num_nearest=5, seed=seed),
        "ours": lambda seed: CLRRT(checker, controller, seed=seed),
        "ours with k=1": lambda seed: CLRRT(checker, controller, num_nearest=1, seed=seed),
        "ours without goal sampling": lambda seed: CLRRT(checker, controller, goal_sample_rate=0.0, seed=seed),
    }

    print("| Variant | Reached goal | Trees under 100 nodes | Median travel time | Median plan time | Min clearance | Reproduced from waypoints |")
    print("|---|---|---|---|---|---|---|")
    for name, make_planner in variants.items():
        num_reached, num_small, num_reproduced = 0, 0, 0
        costs, plan_times = [], []
        min_clearance = np.inf
        for seed in range(args.seeds):
            planner = make_planner(seed)
            begin = time.perf_counter()
            goal_node = planner.plan(start_state, goal_pos, max_iter=args.max_iter)
            plan_times.append(time.perf_counter() - begin)
            num_small += len(planner.tree) < 100

            for node in planner.tree.nodes[1:]:
                states = np.vstack([node.parent.state[np.newaxis], node.trajectory])
                points = np.vstack(
                    [np.linspace(a, b, 5) for a, b in zip(states[:-1, :2], states[1:, :2])]
                )
                min_clearance = min(min_clearance, checker.obstacle_dists(points).min())

            if goal_node is None:
                continue
            num_reached += 1
            costs.append(goal_node.cost)
            waypoints = [node.waypoint for node in planner.tree.nodes_to(goal_node)[1:]]
            num_reproduced += np.array_equal(
                planner.path_trajectory(goal_node),
                planner.simulate_waypoints(start_state, waypoints),
            )
        print(
            f"| {name} | {num_reached}/{args.seeds} | {num_small} | {np.median(costs):.1f} s "
            f"| {np.median(plan_times) * 1e3:.0f} ms | {min_clearance:.3f} m | {num_reproduced}/{num_reached} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
