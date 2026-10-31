"""
    Compare the two CL-RRT versions inside the execution loop, with and without turning in place

    Unlike compare_cl_rrt.py, which asks if one planning call reaches the goal, this moves the robot
    and keeps growing the tree, which is how the paper uses CL-RRT.

    Run from DMA-RRT/implementation:
    python3 compare_execution.py
"""

from __future__ import annotations

import argparse
import numpy as np

from cl_rrt import CLRRT
from cl_rrt_paper import PaperCLRRT
from collision import CollisionChecker
from dynamics import PurePursuit, SkidSteerModel
from execution import CLRRTExecutor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--max_time", type=float, default=300.0, help="give up after this time [s]")
    args = parser.parse_args()

    controller = PurePursuit(SkidSteerModel())
    start_state = np.array([1.0, 1.0, 0.0])
    goal_pos = np.array([24.0, 24.0])

    # (name, planner class, turn rate, whether an obstacle appears)
    variants = [
        ("paper's tree expansion, no turning", PaperCLRRT, 0.0, False),
        ("paper's tree expansion, turning", PaperCLRRT, 1.0, False),
        ("ours, no turning", CLRRT, 0.0, False),
        ("ours, turning", CLRRT, 1.0, False),
        ("ours, turning, new obstacle", CLRRT, 1.0, True),
        ("ours, no turning, new obstacle", CLRRT, 0.0, True),
    ]

    print("| Variant | Reached goal | Median time | 95th percentile | Runs that got stuck | Lazy check removals | Safety stops | Min clearance |")
    print("|---|---|---|---|---|---|---|---|")
    for name, planner_class, turn_rate, has_obstacle in variants:
        times = []
        num_turn_runs, num_lazy, num_stops = 0, 0, 0
        min_clearance = np.inf
        for seed in range(args.seeds):
            checker = CollisionChecker()
            planner = planner_class(checker, controller, seed=seed)
            executor = CLRRTExecutor(
                planner, controller, checker, start_state, goal_pos, turn_rate=turn_rate
            )
            is_obstacle_added = not has_obstacle
            has_turned = False
            while not executor.is_in_goal() and executor.time < args.max_time:
                if not is_obstacle_added and executor.time >= 8.0:
                    future = executor.future_states()
                    if len(future) > 60:
                        checker.add_circle(future[60, :2], 1.5)
                        is_obstacle_added = True
                result = executor.update()
                has_turned |= result.is_turning
                num_lazy += result.num_lazy_removed
                num_stops += result.is_safety_stop
                if len(result.executed_states) > 0:
                    min_clearance = min(
                        min_clearance, checker.obstacle_dists(result.executed_states[:, :2]).min()
                    )
            num_turn_runs += has_turned
            if executor.is_in_goal():
                times.append(executor.time)
        print(
            f"| {name} | {len(times)}/{args.seeds} | {np.median(times):.1f} s | {np.percentile(times, 95):.1f} s "
            f"| {num_turn_runs} | {num_lazy} | {num_stops} | {min_clearance:.3f} m |",
            flush=True,
        )


if __name__ == "__main__":
    main()
