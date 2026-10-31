"""
    Compare a robot that knows the time-parameterized obstacles with one that ignores them

    Run from DMA-RRT/implementation:
    python3 compare_moving_obstacles.py
"""

from __future__ import annotations

import argparse
import numpy as np

from cl_rrt import CLRRT
from collision import CollisionChecker
from dynamics import PurePursuit, SkidSteerModel
from execution import CLRRTExecutor
from scenarios import ROBOT_RADIUS, corridor_obstacles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--max_time", type=float, default=300.0, help="give up after this time [s]")
    args = parser.parse_args()

    controller = PurePursuit(SkidSteerModel())
    start_state = np.array([1.0, 1.0, 0.0])
    goal_pos = np.array([24.0, 24.0])

    print("| Robot | Reached goal | Median time | 95th percentile | Runs overlapping an obstacle | Min distance to an obstacle's edge | Lazy check removals | Safety stops |")
    print("|---|---|---|---|---|---|---|---|")
    for name, is_aware in (("knows the obstacles", True), ("ignores the obstacles", False)):
        times = []
        num_overlaps, num_lazy, num_stops = 0, 0, 0
        min_dist = np.inf
        for seed in range(args.seeds):
            checker = CollisionChecker()
            obstacles = corridor_obstacles(checker, controller)
            if is_aware:
                checker.moving_obstacles = obstacles
            planner = CLRRT(checker, controller, seed=seed)
            executor = CLRRTExecutor(planner, controller, checker, start_state, goal_pos)
            run_min_dist = np.inf
            while not executor.is_in_goal() and executor.time < args.max_time:
                cycle_time, cycle_state = executor.time, executor.state
                result = executor.update()
                num_lazy += result.num_lazy_removed
                num_stops += result.is_safety_stop

                # also check while the robot is not moving, it stays at the last position
                steps = executor.steps_per_cycle
                positions = np.full((steps, 2), cycle_state[:2])
                num_executed = len(result.executed_states)
                if num_executed > 0:
                    positions[:] = result.executed_states[-1, :2]
                    positions[:num_executed] = result.executed_states[:, :2]
                step_times = cycle_time + controller.dt * np.arange(1, steps + 1)
                for obstacle in obstacles:
                    dists = np.linalg.norm(positions - obstacle.positions_at(step_times), axis=1)
                    run_min_dist = min(run_min_dist, (dists - obstacle.radius).min())

            min_dist = min(min_dist, run_min_dist)
            num_overlaps += run_min_dist < ROBOT_RADIUS
            if executor.is_in_goal():
                times.append(executor.time)
        print(
            f"| {name} | {len(times)}/{args.seeds} | {np.median(times):.1f} s | {np.percentile(times, 95):.1f} s "
            f"| {num_overlaps} | {min_dist:.3f} m | {num_lazy} | {num_stops} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
