"""
    Who replans: the merit token of DMA-RRT in when2replan's multi robot run

    Four robots on the sixteen pillar map swap corners (scripts/run_multi_robot.py, no moving obstacles, the other
    robots are seen by lidar), or 2 to 8 robots with --robots: robots 4-7 start in the middle of the edges. Instead of RL or a rule deciding when each robot replans, a token decides who:

    - The cycle is 1 s, the time when2replan's global planner takes (global_planner_calculation_interval = 10 steps).
    - At the start of every cycle each robot computes its bid, the potential improvement of eq. 5 of DMA-RRT:
        cost of the current path - cost of a new Dijkstra path, from the robot's position, in meters
      The current path costs its remaining length, plus BLOCKED_PENALTY if it now runs into an obstacle or another
      robot seen by lidar. The new path starts where DWA predicts the robot to be when the path arrives (planner
      "committed", experiments/when2replan_committed_start.py). Bids below MIN_IMPROVEMENT are 0.
    - Only the token holder may replan, and only if its bid is above 0. The others keep their paths.
    - The token then goes to the active robot with the best bid of this cycle, at random for a tie,
      excluding the holder (DMA-RRT, "merit"). A robot that is done leaves the team.

    The variants combine the replanning method and the start of the new path:
      time_current:     when2replan's time_based_replan, every robot replans every 1 s from its position (baseline)
      time_committed:   the same, from the predicted position
      merit_current:    the token, from the position
      merit_committed:  the token, from the predicted position (DMA-RRT like)
      merit_shared:     merit_committed, and the robot that replans broadcasts its path with its start time, which the
                        others keep out of in Dijkstra and DWA (experiments/when2replan_shared_plans.py). A path in
                        conflict with a shared plan costs CONFLICT_PENALTY more in the bids
      merit_measured, rr_measured: *_shared with the speed of each other robot measured over the last second,
                        instead of SPEED for all
      rr_committed, rr_shared: the same with a round-robin token (the baseline of Sec. 7.2.1 of DMA-RRT): the token
                        goes to the next active robot in a fixed order, whatever the bids. The holder still replans only
                        if its bid is above 0

    Run from DMA-RRT/implementation with the when2replan311 env:
    python3 -m experiments.when2replan_merit_multi_robot
    python3 -m experiments.when2replan_merit_multi_robot --variants merit_committed --seed 1
    python3 -m experiments.when2replan_merit_multi_robot --variants time_current merit_committed merit_shared
    python3 -m experiments.when2replan_merit_multi_robot --robots 8 --variants time_current merit_shared rr_shared
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import pickle
import shutil
import subprocess
import sys
import threading
from typing import Dict, List, Optional

import numpy as np
import yaml

from dma_rrt.collision import REPO_DIR
from experiments.when2replan_committed_start import CommittedStartPlanner, LocalPlannerRecorder, load_yaml
from experiments import when2replan_shared_plans as shared

NAVIGATION_CONFIG = "config/run/multi_robot.yaml"
RL_CONFIG = "config/rl_params/default.yaml"
SCENARIOS = [f"config/scenario/sixteen_pillar/multi_robot/robot{i}.yaml" for i in range(8)]
# robot 0 and the others in robot 0's video, the colors of when2replan's visualizer
ROBOT_COLORS = ["#2c962c", "#8c564b", "#e377c2", "#17becf", "#28288c", "#00806e", "#6e6e6e", "#8c1450"]
VARIANTS = ["time_current", "time_committed", "merit_current", "merit_committed", "merit_shared", "rr_committed",
            "rr_shared", "merit_measured", "rr_measured"]
TOKEN_METHODS = ("merit", "rr")
MIN_IMPROVEMENT = 0.5  # [m] smaller improvements are noise of the grid path
BLOCKED_PENALTY = 100.0  # [m] cost added to a path that runs into an obstacle now


class MeritToken:
    """
    The token and the bids, shared by the robot threads. Only one thread computes at a time (compute lock of
    RobotStateExchanger), so no lock is needed here. It uses its own random generator, not the numpy global one,
    which the robots' Dijkstra planners use
    """

    def __init__(self, num_robots: int, seed: int, cycle_steps: int, round_robin: bool = False) -> None:
        self._num_robots = num_robots
        self._round_robin = round_robin
        self._rng = np.random.default_rng(seed)
        self._cycle_steps = cycle_steps
        self._holders: Dict[int, int] = {0: 0}  # cycle -> token holder, robot 0 starts as in DMA-RRT's Team
        self.bids: Dict[int, Dict[int, float]] = {}  # cycle -> robot -> bid
        self.replans: Dict[int, List[int]] = {}  # cycle -> robots that replanned
        self.done_step: Dict[int, int] = {}  # robot -> env step it was done at

    def holder(self, cycle: int) -> Optional[int]:
        """
        Token holder of the cycle, selected from the bids of the previous cycle by the previous holder
        """
        if cycle not in self._holders:
            previous = self.holder(cycle - 1) if cycle - 1 in self._holders else None
            bids = {r: b for r, b in self.bids.get(cycle - 1, {}).items() if not self.is_done(r, cycle)}
            candidates = {r: b for r, b in bids.items() if r != previous} or bids
            if self._round_robin and candidates:
                # the next active robot after the previous holder, the bids only decide if the holder replans
                last = previous if previous is not None else -1
                order = [(last + k) % self._num_robots for k in range(1, self._num_robots + 1)]
                self._holders[cycle] = next(r for r in order if r in bids)
            elif not candidates:
                # no bids yet (the first cycle starts after the reset step) or no other robot left: keep the token
                self._holders[cycle] = previous if previous is not None and not self.is_done(previous, cycle) else None
            else:
                best = max(candidates.values())
                # sorted: the bids are posted in the order the threads happen to compute in
                ties = sorted(r for r, b in candidates.items() if b == best)
                self._holders[cycle] = int(self._rng.choice(ties))
        return self._holders[cycle]

    def is_done(self, robot_id: int, cycle: int) -> bool:
        """
        Done before this cycle started. A robot done in the same step is seen as active by every thread
        """
        return self.done_step.get(robot_id, np.inf) < cycle * self._cycle_steps

    def post_bid(self, cycle: int, robot_id: int, bid: float) -> None:
        self.bids.setdefault(cycle, {})[robot_id] = bid


def path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0


class PostingPlanner(CommittedStartPlanner):
    """
    Broadcasts every path the robot takes, with the simulator step it is planned at
    """

    def __init__(self, *args, robot_id: int, board, clock: List[int], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._robot_id = robot_id
        self._board = board
        self._clock = clock

    def make_plan(self, start_pos: np.ndarray, goal_pos: np.ndarray):
        indices, path = super().make_plan(start_pos, goal_pos)
        if path is not None and len(indices) > 1:
            self._board.post(self._robot_id, self._clock[0], np.array(path[0]))
        return indices, path


def compute_bid(env, recorder: LocalPlannerRecorder, planner: CommittedStartPlanner, committed: bool, safe_dist: float,
                board=None, robot_id: int = 0, num_robots: int = 0, clock: Optional[List[int]] = None):
    """
    Potential improvement of replanning now [m], and the parts used for it
    With a board, a path in conflict with the shared plans of the others costs CONFLICT_PENALTY more
    """
    obs = env._simulator.get_observation()
    pos = np.asarray(obs.state.pos, dtype=float)
    cost_map = obs.static_map_with_scan

    current = env._previous_global_path[0]
    nearest = int(np.argmin(np.linalg.norm(current - pos, axis=1)))
    remaining = current[nearest:]
    is_blocked = bool(np.any(cost_map.get_obs_dist_array(remaining) < safe_dist - cost_map.get_resolution()))
    cost_current = np.linalg.norm(remaining[0] - pos) + path_length(remaining) + (BLOCKED_PENALTY if is_blocked else 0.0)
    is_conflict = False
    if board is not None:
        step = clock[0]
        seen = shared.seen_robots(env, robot_id, num_robots)
        tracker = planner._planner.tracker
        speeds = tracker.speeds(seen) if tracker is not None else None
        is_conflict = shared.first_conflict(board, remaining, step * board.dt, seen, step, speeds) is not None
        cost_current += shared.CONFLICT_PENALTY if is_conflict else 0.0

    start = pos
    if committed and len(recorder.history) > 0 and recorder.history[-1][1] is not None:
        start = recorder.history[-1][1][-1, :2]  # DWA's prediction of the last step ends at the end of this step + 0.9 s
    inner = planner._planner
    inner.set_costmap(cost_map)
    if board is not None:
        # the new path is used from the end of this cycle on
        _, path = inner.make_plan(start, env._goal_state.pos, t0=(clock[0] + recorder_latency(planner)) * board.dt)
    else:
        _, path = inner.make_plan(start, env._goal_state.pos)
    if path is None or len(path[0]) <= 1:
        return 0.0, cost_current, np.inf, is_blocked, is_conflict
    cost_new = np.linalg.norm(start - pos) + path_length(path[0])
    if board is not None and inner.last_conflict:
        cost_new += shared.CONFLICT_PENALTY
    bid = cost_current - cost_new
    return (bid if bid > MIN_IMPROVEMENT else 0.0), cost_current, cost_new, is_blocked, is_conflict


def recorder_latency(planner: CommittedStartPlanner) -> int:
    return planner._latency_steps


def run_robot(robot_id, method, committed, env, recorder, planner, token, exchanger, results, cycle_steps, safe_dist,
              board=None, clock=None, num_robots=0):
    """
    run_multi_robot.run_robot with the replanning method of the variant
    """
    exchanger.acquire(robot_id)
    try:
        frames, log = [], []
        min_dist = np.inf
        recorder.clear()
        if clock is not None:
            clock[0] = 0
        env.reset()
        done = False
        while not done and env._step_count < env.get_max_steps():
            step = env._step_count
            if method in TOKEN_METHODS:
                action = 0
                if step % cycle_steps == 0:
                    cycle = step // cycle_steps
                    holder = token.holder(cycle)
                    bid, cost_current, cost_new, is_blocked, is_conflict = compute_bid(
                        env, recorder, planner, committed, safe_dist, board, robot_id, num_robots, clock
                    )
                    token.post_bid(cycle, robot_id, bid)
                    if holder == robot_id and bid > 0.0:
                        action = 1
                        token.replans.setdefault(cycle, []).append(robot_id)
                    log.append(dict(cycle=cycle, holder=holder, bid=round(float(bid), 2), replan=action,
                                    blocked=is_blocked, conflict=is_conflict, cost_current=round(float(cost_current), 2),
                                    cost_new=round(float(cost_new), 2)))
            else:
                action = env.time_based_replan()
            _, _, done, info = env.step(action)
            frames.extend(info["visualized_data_list"])
            min_dist = min(min_dist, env.get_dist_to_other_robots())
        token.done_step[robot_id] = env._step_count
        if board is not None:
            board.done_step[robot_id] = clock[0]

        final_state = env.get_robot_state()
        all_done = False
        while not all_done:
            _, all_done = exchanger.exchange(robot_id, final_state, is_done=True)
        data = dict(info["data"])
        data["min_dist_to_other_robots"] = min_dist
        results[robot_id] = (data, frames, final_state, log)
    except Exception as e:
        exchanger.abort()
        results[robot_id] = e
    finally:
        exchanger.release(robot_id)


def subfolder(out_dir: str, kind: str, name: str) -> str:
    """
    Path of one output file, sorted by kind: grid, videos, token, results or replay (the visualizer's pkl files)
    """
    os.makedirs(os.path.join(out_dir, kind), exist_ok=True)
    return os.path.join(out_dir, kind, name)


def run_variant(variant: str, seed: int, out_dir: str, num_robots: int = 4, video: bool = True,
                speed: Optional[float] = None) -> Optional[str]:
    """
    :param speed: assumed speed along the shared paths instead of shared.SPEED, for this robot's own path too
    """
    if speed is not None:
        shared.SPEED = speed
    sys.path.insert(0, str(REPO_DIR / "scripts"))
    import run_multi_robot as mr
    from stable_baselines3.common.utils import set_random_seed

    method, start = variant.split("_")
    is_shared = start in ("shared", "measured")
    committed = start in ("committed", "shared", "measured")
    navigation_config = load_yaml(NAVIGATION_CONFIG)["navigation"]
    common = navigation_config["common"]
    cycle_steps = common["global_planner_calculation_interval"]
    safe_dist = common["robot_radius"] + common["inflation_radius"]

    set_random_seed(seed)
    exchanger = mr.RobotStateExchanger(num_robots, seed)
    token = MeritToken(num_robots, seed, cycle_steps, round_robin=method == "rr")
    board = shared.PlanBoard(num_robots, common["control_interval_time"]) if is_shared else None
    clocks = [[0] for _ in range(num_robots)]
    envs, recorders, planners, trackers = [], [], [], []
    for robot_id, scenario in enumerate(SCENARIOS[:num_robots]):
        env = mr.MultiRobotNavigationStackEnv(
            robot_id=robot_id, exchanger=exchanger, strict_robot_collision=False,
            navigation_config=navigation_config, scenario_list=[load_yaml(scenario)], seed=seed,
            visualize_mode="none", rl_config=load_yaml(RL_CONFIG), save_visualized_data=True,
        )
        env.seed(seed)
        mode = "committed" if committed else "current"
        if is_shared:
            clock = clocks[robot_id]
            tracker = shared.SpeedTracker(robot_id, num_robots, board, env, clock, measured=start == "measured")
            trackers.append(tracker)
            shared.count_steps(env._simulator, clock, tracker)
            shared.patch_dwa(env._local_planner, robot_id, num_robots, board, env, clock, tracker)
            recorder = LocalPlannerRecorder(env._local_planner)
            timed = shared.TimedDijkstra(env._global_planner, robot_id, num_robots, board, env, clock, tracker)
            planner = PostingPlanner(timed, recorder, mode, cycle_steps - 1, robot_id=robot_id, board=board, clock=clock)
        else:
            recorder = LocalPlannerRecorder(env._local_planner)
            planner = CommittedStartPlanner(env._global_planner, recorder, mode, cycle_steps - 1)
        env._local_planner = recorder
        env._global_planner = planner
        envs.append(env)
        recorders.append(recorder)
        planners.append(planner)

    results = [None] * num_robots
    threads = [
        threading.Thread(
            target=run_robot,
            args=(i, method, committed, envs[i], recorders[i], planners[i], token, exchanger, results, cycle_steps,
                  safe_dist, board, clocks[i] if is_shared else None, num_robots),
        )
        for i in range(num_robots)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for result in results:
        if isinstance(result, Exception):
            raise result

    # same frame handling as run_multi_robot.run: park the robots that are done, add the others to every frame
    num_frames = max(len(frames) for _, frames, _, _ in results)
    for _, frames, final_state, _ in results:
        frames.extend([mr.get_parked_frame(frames[-1], final_state) for _ in range(num_frames - len(frames))])
    frames = results[0][1]
    for k, frame in enumerate(frames):
        frame["other_robots"] = [mr.get_robot_frame(j, results[j][1][k]) for j in range(1, num_robots)]

    video_path = None
    if video:
        pkl_path = subfolder(out_dir, "replay", f"{variant}_seed{seed}_data.pkl")
        map_path = subfolder(out_dir, "replay", f"{variant}_seed{seed}_map.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(map_path, "wb") as f:
            pickle.dump(envs[0].get_static_map_polygons(), f, protocol=pickle.HIGHEST_PROTOCOL)
        video_path = subfolder(out_dir, "videos", f"{variant}_seed{seed}.mp4")
        mr.pygame_visualizer.main(pkl_path, map_pkl=map_path, save_path=video_path, headless=True)

    lines = [f"== {variant} ({num_robots} robots, seed {seed})"]
    summary = []
    for robot_id, (data, _, _, log) in enumerate(results):
        outcome = "goal" if data["reach_goal"] else "collision" if data["collision"] else "timeout"
        lines.append(
            f"robot {robot_id}: {outcome:9s} time {data['time']:5.1f} s, replans {data['num_update_global_planner']:3d}, "
            f"travel {data['travel_distance']:5.1f} m, min distance to other robots {data['min_dist_to_other_robots']:.2f} m"
        )
        summary.append(dict(robot=robot_id, outcome=outcome, time=float(data["time"]),
                            replans=int(data["num_update_global_planner"]), travel=float(data["travel_distance"]),
                            min_dist=float(data["min_dist_to_other_robots"]), cycles=log))
        if trackers:
            summary[-1]["prediction_errors"] = {
                float(h): [round(e, 3) for e in trackers[robot_id].errors[h]] for h in shared.ERROR_HORIZONS
            }
    if method in TOKEN_METHODS:
        holders = [token.holder(c) for c in sorted(token.bids)]
        counts = {r: holders.count(r) for r in range(num_robots)}
        replans = {r: sum(r in v for v in token.replans.values()) for r in range(num_robots)}
        lines.append(f"token held (cycles): {counts}, replanned with it: {replans}, cycles: {len(holders)}")
    if board is not None:
        repairs = [p._planner.num_repairs for p in planners]
        lines.append(f"messages: {board.num_messages} plans, {board.num_floats * 4 / 1000:.1f} kB as float32, "
                     f"Dijkstra repairs around shared plans per robot: {repairs}")
        errors = {h: np.concatenate([t.errors[h] for t in trackers]) for h in shared.ERROR_HORIZONS}
        lines.append("prediction error of the other robots: " + ", ".join(
            f"{h:.0f} s ahead mean {np.mean(e):.2f} m, 90% {np.percentile(e, 90):.2f} m" for h, e in errors.items()))
        if video:
            plot_token(token, results, num_robots, subfolder(out_dir, "token", f"{variant}_seed{seed}_token.png"))
    with open(subfolder(out_dir, "results", f"{variant}_seed{seed}.yaml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print("\n".join(lines), flush=True)
    return video_path


def plot_token(token: MeritToken, results, num_robots: int, save_path: str) -> None:
    """
    Timeline of the token: holder, bids and replans of every cycle
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ROBOT_COLORS
    cycles = sorted(token.bids)
    fig, (ax, ax_bid) = plt.subplots(2, 1, figsize=(14, 6.5 + 0.4 * max(num_robots - 4, 0)), sharex=True,
                                     gridspec_kw=dict(height_ratios=[1, 1.3]))
    for c in cycles:
        holder = token.holder(c)
        if holder is not None:
            ax.add_patch(plt.Rectangle((c - 0.45, holder - 0.4), 0.9, 0.8, color=colors[holder], alpha=0.35))
        for r in token.replans.get(c, []):
            ax.plot(c, r, "*", color="black", ms=11)
    for r in range(num_robots):
        log = results[r][3]
        xs = [e["cycle"] for e in log]
        ax_bid.plot(xs, [e["bid"] for e in log], "-o", ms=3, color=colors[r], label=f"robot {r}")
        blocked = [e["cycle"] for e in log if e["blocked"]]
        ax_bid.plot(blocked, [-3] * len(blocked), "|", ms=10, color=colors[r])
        if log:
            ax.plot([log[-1]["cycle"] + 0.6], [r], ">", color=colors[r], ms=8)
    ax.set_yticks(range(num_robots), [f"robot {r}" for r in range(num_robots)])
    ax.set_ylim(-0.6, num_robots - 0.4)
    ax.set_title("token holder of each 1 s cycle (shaded), replans (*), end of the robot's run (>)", loc="left")
    ax_bid.set_ylabel("bid = improvement [m]")
    ax_bid.set_xlabel("cycle = time [s]")
    ax_bid.set_yscale("symlog", linthresh=5)
    ax_bid.set_title(f"bids (0 below {MIN_IMPROVEMENT} m, about {BLOCKED_PENALTY:.0f} when the current path is blocked), "
                     "| at the bottom: current path blocked", loc="left")
    ax_bid.legend(loc="upper right", fontsize=8)
    ax_bid.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def grid_video(videos: List[str], labels: List[str], save_path: str) -> None:
    """
    The variants next to each other (3) or in a 2 x 2 grid (4) with their names, the shorter videos hold their
    last frame
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or len(videos) not in (3, 4):
        return
    inputs = []
    for video in videos:
        inputs += ["-i", video]
    parts = []
    for i, label in enumerate(labels):
        parts.append(
            f"[{i}:v]tpad=stop=-1:stop_mode=clone,scale=700:-2,"
            f"drawtext=text='{label}':x=(w-tw)/2:y=h-40:fontsize=28:fontcolor=black:box=1:boxcolor=white@0.8[v{i}]"
        )
    durations = [float(subprocess.run(
        [shutil.which("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", v],
        capture_output=True, text=True, check=True).stdout) for v in videos]
    if len(videos) == 4:
        filter_graph = ";".join(parts) + ";[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0"
    else:
        filter_graph = ";".join(parts) + ";[v0][v1][v2]hstack=inputs=3"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", *inputs, "-filter_complex", filter_graph, "-t", str(max(durations)),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", save_path],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=VARIANTS[:4], choices=VARIANTS)
    parser.add_argument("--robots", type=int, default=4, choices=range(2, 9), help="robots 0 to N - 1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="default: outputs/when2replan_merit_multi_robot/<N>robot, <N>robot_speed<S> with --speed")
    parser.add_argument("--no_video", action="store_true", help="only the results, faster")
    parser.add_argument("--speed", type=float, default=None,
                        help=f"assumed speed along the shared paths [m/s], default {shared.SPEED}")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join("outputs/when2replan_merit_multi_robot", f"{args.robots}robot")
        if args.speed is not None:
            args.out_dir += f"_speed{args.speed:g}"

    # one process per variant: the robots of a run share threads and the numpy global random state
    with multiprocessing.get_context("spawn").Pool(len(args.variants)) as pool:
        videos = pool.starmap(
            run_variant,
            [(v, args.seed, args.out_dir, args.robots, not args.no_video, args.speed) for v in args.variants],
        )
    if not args.no_video and len(videos) in (3, 4):
        save_path = subfolder(args.out_dir, "grid", f"grid_{'_'.join(args.variants)}_seed{args.seed}.mp4"
                              if len(videos) == 3 else f"grid_seed{args.seed}.mp4")
        grid_video(videos, args.variants, save_path)
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
