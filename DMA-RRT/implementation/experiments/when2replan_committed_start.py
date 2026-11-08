"""
    One when2replan robot replanning from where it will be when the new path arrives, like the root of DMA-RRT

    when2replan emulates a global planner that takes 1 s: at time t the robot asks for a path from its position p(t),
    keeps driving on the old path for 0.9 s, and then switches to the new path, which still starts at p(t), behind it.
    DMA-RRT plans from the state at the end of the committed part instead (Algorithm 2), so a new plan starts where
    the robot is when it gets it.

    "committed" mode does the same in when2replan, without changing the env:
    - at time t, DWA has predicted its next second (10 steps). Its state after 9 steps, p_hat, is where the robot
      is expected to be when the new path is used.
    - the global planner plans from p_hat on the map of time t, and the path is
      p(t) + DWA's predicted states up to p_hat (the committed part) + the new path from p_hat.
    - if planning from p_hat fails, it plans from p(t) as before.
    "current" mode is when2replan as it is.

    Measured for each mode: goal, collision, goal time, replans, and at every switch to a new path
    the distance from the robot to the new path, and the error of p_hat.

    Run from DMA-RRT/implementation with the when2replan311 env:
    python3 -m experiments.when2replan_committed_start --seeds 50
    python3 -m experiments.when2replan_committed_start --video time_based_replan 1   # both modes side by side
"""

from __future__ import annotations

import argparse
import os
import random
from collections import deque
from multiprocessing import Pool
from typing import Dict, List, Optional

import numpy as np
import yaml

from dma_rrt.collision import REPO_DIR

NAVIGATION_CONFIG = "config/run/default.yaml"
SCENARIO = "config/scenario/sixteen_pillar/random.yaml"  # random start/goal, 10 random moving obstacles
RL_CONFIG = "config/rl_params/default.yaml"
MODEL = "trained_model/DijkstraxDWA/16pillar.zip"
METHODS = ["rl_based_replan", "time_based_replan", "stuck_based_replan", "distance_based_replan"]
MODES = ["current", "committed"]


class LocalPlannerRecorder:
    """
    Wraps the local planner: keeps DWA's prediction of each step, and measures the robot at the first step
    with a new path
    """

    def __init__(self, planner) -> None:
        self._planner = planner
        self.history = deque(maxlen=32)  # (robot position, predicted positions (10, 2) or None)
        self.positions: List[np.ndarray] = []
        self._expected: Optional[dict] = None
        self.switch_offsets: List[float] = []  # distance from the robot to the new path when it is used [m]
        self.prediction_errors: List[float] = []  # distance from the robot to p_hat at that time [m]

    def __getattr__(self, name):
        return getattr(self._planner, name)

    def clear(self) -> None:
        self.history.clear()
        self.positions = []
        self._expected = None

    def expect_switch(self, p_hat: Optional[np.ndarray]) -> None:
        self._expected = {"p_hat": p_hat}

    def compute_velocity_command(self, robot_obs, goal_state, reference_path):
        pos = np.array(robot_obs.state.pos, dtype=float)
        if self._expected is not None and reference_path is not None:
            self.switch_offsets.append(float(np.min(np.linalg.norm(reference_path[0] - pos, axis=1))))
            if self._expected["p_hat"] is not None:
                self.prediction_errors.append(float(np.linalg.norm(pos - self._expected["p_hat"])))
            self._expected = None
        output = self._planner.compute_velocity_command(robot_obs, goal_state, reference_path)
        predicted = None
        if output.best_index is not None and output.best_index >= 0 and output.predict_path_list is not None:
            predicted = np.array(output.predict_path_list[output.best_index])
        self.history.append((pos, predicted))
        self.positions.append(pos)
        return output


class CommittedStartPlanner:
    """
    Wraps the global planner: in "committed" mode, plans from DWA's predicted position at the time of the switch
    """

    def __init__(self, planner, recorder: LocalPlannerRecorder, mode: str, latency_steps: int) -> None:
        self._planner = planner
        self._recorder = recorder
        self._mode = mode
        self._latency_steps = latency_steps  # steps on the old path before the switch, 9
        self._map = None
        self.num_committed = 0
        self.num_fallbacks = 0

    def __getattr__(self, name):
        return getattr(self._planner, name)

    def set_costmap(self, cost_map) -> None:
        self._map = cost_map
        self._planner.set_costmap(cost_map)

    def get_costmap(self):
        return self._planner.get_costmap()

    def _prediction_at_request(self, start: np.ndarray) -> Optional[np.ndarray]:
        """
        DWA's prediction made at the request time t, i.e. at the first of the steps on the old path
        :return: predicted positions at t + 0.1 ... t + 1.0 size: (10, 2), None for a replan without latency
        """
        history = self._recorder.history
        if len(history) < self._latency_steps:
            return None
        pos, predicted = history[-self._latency_steps]
        if predicted is None or not np.allclose(pos, start, atol=1e-9):
            return None
        return predicted[:, :2]

    def make_plan(self, start_pos: np.ndarray, goal_pos: np.ndarray):
        predicted = self._prediction_at_request(start_pos)
        if predicted is None:  # at reset, or the first step: no latency, nothing to commit
            return self._planner.make_plan(start_pos, goal_pos)
        if self._mode == "committed":
            committed = predicted[: self._latency_steps]
            p_hat = committed[-1]
            indices, path = self._planner.make_plan(p_hat, goal_pos)
            if path is not None and len(indices) > 1:
                self.num_committed += 1
                self._recorder.expect_switch(p_hat)
                full = np.vstack((np.asarray(start_pos, dtype=float)[np.newaxis], committed, path[0]))
                return self._map.pose_array2index_array(full), full[np.newaxis]
            self.num_fallbacks += 1
        self._recorder.expect_switch(None)
        return self._planner.make_plan(start_pos, goal_pos)


def load_yaml(path: str) -> dict:
    with open(REPO_DIR / path, "r") as f:
        return yaml.safe_load(f)


def run_episode(job, frames: Optional[list] = None, env_out: Optional[list] = None) -> Dict:
    """
    :param frames: if given, the visualized frames of the run are added to it
    :param env_out: if given, the env is added to it
    """
    method, mode, seed = job
    import torch
    from navigation_stack_py.gym_env import NavigationStackEnv
    from navigation_stack_py.rl_modules.dqn.dqn import DQN

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    navigation_config = load_yaml(NAVIGATION_CONFIG)["navigation"]
    env = NavigationStackEnv(
        navigation_config=navigation_config,
        scenario_list=[load_yaml(SCENARIO)],
        rl_config=load_yaml(RL_CONFIG),
        seed=seed,
        visualize_mode="none",
        save_visualized_data=frames is not None,
    )
    recorder = LocalPlannerRecorder(env._local_planner)
    env._local_planner = recorder
    planner = CommittedStartPlanner(
        env._global_planner, recorder, mode, navigation_config["common"]["global_planner_calculation_interval"] - 1
    )
    env._global_planner = planner
    model = DQN.load(str(REPO_DIR / MODEL), seed=seed) if method == "rl_based_replan" else None

    recorder.clear()
    obs = env.reset()
    for _ in range(env.get_max_steps()):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = getattr(env, method)()
        obs, _, done, info = env.step(action)
        if frames is not None:
            frames.extend(info["visualized_data_list"])
        if done:
            break
    if env_out is not None:
        env_out.append(env)
    data = info["data"]
    return dict(
        method=method, mode=mode, seed=seed,
        reach_goal=data["reach_goal"], collision=data["collision"], goal_time=data["goal_time"],
        time=data["time"], replans=data["num_update_global_planner"], travel=data["travel_distance"],
        SGT=data["SGT"], SPL=data["SPL"],
        switch_offsets=recorder.switch_offsets, prediction_errors=recorder.prediction_errors,
        num_committed=planner.num_committed, num_fallbacks=planner.num_fallbacks,
    )


def summarize(results: List[Dict], seeds: List[int]) -> None:
    print(f"\n{len(seeds)} seeds, sixteen pillar with 10 random moving obstacles, Dijkstra x DWA")
    header = (
        f"{'method':22s}{'mode':11s}{'goal':>6s}{'coll':>6s}{'timeout':>8s}{'goal time [s]':>14s}{'SGT':>7s}"
        f"{'SPL':>7s}{'replans':>8s}{'offset at switch [m]':>22s}{'p_hat err [m]':>15s}{'fallback':>9s}"
    )
    print(header)
    for method in METHODS:
        for mode in MODES:
            rows = [r for r in results if r["method"] == method and r["mode"] == mode]
            goals = [r["goal_time"] for r in rows if r["reach_goal"]]
            offsets = np.concatenate([r["switch_offsets"] for r in rows] + [np.empty(0)])
            errors = np.concatenate([r["prediction_errors"] for r in rows] + [np.empty(0)])
            committed = sum(r["num_committed"] for r in rows)
            fallbacks = sum(r["num_fallbacks"] for r in rows)
            timeout = sum(not r["reach_goal"] and not r["collision"] for r in rows)
            offset_text = f"{np.mean(offsets):.2f} / {np.percentile(offsets, 95):.2f}" if len(offsets) else "-"
            error_text = f"{np.mean(errors):.2f} / {np.max(errors):.2f}" if len(errors) else "-"
            fallback_text = f"{fallbacks}/{committed + fallbacks}" if mode == "committed" else "-"
            print(
                f"{method:22s}{mode:11s}{sum(r['reach_goal'] for r in rows):6d}{sum(r['collision'] for r in rows):6d}"
                f"{timeout:8d}{np.mean(goals) if goals else float('nan'):14.2f}{np.mean([r['SGT'] for r in rows]):7.3f}"
                f"{np.mean([r['SPL'] for r in rows]):7.3f}{np.mean([r['replans'] for r in rows]):8.1f}"
                f"{offset_text:>22s}{error_text:>15s}{fallback_text:>9s}"
            )
    print("offset at switch: distance from the robot to the new path when it is first used, mean / 95th percentile")
    print("p_hat err: distance from the robot to the predicted position at that time, mean / max")
    print("goal time: mean over the runs that reached the goal")

    # paired by seed: runs that ended differently
    print("\nSeeds whose outcome differs between the modes (goal / collision / timeout):")
    for method in METHODS:
        diffs = []
        for seed in seeds:
            a = next(r for r in results if r["method"] == method and r["mode"] == "current" and r["seed"] == seed)
            b = next(r for r in results if r["method"] == method and r["mode"] == "committed" and r["seed"] == seed)
            outcome = lambda r: "goal" if r["reach_goal"] else "collision" if r["collision"] else "timeout"
            if outcome(a) != outcome(b):
                diffs.append(f"{seed}: {outcome(a)} -> {outcome(b)}")
        print(f"  {method}: {', '.join(diffs) if diffs else 'none'}")


def render(method: str, seed: int, out_dir: str) -> None:
    """
    Videos of one seed in both modes, rendered with when2replan's visualizer, and side by side
    (left: current, right: committed)
    """
    import pickle
    import sys

    sys.path.insert(0, str(REPO_DIR / "scripts"))  # when2replan's visualizer is a script, not a package
    import pygame_visualizer
    from demos.demo_sparse_tracking import side_by_side

    def subfolder(kind: str, name: str) -> str:
        os.makedirs(os.path.join(out_dir, kind), exist_ok=True)
        return os.path.join(out_dir, kind, name)

    videos = []
    for mode in MODES:
        frames, envs = [], []
        result = run_episode((method, mode, seed), frames=frames, env_out=envs)
        outcome = "goal" if result["reach_goal"] else "collision" if result["collision"] else "timeout"
        print(f"{method} seed {seed} {mode:9s}: {outcome}, time {result['time']:.1f} s, replans {result['replans']}")
        name = f"{method}_seed{seed}_{mode}"
        pkl_path = subfolder("replay", f"{name}_data.pkl")
        map_path = subfolder("replay", f"{name}_map.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(map_path, "wb") as f:
            pickle.dump(envs[0].get_static_map_polygons(), f, protocol=pickle.HIGHEST_PROTOCOL)
        video_path = subfolder("videos", f"{name}.mp4")
        pygame_visualizer.main(pkl_path, map_pkl=map_path, save_path=video_path, headless=True)
        videos.append(video_path)
    save_path = subfolder("comparison", f"{method}_seed{seed}_current_vs_committed.mp4")
    side_by_side(videos[0], videos[1], save_path)
    print(f"Saved {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--save", type=str, default="outputs/when2replan_committed_start/results.yaml")
    parser.add_argument("--video", nargs=2, metavar=("METHOD", "SEED"), default=None,
                        help="render one seed in both modes instead of the comparison")
    args = parser.parse_args()

    if args.video is not None:
        render(args.video[0], int(args.video[1]), os.path.dirname(args.save))
        return

    METHODS[:] = args.methods
    seeds = list(range(args.seeds))
    jobs = [(method, mode, seed) for method in METHODS for mode in MODES for seed in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(run_episode, jobs, chunksize=1)
    summarize(results, seeds)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, "w") as f:
        def plain(value):
            if isinstance(value, list):
                return [round(float(x), 3) for x in value]
            return value.item() if isinstance(value, np.generic) else value

        rows = [{k: plain(v) for k, v in r.items()} for r in results]
        yaml.safe_dump(rows, f, sort_keys=False)
    print(f"\nSaved {args.save}")


if __name__ == "__main__":
    main()
