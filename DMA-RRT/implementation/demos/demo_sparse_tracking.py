"""
    One robot of when2replan (simulator + DWA) tracking the Dijkstra path, dense as planned or as a sparse waypoint graph

    The sparse version reduces each Dijkstra path to a line-of-sight waypoint graph (dma_rrt/sparse_path.py),
    with the edges checked on the same map the global planner uses (static map + scan).
    DWA searches its sub-goal along the points of the reference path, so the waypoints are resampled to the grid
    spacing before tracking. The videos are rendered with when2replan's pygame visualizer,
    and the two runs are put side by side (left: dense, right: sparse).

    Run from DMA-RRT/implementation:
    python3 -m demos.demo_sparse_tracking
    python3 -m demos.demo_sparse_tracking --method time_based_replan --max_edge 3.0
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import sys

import numpy as np
import yaml

from dma_rrt.collision import REPO_DIR
from dma_rrt.sparse_path import line_of_sight, map_edge_checker, path_length, resample

sys.path.insert(0, str(REPO_DIR / "scripts"))  # when2replan's visualizer is a script, not a package
import pygame_visualizer  # noqa: E402
from navigation_stack_py.gym_env import NavigationStackEnv  # noqa: E402

NAVIGATION_CONFIG = "config/run/multi_robot.yaml"
SCENARIO = "config/scenario/sixteen_pillar/multi_robot/robot0.yaml"  # (1, 1) -> (24, 24), no moving obstacles
RL_CONFIG = "config/rl_params/default.yaml"


class SparseGlobalPlanner:
    """
    Wraps when2replan's global planner: plan the dense path, reduce it to waypoints, and resample it for DWA
    """

    def __init__(self, planner, safe_dist: float, max_edge: float) -> None:
        self._planner = planner
        self._safe_dist = safe_dist
        self._max_edge = max_edge
        self._map = None
        self.num_waypoints = []  # waypoints of each plan

    def set_costmap(self, cost_map) -> None:
        self._map = cost_map  # not inflated, the edges are checked with the safe distance
        self._planner.set_costmap(cost_map)

    def get_costmap(self):
        return self._planner.get_costmap()

    def make_plan(self, start_pos: np.ndarray, goal_pos: np.ndarray):
        indices, path = self._planner.make_plan(start_pos, goal_pos)
        if path is None or len(indices) <= 1:
            return indices, path
        dense = path[0]
        keep = line_of_sight(dense, map_edge_checker(self._map, self._safe_dist), self._max_edge)
        self.num_waypoints.append(len(keep))
        tracked = resample(dense[keep], self._map.get_resolution())
        return self._map.pose_array2index_array(tracked), tracked[np.newaxis]


def load_yaml(path: str) -> dict:
    with open(REPO_DIR / path, "r") as f:
        return yaml.safe_load(f)


def run(sparse: bool, method: str, max_edge: float, seed: int, out_dir: str) -> str:
    navigation_config = load_yaml(NAVIGATION_CONFIG)["navigation"]
    env = NavigationStackEnv(
        navigation_config=navigation_config,
        scenario_list=[load_yaml(SCENARIO)],
        rl_config=load_yaml(RL_CONFIG),
        seed=seed,
        visualize_mode="none",
        save_visualized_data=True,
    )
    if sparse:
        common = navigation_config["common"]
        env._global_planner = SparseGlobalPlanner(
            env._global_planner, common["robot_radius"] + common["inflation_radius"], max_edge
        )
    np.random.seed(seed)  # the same Dijkstra ties in both runs

    frames = []
    env.reset()
    for _ in range(env.get_max_steps()):
        _, _, done, info = env.step(getattr(env, method)())
        frames.extend(info["visualized_data_list"])
        if done:
            break

    data = info["data"]
    traj = np.array([frame["robot_state"].pos for frame in frames])
    name = "sparse" if sparse else "dense"
    print(
        f"{name:6s}: reach goal {data['reach_goal']}, collision {data['collision']}, goal time {data['goal_time']} s, "
        f"traveled {path_length(traj):.2f} m, replans {data['num_update_global_planner']}, "
        f"reference path points {len(frames[0]['global_path'])}"
        + (f", waypoints per plan {env._global_planner.num_waypoints}" if sparse else "")
    )

    pkl_path = os.path.join(out_dir, f"{method}_{name}_data.pkl")
    map_path = os.path.join(out_dir, "map.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(map_path, "wb") as f:
        pickle.dump(env.get_static_map_polygons(), f, protocol=pickle.HIGHEST_PROTOCOL)
    video_path = os.path.join(out_dir, f"{method}_{name}.mp4")
    pygame_visualizer.main(pkl_path, map_pkl=map_path, save_path=video_path, headless=True)
    return video_path


def side_by_side(left: str, right: str, save_path: str) -> None:
    """
    Both videos next to each other, the shorter one holds its last frame
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("No ffmpeg in PATH, the side by side video is not made")
        return
    subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error", "-i", left, "-i", right,
            "-filter_complex", "[0:v]tpad=stop=-1:stop_mode=clone[a];[1:v]tpad=stop=-1:stop_mode=clone[b];"
            "[a][b]hstack=inputs=2:shortest=0,scale=1400:-2",
            "-t", str(max(video_duration(left), video_duration(right))),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", save_path,
        ],
        check=True,
    )


def video_duration(path: str) -> float:
    ffprobe = shutil.which("ffprobe")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="no_replan", help="replanning method of when2replan's env")
    parser.add_argument("--max_edge", type=float, default=5.0, help="maximum edge length of the waypoint graph [m]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="outputs/when2replan_sparse_path")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dense_video = run(False, args.method, args.max_edge, args.seed, args.out_dir)
    sparse_video = run(True, args.method, args.max_edge, args.seed, args.out_dir)
    save_path = os.path.join(args.out_dir, f"{args.method}_dense_vs_sparse.mp4")
    side_by_side(dense_video, sparse_video, save_path)
    print(f"Saved {dense_video}, {sparse_video}, {save_path}")


if __name__ == "__main__":
    main()
