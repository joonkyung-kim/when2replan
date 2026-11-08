"""
    Shared timed plans for when2replan's multi robot run, the part of DMA-RRT that keeps robots apart

    A robot that changes its path broadcasts it with its start time (PlanBoard.post). The others predict where it is
    at any time by moving along that path at SPEED. DWA does not keep a schedule, so the prediction starts from where
    the robot is seen now (lidar, the other robots of the simulator), not from the start time. A message is visible
    from the next simulator step on, so the threads see the same messages whatever order they compute in.

    The receivers use the predictions in both planners:
    - TimedDijkstra (global): times along a new path at SPEED, and if another robot is predicted within CONFLICT_DIST
      around the time this robot passes, its positions around that time become obstacle discs and the path is planned
      again (at most MAX_REPAIRS times)
    - patch_dwa (local): each DWA candidate is compared step by step with the predicted positions of the others over
      its 1 s horizon, and pays DWA_PENALTY for coming closer than DWA_DIST

    The speed of the others is SPEED, or with a SpeedTracker in "measured" mode, the speed each robot sees for each
    other robot over the last second. The tracker also measures the error of the predictions 1 s and 3 s ahead.

    Unlike DMA-RRT this is no guarantee: the speed is a guess, Dijkstra cannot plan to wait, and DWA still decides
    every step on its own.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

SPEED = 1.6  # [m/s] assumed speed along a shared path, about the mean speed of DWA in these runs
MAX_SPEED = 2.0  # [m/s] DWA's limit, a measured speed is clipped to [0, MAX_SPEED]
SPEED_WINDOW = 1.0  # [s] a measured speed is the distance moved over this time
ERROR_HORIZONS = (1.0, 3.0)  # [s] the error of the predictions is measured this far ahead
ROBOT_RADIUS = 1.0
CONFLICT_DIST = 2.3  # [m] centers closer than this at the same time are a conflict of the global paths
TIME_WINDOW = 1.0  # [s] a conflict is checked at t - TIME_WINDOW, t, t + TIME_WINDOW, the speed is only a guess
HORIZON = 15.0  # [s] conflicts further in the future are not checked
DISC_RADIUS = 1.3  # [m] obstacle disc for another robot, Dijkstra adds this robot's radius and inflation
MAX_REPAIRS = 8
DWA_DIST = 2.2  # [m]
DWA_PENALTY = 1000.0
CONFLICT_PENALTY = 50.0  # [m] cost of a path in conflict with a shared plan, used in the bids


def arc_length(path: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))


class PlanBoard:
    """
    The messages of all robots
    """

    def __init__(self, num_robots: int, dt: float) -> None:
        self.dt = dt
        self.posts: Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]] = {r: [] for r in range(num_robots)}
        self.done_step: Dict[int, int] = {}
        self.num_messages = 0
        self.num_floats = 0  # size of the messages: start time + path points

    def post(self, robot_id: int, step: int, path: np.ndarray) -> None:
        """
        :param step: simulator step of the sender, the start time is step * dt
        """
        self.posts[robot_id].append((step, path, arc_length(path)))
        self.num_messages += 1
        self.num_floats += 1 + path.size

    def visible(self, robot_id: int, step: int):
        for post_step, path, cum in reversed(self.posts[robot_id]):
            if post_step < step:
                return path, cum
        return None

    def predict(
        self, robot_id: int, seen_pos: np.ndarray, step: int, times: np.ndarray, speed: Optional[float] = None
    ) -> np.ndarray:
        """
        Where robot_id is at the given times, as known at this step
        :param speed: speed along the path, SPEED if None
        :return: positions size: (len(times), 2)
        """
        speed = SPEED if speed is None else speed
        times = np.atleast_1d(times)
        post = self.visible(robot_id, step)
        if post is None or self.done_step.get(robot_id, np.inf) < step:
            return np.repeat(np.asarray(seen_pos, dtype=float)[np.newaxis], len(times), axis=0)
        path, cum = post
        nearest = int(np.argmin(np.linalg.norm(path - seen_pos, axis=1)))
        s = np.clip(cum[nearest] + speed * (times - step * self.dt), cum[nearest], cum[-1])
        return np.stack((np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])), axis=1)


def seen_robots(env, robot_id: int, num_robots: int) -> Dict[int, np.ndarray]:
    """
    Positions of the other robots at this step, as the robot's lidar sees them
    """
    others = [r for r in range(num_robots) if r != robot_id]
    return {r: np.asarray(o.pos, dtype=float) for r, o in zip(others, env._simulator._other_robots)}


class SpeedTracker:
    """
    What one robot sees of the others at every simulator step: their speeds, and the error of its predictions
    """

    def __init__(self, robot_id: int, num_robots: int, board: PlanBoard, env, clock: List[int], measured: bool):
        self._robot_id = robot_id
        self._num_robots = num_robots
        self._board = board
        self._env = env
        self._clock = clock
        self._measured = measured
        self._history = deque(maxlen=int(round(SPEED_WINDOW / board.dt)) + 1)  # (step, seen positions)
        self._pending = {h: deque() for h in ERROR_HORIZONS}  # (due step, robot id, predicted position)
        self.errors = {h: [] for h in ERROR_HORIZONS}

    def speed(self, other_id: int) -> float:
        if not self._measured or len(self._history) < self._history.maxlen:
            return SPEED
        (step0, seen0), (step1, seen1) = self._history[0], self._history[-1]
        if other_id not in seen0 or other_id not in seen1:
            return SPEED
        moved = np.linalg.norm(seen1[other_id] - seen0[other_id])
        return float(np.clip(moved / ((step1 - step0) * self._board.dt), 0.0, MAX_SPEED))

    def speeds(self, other_ids) -> Dict[int, float]:
        return {r: self.speed(r) for r in other_ids}

    def update(self) -> None:
        """
        After each simulator step: check the predictions due now, and predict ahead from what is seen now
        """
        step = self._clock[0]
        seen = seen_robots(self._env, self._robot_id, self._num_robots)
        self._history.append((step, seen))
        for horizon, pending in self._pending.items():
            while pending and pending[0][0] <= step:
                _, other_id, predicted = pending.popleft()
                if other_id in seen:
                    self.errors[horizon].append(float(np.linalg.norm(seen[other_id] - predicted)))
        for other_id, pos in seen.items():
            if self._board.done_step.get(other_id, np.inf) < step:
                continue  # a robot that is done stays where it is
            for horizon, pending in self._pending.items():
                time = step * self._board.dt + horizon
                predicted = self._board.predict(other_id, pos, step, np.array([time]), self.speed(other_id))[0]
                pending.append((step + int(round(horizon / self._board.dt)), other_id, predicted))


def first_conflict(board: PlanBoard, path: np.ndarray, t0: float, seen: Dict[int, np.ndarray], step: int,
                   speeds: Optional[Dict[int, float]] = None):
    """
    First time a path followed at SPEED from t0 comes within CONFLICT_DIST of another robot's shared plan
    :param speeds: speed of each other robot, SPEED if None
    :return: (time, robot id) or None
    """
    cum = arc_length(path)
    times = t0 + cum / SPEED
    mask = times <= t0 + HORIZON
    points, times = path[mask][::3], times[mask][::3]
    best = None
    for other_id, pos in seen.items():
        for shift in (-TIME_WINDOW, 0.0, TIME_WINDOW):
            speed = None if speeds is None else speeds.get(other_id)
            other = board.predict(other_id, pos, step, np.maximum(times + shift, step * board.dt), speed)
            hits = np.flatnonzero(np.linalg.norm(points - other, axis=1) < CONFLICT_DIST)
            if len(hits) > 0 and (best is None or times[hits[0]] < best[0]):
                best = (float(times[hits[0]]), other_id)
    return best


def add_discs(cost_map, centers: List[np.ndarray], radius: float):
    """
    The map with obstacle discs added, the occupancy index of (x, y) is (x / resolution, y / resolution)
    """
    from navigation_stack_py.utils import MapHandler

    occupancy = cost_map.get_map_as_np("occupancy").copy()
    res = cost_map.get_resolution()
    ii, jj = np.ogrid[: occupancy.shape[0], : occupancy.shape[1]]
    for ij in cost_map.pose_array2index_array(np.asarray(centers)):
        occupancy[(ii - ij[0]) ** 2 + (jj - ij[1]) ** 2 <= (radius / res) ** 2] = 1.0
    new_map = MapHandler()
    new_map.load_map_from_occupancy_map(occupancy, cost_map.get_origin(), res)
    return new_map


class TimedDijkstra:
    """
    when2replan's Dijkstra planner that keeps out of the shared plans of the other robots
    """

    def __init__(self, planner, robot_id: int, num_robots: int, board: PlanBoard, env, clock: List[int],
                 tracker: Optional[SpeedTracker] = None) -> None:
        self._planner = planner
        self.tracker = tracker
        self._robot_id = robot_id
        self._num_robots = num_robots
        self._board = board
        self._env = env
        self._clock = clock
        self._base = None
        self.last_conflict = False
        self.num_repairs = 0

    def __getattr__(self, name):
        return getattr(self._planner, name)

    def set_costmap(self, cost_map) -> None:
        self._base = cost_map
        self._planner.set_costmap(cost_map)

    def get_costmap(self):
        return self._planner.get_costmap()

    def make_plan(self, start_pos: np.ndarray, goal_pos: np.ndarray, t0: Optional[float] = None):
        step = self._clock[0]
        self.last_conflict = False
        if step == 0:  # the first plans at the reset are made at the same time, without messages
            return self._planner.make_plan(start_pos, goal_pos)
        t0 = step * self._board.dt if t0 is None else t0
        seen = seen_robots(self._env, self._robot_id, self._num_robots)
        speeds = self.tracker.speeds(seen) if self.tracker is not None else None
        start, goal = np.asarray(start_pos, dtype=float), np.asarray(goal_pos, dtype=float)
        keep_out = DISC_RADIUS + ROBOT_RADIUS + 0.2  # discs this close to the start or the goal would block them
        discs: List[np.ndarray] = []
        last = None
        for _ in range(MAX_REPAIRS + 1):
            self._planner.set_costmap(self._base if not discs else add_discs(self._base, discs, DISC_RADIUS))
            indices, path = self._planner.make_plan(start, goal)
            if path is None or len(indices) <= 1:
                break  # the discs closed the way, keep the last path
            last = (indices, path)
            conflict = first_conflict(self._board, path[0], t0, seen, step, speeds)
            if conflict is None:
                return last
            t_conflict, other_id = conflict
            times = np.arange(t_conflict - TIME_WINDOW, t_conflict + TIME_WINDOW + 1e-9, 0.25)
            centers = self._board.predict(other_id, seen[other_id], step, np.maximum(times, step * self._board.dt),
                                          None if speeds is None else speeds[other_id])
            centers = [c for c in centers
                       if np.linalg.norm(c - start) > keep_out and np.linalg.norm(c - goal) > keep_out]
            if not centers:
                break
            discs += centers
            self.num_repairs += 1
        if last is None:
            self._planner.set_costmap(self._base)
            return self._planner.make_plan(start, goal)
        self.last_conflict = True
        return last


def patch_dwa(dwa, robot_id: int, num_robots: int, board: PlanBoard, env, clock: List[int],
              tracker: Optional[SpeedTracker] = None) -> None:
    """
    Add the shared plans to the cost of DWA's candidates
    """
    original = dwa._eval_trajectory
    dt = board.dt
    cache = {}

    def predictions():
        step = clock[0]
        if cache.get("step") != step:
            seen = seen_robots(env, robot_id, num_robots)
            times = step * dt + dt * np.arange(1, dwa.PREDICTION_STEP + 1)
            cache["step"] = step
            speeds = tracker.speeds(seen) if tracker is not None else {}
            cache["others"] = (
                np.array([board.predict(r, pos, step, times, speeds.get(r)) for r, pos in seen.items()]) if seen else None
            )
        return cache["others"]

    def eval_trajectory(traj, goal_state, reference_path, sub_goal_index):
        cost = original(traj, goal_state, reference_path, sub_goal_index)
        others = predictions()
        if not np.isfinite(cost) or others is None:
            return cost
        dists = np.linalg.norm(traj[np.newaxis, :, :2] - others, axis=2)
        violation = np.clip(DWA_DIST - dists, 0.0, None).sum()
        return cost + DWA_PENALTY * (1.0 + violation) if violation > 0 else cost

    dwa._eval_trajectory = eval_trajectory


def count_steps(simulator, clock: List[int], tracker: Optional[SpeedTracker] = None) -> None:
    """
    clock[0] = simulator steps since the reset, the env's step count is not updated during its rollouts
    The tracker is updated after every step
    """
    original = simulator.step

    def step(control_cmd):
        clock[0] += 1
        result = original(control_cmd)
        if tracker is not None:
            tracker.update()
        return result

    simulator.step = step
