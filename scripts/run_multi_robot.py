"""
Run multiple robots in the same map at the same time.
Each robot has its own navigation stack (global planner, local planner, and replanning method)
and only knows the other robots through its lidar, i.e., they are drawn as moving obstacles in its simulator.
The robots run in separate threads and are synchronized at every simulator step,
so a replanning action (which rolls out several steps inside env.step) is handled the same as in the single robot env.
Only one robot computes at a time (the others wait at the synchronization) and each robot has its own numpy global random state,
because the Dijkstra path extraction of pymap2d breaks ties with the numpy global random generator,
so sharing it between robots makes the paths depend on the thread timing.
"""
from typing import List, Tuple

import os
import pickle
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from navigation_stack_py.gym_env import NavigationStackEnv
from navigation_stack_py.simulator import Simulator
from navigation_stack_py.common import AgentState, MovingObstacle, MotionModel, Shape
from navigation_stack_py.utils import DataLogger

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from navigation_stack_py.rl_modules.dqn.dqn import DQN
from utils import HyperParameters
import pygame_visualizer

from omegaconf import OmegaConf, DictConfig
import hydra

# the visualizer has a distinct color for up to this number of robots
MAX_NUM_ROBOTS = 8


class RobotStateExchanger:
    """
    Exchange robot states between the robot threads at every simulator step
    A robot thread computes only while holding the compute lock, which is released while waiting for the other robots.
    The numpy global random state of the robot is swapped in while holding the lock.
    """

    def __init__(self, num_robots: int, seed: int):
        self._states: List[AgentState] = [None] * num_robots
        self._dones: List[bool] = [False] * num_robots
        self._barrier = threading.Barrier(num_robots)
        self._compute_lock = threading.Lock()

        # independent numpy global random state per robot
        self._random_states = []
        for robot_id in range(num_robots):
            np.random.seed([seed, robot_id])
            self._random_states.append(np.random.get_state())

    def acquire(self, robot_id: int):
        self._compute_lock.acquire()
        np.random.set_state(self._random_states[robot_id])

    def release(self, robot_id: int):
        self._random_states[robot_id] = np.random.get_state()
        self._compute_lock.release()

    def exchange(
        self, robot_id: int, state: AgentState, is_done: bool = False
    ) -> Tuple[List[AgentState], bool]:
        """
        Publish own state and get the states of the other robots at the same simulator step
        return: states of the other robots, whether all robots are done
        """
        self._states[robot_id] = state
        self._dones[robot_id] = is_done

        # let the other robots compute while waiting
        self.release(robot_id)
        try:
            # wait until all robots have moved
            self._barrier.wait()
            other_states = [s for i, s in enumerate(self._states) if i != robot_id]
            all_done = all(self._dones)

            # wait until all robots have read before the next write
            self._barrier.wait()
        finally:
            self.acquire(robot_id)

        return other_states, all_done

    def abort(self):
        # release the other threads if a robot fails
        self._barrier.abort()


class MultiRobotSimulator(Simulator):
    """
    Simulator of one robot, the other robots are added as moving obstacles
    If strict_robot_collision, robots collide when the distance between their centers is less than 2 * robot_radius.
    Otherwise only the collision check on the grid map is used, which detects it only below about 1.6-1.7 m for 1.0 m radius.
    """

    def __init__(
        self,
        robot_id: int,
        exchanger: RobotStateExchanger,
        strict_robot_collision: bool,
    ) -> None:
        super().__init__()
        self._robot_id = robot_id
        self._exchanger = exchanger
        self._strict_robot_collision = strict_robot_collision
        self._other_robots: List[MovingObstacle] = []
        self._is_robot_collision = False
        self._prev_is_robot_collision = False

    def reset(self, params):
        # the other robots are unknown until the first step, start poses should be far from each other
        self._other_robots = []
        self._is_robot_collision = False
        self._prev_is_robot_collision = False
        return super().reset(params)

    def step(self, control_cmd) -> Tuple:
        # the env checks collision on the observation from before this step, so keep its robot collision flag
        self._prev_is_robot_collision = self._is_robot_collision
        return super().step(control_cmd)

    def is_robot_collision(self) -> bool:
        """
        Robot collision flag of the observation the env used in its last step
        """
        return self._prev_is_robot_collision

    def get_dist_to_other_robots(self) -> float:
        if len(self._other_robots) == 0:
            return np.inf
        return min(
            np.linalg.norm(robot.pos - self._robot_state.pos)
            for robot in self._other_robots
        )

    def _update_moving_obstacles(self, obs_list: List) -> List:
        # Simulator.step calls this after the robot has moved and before the obstacle map is generated,
        # so the other robots are updated here to the same time step as this robot
        other_states, _ = self._exchanger.exchange(self._robot_id, self._robot_state)
        self._other_robots = [self._state_to_obstacle(s) for s in other_states]

        return super()._update_moving_obstacles(obs_list)

    def _observe(self, robot_state: AgentState, min_dist_to_moving_obs: float):
        observation = super()._observe(robot_state, min_dist_to_moving_obs)

        # strict robot collision check by the distance between the robot centers
        self._is_robot_collision = (
            self._strict_robot_collision
            and self.get_dist_to_other_robots() < 2.0 * self.ROBOT_RADIUS
        )
        if self._is_robot_collision:
            observation = observation._replace(is_collision=True)

        return observation

    def _generate_obstacle_map(self, static_obstacle_map, moving_obstacles: List):
        return super()._generate_obstacle_map(
            static_obstacle_map, moving_obstacles + self._other_robots
        )

    def _state_to_obstacle(self, state: AgentState) -> MovingObstacle:
        return MovingObstacle(
            pos=state.pos,
            yaw=state.yaw,
            linear_vel=state.linear_vel,
            target_vel=state.linear_vel,
            angular_vel=state.angular_vel,
            shape=Shape.CIRCLE,
            size=np.array([self.ROBOT_RADIUS, 0.0]),
            motion_model=MotionModel.POINT_MASS_MODEL,
        )


class MultiRobotNavigationStackEnv(NavigationStackEnv):
    """
    NavigationStackEnv of one robot among multiple robots
    """

    def __init__(
        self,
        robot_id: int,
        exchanger: RobotStateExchanger,
        strict_robot_collision: bool,
        **kwargs
    ):
        super().__init__(**kwargs)
        # replace the single robot simulator
        self._simulator = MultiRobotSimulator(
            robot_id, exchanger, strict_robot_collision
        )

    def get_robot_state(self) -> AgentState:
        return self._simulator.get_observation().state

    def get_dist_to_other_robots(self) -> float:
        return self._simulator.get_dist_to_other_robots()

    def is_robot_collision(self) -> bool:
        return self._simulator.is_robot_collision()


def load_model(method_name: str, params: HyperParameters, env, seed: int):
    if method_name != "rl_based_replan":
        return None

    model_path = params.saved_model_path
    # check existence of model
    if not os.path.exists(model_path):
        raise Exception("Model does not exist: {}".format(model_path))
    if params.rl_algorithm == "DQN":
        model = DQN.load(model_path, env=env, seed=seed)
    elif params.rl_algorithm == "PPO":
        model = PPO.load(model_path)
    elif params.rl_algorithm == "A2C":
        model = A2C.load(model_path)
    else:
        raise Exception("Unknown model: {}".format(params.rl_algorithm))
    return model


def run_robot(
    robot_id: int,
    method_name: str,
    env,
    model,
    params: HyperParameters,
    csv_logger: DataLogger,
    exchanger: RobotStateExchanger,
    results: List,
):
    """
    Run one robot until done, then keep it at the final pose as an obstacle until all robots are done
    """
    exchanger.acquire(robot_id)
    try:
        traj_data_list = []
        min_dist_to_other_robots = np.inf

        obs = env.reset()
        for i in range(env.get_max_steps()):
            if method_name == "rl_based_replan":
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = getattr(env, method_name)()

            obs, reward, done, info = env.step(action)

            # data log
            data = info["data"]
            data["robot_id"] = robot_id
            data["dist_to_other_robots"] = env.get_dist_to_other_robots()
            data["robot_collision"] = int(env.is_robot_collision())
            min_dist_to_other_robots = min(
                min_dist_to_other_robots, data["dist_to_other_robots"]
            )

            # log q_values
            if method_name == "rl_based_replan":
                if params.rl_algorithm == "DQN":
                    q_value = model.get_q_value(obs)
                    for j in range(len(q_value[0])):
                        data["q_value_{}".format(j)] = q_value[0][j].item()

            csv_logger.log(data)

            # visualize data
            traj_data_list.extend(info["visualized_data_list"])

            if done:
                break

        # stay at the final pose until all robots are done
        final_state = env.get_robot_state()
        all_done = False
        while not all_done:
            _, all_done = exchanger.exchange(robot_id, final_state, is_done=True)

        data["min_dist_to_other_robots"] = min_dist_to_other_robots
        results[robot_id] = (data, traj_data_list, final_state)
    except Exception as e:
        exchanger.abort()
        results[robot_id] = e
    finally:
        exchanger.release(robot_id)


def get_parked_frame(last_frame: dict, final_state: AgentState) -> dict:
    """
    Frame of a robot that is already done: it stays at its final state without scan and paths
    """
    frame = dict(last_frame)
    frame["robot_state"] = final_state
    frame["scan"] = np.empty((0, 2))
    frame["is_replan"] = False
    frame["global_path"] = None
    frame["local_path_list"] = None
    frame["local_path_best_index"] = None
    return frame


def get_robot_frame(robot_id: int, frame: dict) -> dict:
    """
    Get the state, goal, and paths of a robot in a frame to render it in the videos of the other robots
    """
    return {
        "robot_id": robot_id,
        "robot_state": frame["robot_state"],
        "is_replan": frame["is_replan"],
        "is_collision": frame.get("is_collision", False),
        "goal": frame["goal"],
        "global_path": frame["global_path"],
        "local_path_list": frame["local_path_list"],
        "local_path_best_index": frame["local_path_best_index"],
    }


def run(method_name: str, params: HyperParameters, seed: int):
    print("Running {}...".format(method_name))
    print("Seed: {}".format(seed))
    # robots run in threads, so online rendering is not supported; the mp4 is rendered offline from the logged data
    if params.view_animation or params.save_figure or params.visualize_mode != "none":
        raise ValueError(
            "Multi robot run supports only run.visualize_mode=none, view_animation=False, and save_figure=False"
        )
    if params.save_animation and params.movie_type != "mp4":
        raise ValueError("Only movie_type=mp4 is supported, use mp4_to_gif.sh for gif")

    # one robot per scenario
    num_robots = len(params.navigation_scenario_list)
    if num_robots < 2 or num_robots > MAX_NUM_ROBOTS:
        raise ValueError(
            "Multi robot run supports 2 to {} robots, got {}".format(
                MAX_NUM_ROBOTS, num_robots
            )
        )
    set_random_seed(seed)
    exchanger = RobotStateExchanger(num_robots, seed)

    # log files are organized as <seed>/<method>/csv, <seed>/<method>/pkl, and <seed>/<method>/video
    save_dir = os.path.join(params.log_dir, str(seed), method_name)
    csv_dir = os.path.join(save_dir, "csv")
    pkl_dir = os.path.join(save_dir, "pkl")
    video_dir = os.path.join(save_dir, "video")
    for log_dir in [csv_dir, pkl_dir, video_dir]:
        os.makedirs(log_dir, exist_ok=True)

    env_list = []
    model_list = []
    csv_logger_list = []
    for robot_id, scenario in enumerate(params.navigation_scenario_list):
        env = MultiRobotNavigationStackEnv(
            robot_id=robot_id,
            exchanger=exchanger,
            strict_robot_collision=params.strict_robot_collision,
            navigation_config=params.navigation_config,
            scenario_list=[scenario],
            seed=seed,
            visualize_mode=params.visualize_mode,
            rl_config=params.rl_config,
            save_visualized_data=True,
        )
        env.seed(seed)
        env.action_space.seed(seed)
        env = Monitor(env, allow_early_resets=True)
        env_list.append(env)
        model_list.append(load_model(method_name, params, env, seed))

        csv_name = method_name + "_robot" + str(robot_id) + ".csv"
        csv_logger_list.append(DataLogger(os.path.join(csv_dir, csv_name)))

    # run all robots at the same time
    results = [None] * num_robots
    threads = [
        threading.Thread(
            target=run_robot,
            args=(
                robot_id,
                method_name,
                env_list[robot_id],
                model_list[robot_id],
                params,
                csv_logger_list[robot_id],
                exchanger,
                results,
            ),
        )
        for robot_id in range(num_robots)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # raise the original error, not the broken barrier error of the other robots
    errors = [r for r in results if isinstance(r, Exception)]
    for error in errors:
        if not isinstance(error, threading.BrokenBarrierError):
            raise error
    if len(errors) > 0:
        raise errors[0]

    # frame k of every robot is the same simulator step,
    # robots that are done stay at the final pose until the last robot is done, so all videos have the same length
    num_frames = max(len(traj_data_list) for _, traj_data_list, _ in results)
    for _, traj_data_list, final_state in results:
        last_frame = traj_data_list[-1]
        traj_data_list.extend(
            [
                get_parked_frame(last_frame, final_state)
                for _ in range(num_frames - len(traj_data_list))
            ]
        )

    # save map with pickle
    static_map_polygons = getattr(env_list[0], "get_static_map_polygons")()
    map_poligon_path = os.path.join(pkl_dir, "map.pkl")
    f = open(map_poligon_path, "wb")
    pickle.dump(static_map_polygons, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    render_jobs = []
    for robot_id, (data, traj_data_list, _) in enumerate(results):
        csv_logger_list[robot_id].save()

        # add the other robots to every frame to render them as robots with their paths
        for k, visualized_data in enumerate(traj_data_list):
            visualized_data["other_robots"] = [
                get_robot_frame(other_id, other_traj[k])
                for other_id, (_, other_traj, _) in enumerate(results)
                if other_id != robot_id
            ]

        # save traj data with pickle for visualization
        traj_data_name = method_name + "_robot" + str(robot_id) + "_data.pkl"
        traj_data_path = os.path.join(pkl_dir, traj_data_name)
        f = open(traj_data_path, "wb")
        pickle.dump(traj_data_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

        # the movie is rendered offline from the logged data after all robots are saved
        movie_name = (
            method_name + "_robot" + str(robot_id) + "_" + str(seed) + "." + params.movie_type
        )
        render_jobs.append((traj_data_path, os.path.join(video_dir, movie_name)))

        print(
            "Robot {}: reach goal: {}, collision: {} (with robot: {}), goal time [s]: {}, SGT: {:.3f}, replan count: {}, min distance to other robots [m]: {:.2f}".format(
                robot_id,
                data["reach_goal"],
                data["collision"],
                data["robot_collision"],
                data["goal_time"],
                data["SGT"],
                data["num_update_global_planner"],
                data["min_dist_to_other_robots"],
            )
        )

    if params.save_animation:
        render_movies(render_jobs, map_poligon_path)


def render_movie(traj_data_path: str, map_pkl: str, save_path: str):
    pygame_visualizer.main(
        traj_data_path, map_pkl=map_pkl, save_path=save_path, headless=True
    )


def render_movies(render_jobs: List[Tuple[str, str]], map_pkl: str):
    """
    Render the movies of all robots in parallel, one process per movie
    """
    print("Rendering {} movies...".format(len(render_jobs)))
    # spawn, not fork, because the parent process has torch and threads
    with ProcessPoolExecutor(
        max_workers=min(len(render_jobs), os.cpu_count()),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(render_movie, traj_data_path, map_pkl, save_path)
            for traj_data_path, save_path in render_jobs
        ]
        # raise the error of a failed rendering
        for future in futures:
            future.result()


def run_all(params: HyperParameters):
    for seed in params.seeds:
        for method_name in params.method_list:
            run(method_name, params, seed)


@hydra.main(
    config_name="multi_robot.yaml", config_path="../config/run", version_base=None
)
def main(cfg: DictConfig):
    # arg pars
    config = OmegaConf.to_container(cfg, resolve=True)
    params = HyperParameters(config=config, mode="run")
    # multi robot only parameters, not in HyperParameters to keep the config of run.py unchanged
    params.strict_robot_collision = config["run"]["strict_robot_collision"]
    num_robots = config["run"]["num_robots"]
    if num_robots > len(params.navigation_scenario_list):
        raise ValueError(
            "num_robots={} but only {} navigation_scenarios are given".format(
                num_robots, len(params.navigation_scenario_list)
            )
        )
    params.navigation_scenario_list = params.navigation_scenario_list[:num_robots]
    run_all(params)


if __name__ == "__main__":
    main()
