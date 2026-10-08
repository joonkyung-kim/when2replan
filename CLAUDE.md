# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Code for the ICRA 2024 paper "When to Replan? An Adaptive Replanning Strategy for Autonomous Navigation using Deep Reinforcement Learning". An RL agent (DQN) decides at each step whether to re-run the global planner, inside a pure-Python 2D navigation stack (simulator + global planner + local planner) wrapped as a gym env.

## Setup

The repo now targets **Python 3.11** (`setup.py`, `python_requires>=3.11,<3.12`). The original Python 3.8 dependency set is in `setup_original.py`; the README's install steps describe that older setup. Install notes from `setup.py`:

```bash
conda create -n when2replan311 python=3.11 -y && conda activate when2replan311
pip install "pip<24.1" "setuptools==65.5.0" "wheel<0.40"   # needed to build gym==0.21.0
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Keep numpy on 1.x (gym 0.21.0 breaks on numpy 2). stable-baselines3 is pinned to a specific git commit and the old gym (not gymnasium) API is used throughout (`obs, reward, done, info = env.step(...)`).

## Commands

All entry points are Hydra apps in `scripts/` and are run **from `scripts/`** (they `import utils` / `import pygame_visualizer` as sibling modules). Config files are in `config/{run,train,eval}/default.yaml`; override any key on the command line with Hydra syntax.

```bash
cd scripts

# Run every method in run.method_list on given seeds with a pre-trained model
python3 run.py \
  run.model_dir="../trained_model" run.model_name="DijkstraxDWA/16pillar.zip" \
  run.log_save_dir="run/DijkstraxDWA/16pillar" \
  run.navigation_scenarios=["config/scenario/sixteen_pillar/random.yaml"] \
  run.seeds=[37] navigation.common.global_planner="Dijkstra" navigation.common.local_planner="DWA"

# Re-render a logged run (also: `run_all <dir>`; pass --headless to skip the window)
python3 pygame_visualizer.py main ../log/run/DijkstraxDWA/16pillar/37/rl_based_replan_data.pkl

python3 train.py      # training; tensorboard --logdir ./log/tensorboard, or `mlflow ui` from scripts/
python3 eval.py       # parallel evaluation of eval.eval_method_list; results via `mlflow ui` (scripts/mlruns) and log/
```

`scripts/run_all_for_video.sh` shows the per-planner/map combinations used for the paper videos.

**Tests:** `test/` is not a working test suite. Most files are standalone scripts that reference config paths that no longer exist (`config/navigation/...`, `config/default.yaml`) and run at import time. Only `test/test_map.py` contains pytest functions, and it expects to be run from inside `test/` (it uses `../maps/...`). To check a change, do a short `run.py` invocation instead.

## Path conventions

- Paths in configs (`rl_config`, `navigation_scenarios`, `known_static_map` in scenario YAMLs) are **relative to the repo root**. They are joined to the project dir in `scripts/utils.py:HyperParameters` and `ParameterHandler._get_path`, not to the CWD.
- `log_dir = <repo>/<log_root_dir>/<log_save_dir>` and the model path is `<repo>/<log_root_dir>/<model_dir>/<model_name>`. So `run.model_dir="../trained_model"` resolves to `<repo>/trained_model`. Training writes to `log/model/best_model.zip` by default.
- `log/`, `scripts/outputs/` (Hydra) and `scripts/mlruns/` are gitignored.

## Architecture

**Config flow:** Hydra config → `OmegaConf.to_container` → `HyperParameters(config, mode)` (`scripts/utils.py`, where `mode` ∈ run/train/eval/tune_baselines) loads the scenario YAMLs and RL params → `gym.make(env_id, navigation_config=config["navigation"], scenario_list=[...], rl_config=..., seed=...)`. Adding a config key for a script means also reading it in `HyperParameters`.

**`navigation_stack_py/gym_env/navigation_stack_env.py` (`NavigationStackEnv-v0`)** is the core of the project and ties the other parts together:
- On each `reset()` it samples a scenario from `scenario_list`, re-inits `ParameterHandler` (navigation config + scenario: map, start/goal candidates, moving obstacles), and resamples until the scenario is feasible. It also computes the optimal path length/time used for the SGT/SPL rewards.
- Action space `Discrete(2)`: `0` keep the current reference path, `1` replan. A replan first rolls out the local planner with the *old* path for `global_planner_calculation_interval - 1` steps, which emulates non-blocking global-planner latency. Then it plans on `robot_obs.static_map_with_scan`. If planning fails, the old path is kept.
- The rule-based baselines are **env methods** that return an action: `time_based_replan`, `time_based_replan_patience`, `stuck_based_replan`, `distance_based_replan`, `random_based_replan`, `no_replan`. `run.py` calls them with `getattr(env, method_name)()`; `eval.py` uses `envs.env_method(name)`. The special method name `rl_based_replan` uses the loaded model instead. A new baseline is a new env method plus an entry in `method_list` / `eval_method_list`.
- Observation (scan, previous/reference path samples, goal) and reward weights come from `config/rl_params/default.yaml` (`RLParameters`). Changing the observation dims changes `observation_space` and invalidates the trained models in `trained_model/`.
- `info["data"]` holds per-step metrics for `DataLogger` CSVs. `info["visualized_data_list"]` holds frames that `run.py` pickles (`<method>_data.pkl`, plus `map.pkl`) for `pygame_visualizer.py`.
- `navigation_stack_goal_env.py` (`NavigationStackGoalEnv-v0`) is a near-duplicate goal-conditioned variant for HER. Changes to the main env often need to be mirrored there.

**Planners** are chosen by string in `navigation.common`. Global: `Dijkstra`/`RRT`/`RRT-star`/`PRM` (`global_planner/`, `set_costmap()` + `make_plan(start, goal)`). The Dijkstra path depends on the numpy global random state: pymap2d's `path_from_dijkstra_field` breaks ties between equal-cost cells with `np.random`. So anything that shares that state (threads, extra `np.random` calls) changes the paths. `scripts/run_multi_robot.py` gives each robot its own state for this reason. Local: `DWA`/`MPC` (`local_planner/`). `simulator/simulator.py` steps the robot and moving obstacles and produces scans/costmaps. `utils/costmap_handler.py` (`MapHandler`) wraps `pymap2d`/CMap2D maps and is performance-sensitive.

**RL:** `navigation_stack_py/rl_modules/` holds a vendored and modified copy of SB3's DQN (prioritized replay buffer, HER buffer, `get_q_value()` used to log Q-values in `run.py`). `train.py` only handles `rl_algorithm` = `DQN` and `DQN-PRB` (the default). `DQN-HER` is mentioned in the config, but `train.py` has no branch for it and only imports `HerReplayBuffer`. Load models through this `DQN` class, not `stable_baselines3.DQN`.

**Maps/scenarios:** maps are `.pgm` + `map.yaml` in `maps/` (generated with arena-rosnav's map_creator). A scenario YAML sets `known_static_map`, and with `random_map_mode: True` that points to a directory of map subdirs. It also sets start/goal pose types (`candidates`/`specified`/`random`) and moving obstacles.
