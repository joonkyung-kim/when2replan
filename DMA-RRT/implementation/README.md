# DMA-RRT implementation

Steps 0-2 of the plan in [../README.md](../README.md): a basic RRT, collision checking on a map, and CL-RRT tree
expansion. The robots do not move yet: the planners only produce trees and simulated trajectories.

## Setup

Use the environment of this repository (see the [top-level CLAUDE.md](../../CLAUDE.md)), plus an encoder for the videos.
`imageio-ffmpeg` is not listed in `setup.py`.

```bash
conda activate when2replan311
pip install imageio-ffmpeg
cd DMA-RRT/implementation   # all scripts run from this folder
```

## Files

| File | Content |
|---|---|
| `tree.py` | `Node` and `Tree`: nearest and k-nearest search, path extraction |
| `collision.py` | `CollisionChecker`: distance to obstacles through `MapHandler`, edge and trajectory checks |
| `rrt.py` | Basic RRT (steps 0-1) |
| `dynamics.py` | Skid-steer model (eq. 11) and pure pursuit (eq. 12) |
| `cl_rrt.py` | `CLRRT`: our CL-RRT tree expansion (step 2), used by the next steps |
| `cl_rrt_paper.py` | `PaperCLRRT`: Algorithm 1 of the paper as written, kept as a baseline |
| `compare_cl_rrt.py` | Comparison table of the two CL-RRT versions |
| `demo_rrt.py` | Basic RRT: figure and tree growth video |
| `visualize_rrt_steps.py` | Basic RRT: video explaining each step of one iteration |
| `demo_cl_rrt.py` | CL-RRT: figure and video with the simulated robot (`--paper` for the paper's version) |

## Running

```bash
python3 demo_rrt.py --save rrt.png
python3 demo_rrt.py --map maps/white/white/map.yaml --start 2 2 --goal 18 18 --save rrt_empty.png   # no obstacles
python3 visualize_rrt_steps.py --video rrt_steps.mp4

python3 demo_cl_rrt.py --video cl_rrt.mp4
python3 demo_cl_rrt.py --paper --video cl_rrt_paper.mp4

python3 compare_cl_rrt.py   # takes a few minutes
```

Without `--save` or `--video`, the demos open a matplotlib window.

The default scene is the sixteen pillar map, from `(1, 1)` to `(24, 24)`. The robot is treated as a point that must
stay `robot_radius + inflation_radius = 1.1 m` away from obstacles, the same values as `config/run/default.yaml`.

## CL-RRT: how it works

One expansion, for one sample:

1. Sample a point in the free space.
2. Take the closest node in the tree.
3. Draw a reference line from that node's waypoint toward the sample.
4. Simulate the robot tracking the line with pure pursuit, starting from the node's state.
5. If every simulated state is safe, add a node that stores the waypoint and the trajectory.

A node is a **waypoint on the reference**, not a point the robot passes exactly. The trajectory to a node depends
only on the parent's state and the two waypoints, and the simulation is deterministic. So a whole path can be
reproduced from the initial state and its waypoints. This is what later lets robots exchange only waypoints
(Algorithm 5). Both CL-RRT versions were checked to reproduce every path exactly.

The cost of a path is its travel time, as in the paper.

### Parameters

The paper does not give vehicle parameters for this map, so these are our choices, except where noted.

| Parameter | Value | Source |
|---|---|---|
| Max wheel speed | 2.0 m/s | `vel_x_limit` in `config/run/default.yaml` |
| Control interval | 0.1 s | `control_interval_time` in `config/run/default.yaml` |
| Wheel base | 2.0 m | Our choice, the robot's diameter |
| Speed command | 1.0 m/s | Our choice |
| Look-ahead distance L1 | 1.5 m | Our choice |
| Anchor distance l_a | 0 m | Our choice |
| Goal radius | 1.0 m | Our choice |

### Sign of eq. 12

The paper defines `delta_v = v_L - v_R` with `yaw_rate = delta_v / L_w`, and eq. 12 has a minus sign. With yaw and
`eta` positive counterclockwise, using these signs literally steers away from the reference: the robot ended 11.29 m
from a line it started 2 m from. The paper's angle convention is probably different. We define
`delta_v = v_R - v_L` and use `delta_v = v * L_w * sin(eta) / (L1 / 2 + l_a * cos(eta))`. The magnitude is the same as
in the paper, and the robot converges to the line.

`eta` is also clipped to +-90 degrees. `sin(eta)` falls back to 0 for a target behind the robot, which would make it
drive straight away from the target.

## Paper's CL-RRT and ours

`PaperCLRRT` follows Algorithm 1 as written. `CLRRT` is the version we use for the next steps.

| | Paper (`PaperCLRRT`) | Ours (`CLRRT`) |
|---|---|---|
| Nodes tried for a sample | Only the nearest | Up to the 5 nearest, until one succeeds |
| Goal sampling | None, samples are uniform | 5% of the samples are the goal |
| Reference length | Up to the sample | At most 2 m |
| When the simulation hits an obstacle | The part before it becomes a node | The whole trajectory is dropped |

Both sample from the free space and use the same dynamics, controller, parameters, cost, and collision checker.

### Why ours differs

- **Trying more nodes** fixes a problem specific to planning with dynamics. A node can be unable to go anywhere
  because of its heading and turning radius, yet still be the closest node to a large area. With one seed and
  only the nearest node, a single node was chosen 2681 times and failed 2680 times, and the tree stopped at
  8 nodes after 5000 samples. A basic RRT has no heading, so it does not have this problem.
- **The length limit and dropping the whole trajectory** keep the tree simple. Every node is a complete, simulated
  segment, and there is no second simulation for a shortened reference.
- **Goal sampling** is a common RRT technique. It makes planning faster, but it is not what makes ours reliable (see below).

### What the paper leaves open

`PaperCLRRT` needs two decisions that the paper does not specify, so it is our reading of the algorithm and not
the authors' implementation.

1. **The node after a collision.** It is placed on the reference, next to the last safe state. The robot turns
   differently when the reference ends there, so the trajectory is simulated again, and the node is dropped if that
   trajectory is unsafe. This keeps paths reproducible from waypoints.
2. **The time limit.** The loop in Algorithm 1 has no limit, but the robot can circle a sample it cannot reach
   forever. The simulation gives up after `max(20 s, 2 * line length / speed)` and adds no node.

### Results

Output of `compare_cl_rrt.py`: sixteen pillar map, 200 seeds, at most 5000 samples per seed, stopping at the first node in
the goal region. Plan times are from one run on one machine.

| Variant | Reached goal | Trees under 100 nodes | Median travel time | Median plan time | Min clearance | Reproduced from waypoints |
|---|---|---|---|---|---|---|
| paper as written | 147/200 | 57 | 53.1 s | 365 ms | 1.063 m | 147/147 |
| paper + goal sampling 5% | 152/200 | 91 | 50.6 s | 322 ms | 1.063 m | 152/152 |
| paper + k=5 | 193/200 | 48 | 48.4 s | 353 ms | 1.063 m | 193/193 |
| paper + k=5 + goal sampling 5% | 191/200 | 116 | 48.3 s | 256 ms | 1.063 m | 191/191 |
| ours | 200/200 | 0 | 48.4 s | 188 ms | 1.063 m | 200/200 |
| ours with k=1 | 193/200 | 7 | 52.5 s | 346 ms | 1.063 m | 193/193 |
| ours without goal sampling | 200/200 | 0 | 48.6 s | 244 ms | 1.063 m | 200/200 |

Trees under 100 nodes include runs that reached the goal quickly, so this column does not count failures only.

- **k=5 has the largest effect.** It raises the paper's version from 147 to 193, and removing it from ours drops
  200 to 193.
- **Goal sampling barely changes the success rate.** Ours still reaches 200/200 without it, at a higher plan time.
- **Safety and reproducibility are the same in all variants.**

### How to read these results

This is not evidence that the paper's method fails.

- The vehicle parameters and the map are ours, not the paper's.
- This measures whether one planning call reaches the goal. In the paper, a robot executes partial paths and keeps growing
  its tree while moving (Algorithm 2), and it is evaluated on the goals reached in ten minutes. Step 3 implements that loop.
- We did not trace why the paper's version fails as we did for the 8-node tree. Nodes stopped in front of obstacles
  behaving like the blocked node above is a likely explanation, but it is not confirmed.

## Basic RRT results

Sixteen pillar map, `step_size = 0.5 m`, `goal_radius = 0.5 m`, 50 seeds. All reached the goal. The median was 821.5
nodes, 44.98 m (straight line: 32.53 m), and 59 ms. Sampling only in the free space instead of the whole map reduced the median
iterations from 1518 to 984 over 200 seeds. `rrt.py` still samples the whole map, so that rejected samples
show up in the explanation video. `visualize_rrt_steps.py` uses `step_size = 2.0 m` so that the steps are visible, and
therefore finds different paths from `demo_rrt.py`.

## Known limitations

- **Clearance is measured, not guaranteed.** States are checked at the map's 0.1 m resolution. Measured at a finer spacing,
  the minimum distance to an obstacle is 1.063 m, below the 1.1 m limit but above the 1.0 m robot radius.
  The 0.1 m margin absorbs this difference.
- **The map has no walls on its border.** `MapHandler` silently clips positions outside the map to its border, so
  `find_first_unsafe` rejects them explicitly. Only the robot's center is kept in the map, since the scenario's
  start `(1, 1)` is only 1.0 m from the border.
- **A goal candidate in the scenarios is inside a pillar.** `(21, 21)` in `config/scenario/sixteen_pillar/static.yaml`
  and `dynamic.yaml` is in a pillar. The demos report an error for such a start or goal, and use `(24, 24)` by default.
- **`CLRRT` trajectories do not end in a stop yet.** The safe terminal state needed for the paper's guarantee (eq. 7-8) comes in step 3.
- **No timing guarantee.** Plans use a number of samples instead of a time budget, to be reproducible.
