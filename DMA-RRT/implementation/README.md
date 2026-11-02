# DMA-RRT implementation

Steps 0-6 of the plan in [../README.md](../README.md): a basic RRT, collision checking on a map, CL-RRT tree
expansion, the execution loop that moves a robot while its tree keeps growing, collision checking against
time-parameterized obstacles, a team of robots sharing plans with a token, and DMA-RRT, which passes the token by bids.
Emergency stops (Cooperative DMA-RRT) are not implemented yet.

## Setup

Use the environment of this repository (see the [top-level CLAUDE.md](../../CLAUDE.md)), plus an encoder for the videos.
`imageio-ffmpeg` is not listed in `setup.py`.

```bash
conda activate when2replan311
pip install imageio-ffmpeg
cd DMA-RRT/implementation   # all scripts run from this folder
```

## Files

```
implementation/
├── dma_rrt/       the algorithms
├── demos/         figures and videos
├── experiments/   comparison tables
├── maps/          the open map for ten robots, and the maps read from the paper's figures
└── outputs/       generated videos and figures, not in git
```

Scripts are run as modules from this folder, like `python3 -m demos.demo_rrt`, so that they can import `dma_rrt`.

| File | Content |
|---|---|
| `dma_rrt/tree.py` | `Node` and `Tree`: nearest and k-nearest search, path extraction |
| `dma_rrt/collision.py` | `CollisionChecker`: distance to obstacles through `MapHandler`, edge and trajectory checks |
| `dma_rrt/rrt.py` | Basic RRT (steps 0-1) |
| `dma_rrt/dynamics.py` | Skid-steer model (eq. 11) and pure pursuit (eq. 12) |
| `dma_rrt/cl_rrt.py` | `CLRRT`: our CL-RRT tree expansion (step 2), used by the next steps |
| `dma_rrt/cl_rrt_paper.py` | `PaperCLRRT`: Algorithm 1 of the paper as written, kept as a baseline |
| `dma_rrt/execution.py` | `CLRRTExecutor`: execution loop with the lazy check (step 3) |
| `dma_rrt/agent.py` | `Agent`: one robot of a team, with its messages (step 5) |
| `dma_rrt/cooperative.py` | `CooperativeAgent`: emergency stops of Cooperative DMA-RRT (step 7) |
| `dma_rrt/team.py` | `Team`: robots coordinated with emergency stops (step 7), bids (step 6), a round-robin token (step 5), or not at all |
| `dma_rrt/async_team.py` | `AsyncTeam`: each robot runs its own cycles on a virtual clock, with frequent bids (step 9) |
| `dma_rrt/scenarios.py` | Test scenes: robots with fixed plans, robots swapping points on the pillar map, ten robots on the open map, the scenes of Fig. 4 and Fig. 7 of the paper |
| `maps/make_paper_maps.py` | Makes the maps of Fig. 4 and Fig. 7(a) of the paper, read from the figures |
| `demos/demo_rrt.py` | Basic RRT: figure and tree growth video |
| `demos/visualize_rrt_steps.py` | Basic RRT: video explaining each step of one iteration |
| `demos/demo_cl_rrt.py` | CL-RRT: figure and video with the simulated robot (`--paper` for the paper's version) |
| `demos/demo_execution.py` | Execution loop: the robot moves, and an obstacle appears on its plan |
| `demos/demo_team.py` | Team: robots share their plans, with the token holder marked |
| `experiments/compare_cl_rrt.py` | Comparison of the two CL-RRT versions for one planning call |
| `experiments/compare_execution.py` | Comparison of the two CL-RRT versions in the execution loop |
| `experiments/compare_moving_obstacles.py` | Comparison of a robot that knows the moving obstacles with one that ignores them |
| `experiments/compare_team.py` | Comparison of emergency stops, bids, the round-robin token, and no coordination |
| `experiments/compare_async.py` | Comparison of the timing variants of `AsyncTeam` |

## Running

```bash
python3 -m demos.demo_rrt --save outputs/rrt.png
python3 -m demos.demo_rrt --map maps/white/white/map.yaml --start 2 2 --goal 18 18 --save outputs/rrt_empty.png   # no obstacles
python3 -m demos.visualize_rrt_steps --video outputs/rrt_steps.mp4

python3 -m demos.demo_cl_rrt --video outputs/cl_rrt.mp4
python3 -m demos.demo_cl_rrt --paper --video outputs/cl_rrt_paper.mp4

python3 -m demos.demo_execution --video outputs/execution.mp4
python3 -m demos.demo_execution --no_obstacle --save outputs/execution.png
python3 -m demos.demo_execution --moving --no_obstacle --video outputs/moving.mp4

python3 -m demos.demo_team --video outputs/team.mp4
python3 -m demos.demo_team --robots 8 --video outputs/team8.mp4
python3 -m demos.demo_team --open_map --video outputs/team_open.mp4
python3 -m demos.demo_team --coordination cooperative --video outputs/team_cooperative.mp4
python3 -m demos.demo_team --scene hallway_2 --video outputs/team_hallway.mp4
python3 -m demos.demo_team --scene four_agents_4 --coordination cooperative --video outputs/team_four_agents_cooperative.mp4
python3 -m demos.demo_team --scene open_10 --asynchronous --video outputs/team_async_open_10.mp4
python3 -m demos.demo_team --scene four_agents_4 --asynchronous --paper --video outputs/team_paper_four_agents_4.mp4

python3 -m experiments.compare_cl_rrt     # takes a few minutes
python3 -m experiments.compare_execution  # takes longer, because robots that get stuck run until the time limit
python3 -m experiments.compare_moving_obstacles
python3 -m experiments.compare_team  # runs seeds in parallel, see --workers. Takes hours with 20 processes
python3 -m experiments.compare_team --scenes pillar_8 --coordinations cooperative --stop_interval 16
python3 -m experiments.compare_async  # takes hours with 20 processes
python3 -m experiments.compare_async --scenes hallway_2 four_agents_4 open_10 --coordinations round_robin merit cooperative cooperative_paper --variants timed_fast_better --paper_tree --clip_each_wheel
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
  its tree while moving (Algorithm 2), and it is evaluated on the goals reached in ten minutes. See the execution
  loop results below.
- We did not trace why one planning call of the paper's version fails as we did for the 8-node tree. Nodes stopped in
  front of obstacles behaving like the blocked node above is a likely explanation. The execution loop results support
  it, but they do not confirm it for this table.

## Execution loop (step 3)

`CLRRTExecutor` follows Algorithm 2. In each 1 s cycle:

1. **Commit.** Extend the committed path until it covers this cycle. The last node on it becomes the root of the tree,
   and only its descendants stay. The robot cannot avoid this part anymore, so every plan starts from the root.
2. **Grow.** Run 30 samples. The paper uses a time budget, but a sample count is reproducible.
3. **Select.** A path reaching the goal is always preferred, as in Sec. 8.2. Otherwise use the lowest estimate of
   `cost + 2 * straight line time to the goal`.
4. **Lazy check (Algorithm 3).** Simulate the selected path again against the current obstacles. If a node is unsafe, remove
   it with its children and select again.
5. **Move.** Follow the committed path for the cycle.

### Safe stops

- **Terminal state (eq. 7-8).** The robot stops at the end of its plan. The model has no acceleration limit, and
  all nodes are safe, so every node is a safe place to stop. An acceleration limit would need a braking distance here.
- **Safety action.** If the committed path becomes unsafe, the robot stops where it is and starts a new tree.
- **Checks when committing.** A node is checked again just before it is committed, and the robot stops before it if it became
  unsafe. Without this check, the robot committed to a segment an obstacle had just blocked and drove up to 1 s of it
  before the safety action reacted. That brought it to 0.819 m from an obstacle, inside its 1.0 m radius, in 1 of 200 runs.

### Turning in place (not in the paper)

A stopped robot can have almost no feasible path, for example when it faces a pillar or a wall. Its tree then stops growing,
and the robot stays there. When the robot is stopped and its tree got fewer than 2 new nodes in a cycle (`min_tree_growth`), it turns in place at
1 rad/s and starts a new tree. This is always safe for a circular robot, which occupies the same space while turning.
The first version only turned when the tree had nothing but the root, which misses a tree with a few useless nodes (step 8).
Step 10 shows that a small `L1` could replace this rule, but then the trajectories are no longer smooth as in the paper, so we keep the rule.

### The new obstacle is our test, not part of the paper

The paper assumes a known environment with only static obstacles (Sec. 5.1), and its experiments use an open map and a map
with fixed obstacles (Sec. 7). Moving objects are only mentioned as a possible extension and future work (Sec. 6.4, 9).
The sixteen pillar map is from this repository, not from the paper.

The constraints that change in the paper are mainly the paths of the other robots, and the lazy check exists for such changes
(Sec. 3.2). With one robot on a static map, the lazy check never removes anything. `CollisionChecker.add_circle` therefore
adds an obstacle during the run, standing in for another robot changing its path until step 4 adds real ones.

The "new obstacle" results below test the safety behavior of this code, and they are not a reproduction of a result from the
paper. Use `--no_obstacle` with `demo_execution.py` for the paper's conditions.

### Differences from the paper in this step

| | Paper | Ours |
|---|---|---|
| Planning time per cycle | Time budget | 30 samples, to be reproducible |
| Plan selection without a path to the goal | Cost estimates, not detailed | `cost + 2 * straight line time to the goal` |
| Committed part of the path | Not detailed | Up to the first node covering the cycle |
| Stopped with no feasible path | Not addressed | Turn in place |
| Obstacles during a run | Not used | Circle added for testing |

### Results

Output of `compare_execution.py`: sixteen pillar map, 200 seeds, 300 s limit. "New obstacle" adds a circle of radius 1.5 m
at 8 s, 6 s ahead on the current plan. Times include all waiting, and the first cycle, in which the robot has no plan yet.

| Variant | Reached goal | Median time | 95th percentile | Runs that got stuck | Lazy check removals | Safety stops | Min clearance |
|---|---|---|---|---|---|---|---|
| paper's tree expansion, no turning | 4/200 | 56.0 s | 60.5 s | 196 | 0 | 0 | 1.100 m |
| paper's tree expansion, turning | 200/200 | 74.0 s | 136.1 s | 196 | 0 | 0 | 1.100 m |
| ours, no turning | 194/200 | 49.0 s | 59.0 s | 16 | 0 | 0 | 1.100 m |
| ours, turning | 200/200 | 49.0 s | 59.0 s | 16 | 0 | 0 | 1.100 m |
| ours, turning, new obstacle | 200/200 | 52.0 s | 79.0 s | 72 | 354 | 2 | 1.100 m |
| ours, no turning, new obstacle | 133/200 | 51.0 s | 63.4 s | 72 | 349 | 2 | 1.100 m |

Min clearance is measured at the executed states. Between the states it can be slightly lower, see the known limitations.
"Runs that got stuck" counts runs with at least one cycle in which the robot was stopped with only the root in its tree.

- **Stopped robots are the main failure.** With the paper's expansion and no turning, 40 of 40 inspected runs got stuck, with a
  median of 3 s and 0.6 m driven. In 39 of the 40, the space 0.3 m straight ahead was unsafe.
- **Why the paper's expansion suffers more here.** It keeps trajectories up to the point just before an obstacle, so a
  plan often ends facing that obstacle. With ours, every node is the end of a complete 2 m segment.
- **This does not show that the paper's method fails.** Stopping at the end of a plan, the selection rule, and the
  parameters are ours. The paper does not report this deadlock for its vehicles, and its planner may avoid it in
  ways the paper does not describe.
- **Safety held in all variants** after the fix above, including when an obstacle appears on the plan.

## Time-parameterized obstacles (step 4)

As in Sec. 7.1 of the paper, another robot's trajectory is a moving obstacle: a position for each time. The question
changes from "is this position free?" to "is this position free when the robot is there?". `MovingObstacle` keeps the
obstacle at its first state before its start, and at its last state after its end, which is its safe terminal state.

The distance limit is `robot_radius + inflation_radius` from the obstacle's edge. For another robot of radius 1.0 m,
the centers stay at least 2.1 m apart.

### Time along a path

- **Nodes do not store times.** The time at a node is `root time + node cost - root cost`. A robot waiting at the root
  moves the root time forward, which shifts every path after it. A path that was safe can become unsafe this way,
  and the lazy check catches it.
- **Time is checked in all four places:** tree expansion, the lazy check, the committed path, and the check when committing.

### Safe to stop (eq. 7)

The robot stops at the end of its plan and stays there, so that position has to stay safe for all future time. A position
that is free when the robot arrives can be on another robot's way later. `is_stop_safe` checks this, and `best_node`
only ends a plan at such a node. A node that is not safe to stop at can still be passed through. If no node qualifies,
the robot waits at the root.

### Results

Output of `compare_moving_obstacles.py`: sixteen pillar map, 200 seeds, 300 s limit. Three robots drive through the corridors at
1 m/s from time 0, crossing the way to the goal, and stay at the ends of their lines. The distance is checked at every control
step, including while the robot is not moving.

| Robot | Reached goal | Median time | 95th percentile | Runs overlapping an obstacle | Min distance to an obstacle's edge | Lazy check removals | Safety stops |
|---|---|---|---|---|---|---|---|
| knows the obstacles | 200/200 | 51.0 s | 69.0 s | 0 | 1.128 m | 0 | 0 |
| ignores the obstacles | 200/200 | 49.0 s | 59.0 s | 166 | -0.993 m | 0 | 0 |

The robot that ignores them overlaps a moving robot in 166 of 200 runs. A negative distance means its center went inside the other
robot. Avoiding them costs 2 s at the median.

### Limits of this test

- **The moving obstacles do not react, and their whole trajectories are known.** This is the situation a robot without the token
  has in the paper, but here the trajectories never change. The lazy check removed nothing in these 200 runs.
- **A stopped robot cannot avoid an obstacle that drives into it.** It did not happen in these runs, since plans only end where
  it is safe to stop. In the paper, the other robots also plan around this robot's terminal state, which comes with step 5.
- **The straight line estimate has local minima.** With seed 0 the robot took 123 s, against the 95th percentile of 69 s. A moving robot
  parks at `(24.5, 18.7)` and blocks the east corridor to the goal, but the estimate keeps drawing the robot toward that dead end
  until it goes around. This costs time, not safety. An estimate that knows the map would help, but would not know about moving obstacles.

## Team with a round-robin token (step 5)

Each robot has its own `Agent`: a tree, a collision checker, and what it knows about the others from messages. `Team`
runs the cycles and delivers the messages. There is no shared state, and the network is lossless with no delay, as assumed in Sec. 5.1.

In each 1 s cycle:

1. **The token holder** commits, grows its tree, selects a plan against the plans the others shared, and moves.
2. **It broadcasts** the new plan and the next token holder.
3. **The others** rebuild its trajectory by simulating the waypoints, keep their own plans, and grow their trees.
4. **The token goes to the next robot** in a fixed order. This is the baseline of Sec. 7.2.1, and step 6 replaces the fixed order with bids.

A robot's plan is safe against the others' current plans, and it can only be invalidated by a change that the next token holder
checks against it. This is the induction of Theorem 1. Keeping it needs the following.

- **A message starts at the root.** The other robots keep the old trajectory until the root's time, because the committed part cannot change.
- **A plan starts when the robot really moves.** New nodes are only committed in the next cycle, so a robot stopped at the root leaves then.
  Announcing the current time instead made the other robots believe it moved up to 1 s early.
- **A plan that cannot stop at the root keeps the current plan.** The others planned on the robot passing the root, so stopping there
  could conflict with them.
- **The robot does not stop at a goal mid-plan.** It keeps the announced timing, and just moves on to its next goal.

The paper's message has waypoints and the winner. Ours also has the state and the time at the start of the waypoints, because a robot
can turn in place, which changes the state the waypoints start from.

Results are in the next section, together with the bids.

### Limits of this team

- **The minimum distance of 2.100 m is at the limit.** It is measured at the control steps. Between the steps, it can be slightly lower.
- **Lazy check removals with the token are not a violation.** They remove paths in a robot's own tree, which were grown before another robot's plan changed.
- **Two robots cannot pass each other in a corridor here.** A corridor leaves 1.9 m for a robot's center, and two centers need 2.1 m. Only intersections
  and other corridors let them pass, so narrow passages will matter for the emergency stops of step 7.
- **There is no real network or separate computers.** The paper runs each robot on its own computer over UDP.

## DMA-RRT: bids (step 6)

Instead of a fixed order, the token goes to the robot with the best bid (Sec. 4.1, Algorithms 4-5). A bid is the potential path
improvement of eq. 5: the cost of the current plan minus the cost of the best path in the robot's tree. The tree already
exists, so a bid needs no extra planning. Bids only choose the next token holder, so the safety argument of step 5 does not change.

### Decisions for the bids

Four problems came up while implementing this. The first three are needed for eq. 5 to work as the paper describes it, and the fourth is an addition.

1. **The token holder bids too.** Algorithm 4 only has the others bid. With one bid per cycle, the others kept the high bid that won
   a robot the token, and could pass it straight back. The paper sends bids much more often than plans (Sec. 5.3), so a stale bid does not last there.
2. **A path not reaching the goal has a large penalty in the cost.** The straight line estimate is optimistic, so a winding path to the goal looked worse than
   stopping: 674 of 698 bids of stopped robots with a goal path in the tree were 0 or less, and one robot never got the token in 200 s. The paper says robots that
   find paths to their goals bid higher (Sec. 4.1), and the penalty makes that true. It cancels out between two paths not reaching the goal.
3. **A bid is not negative.** The current plan is in the tree, so the best path in the tree cannot be worse than it. But `best_node` leaves out the root, so
   that a robot does not choose to stay, and the root is the current plan of a stopped robot. Comparing with a worse path gave stopped robots negative bids, and they lost to every robot
   bidding 0. The paper does not state that the bid is not negative, this is our reading of eq. 5. `merit_negative` keeps what we first implemented.
4. **`merit_wait` adds the time a robot has been stopped to its bid.** This is not in the paper, and `merit` is kept without it. It was first added against the starved robots of the previous point (step 8),
   and still adds a little on `pillar_8` and `open_10`.

### Scenes

- **`pillar_N`:** N robots on the sixteen pillar map, in pairs swapping opposite points on the border, back and forth.
- **`open_10`:** Ten robots on an empty 40 m x 40 m map (`maps/open_40`), following the setup of Sec. 7.2. Five start at the bottom and five at the top, and all visit the same
  ten goals on a circle in order, looping. The positions and the map size are our choice.
- **`hallway_2`:** Two robots on the map of Fig. 4, starting as in the figure, going back and forth through a 4 m wide hallway. Two robots cannot pass each other in it.
- **`four_agents_4`:** Four robots on the map of Fig. 7(a), starting as in the figure, all alternating between the same two goals (Sec. 7.3).

The paper gives no dimensions for these two maps. A robot circle is about 6 pixels in its figures, so we use 6 pixels = 1 m for our robots of radius 1 m. The obstacles are read from the figures by eye.

### Results

Output of `compare_team.py`: 24 seeds, 600 s per run, the paper's measure of goals in ten minutes. The interval is a 95% interval of the mean over runs. With two robots, the bids have
only one robot to choose, so all of them are identical to `round_robin`. `merit_negative` was not run on `pillar_6`.

| Scene | Coordination | Goals per robot in 600 s | 95% interval | Std. dev. between robots | Min distance between centers | Runs under 2.1 m | Cycles with conflicting plans | Wrong beliefs | Safety stops | Lazy check removals | Token share, most / least |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pillar_2 | merit | 10.73 | [10.53, 10.92] | 0.57 | 2.111 m | 0/24 | 0/14400 | 0/28800 | 0 | 51 | 50% / 50% |
| pillar_2 | round_robin | 10.73 | [10.53, 10.92] | 0.57 | 2.111 m | 0/24 | 0/14400 | 0/28800 | 0 | 51 | 50% / 50% |
| pillar_2 | none | 10.90 | [10.78, 11.01] | 0.47 | 2.110 m | 0/24 | 41/14400 | 0/28800 | 0 | 85 | - |
| pillar_4 | merit | 10.69 | [10.59, 10.79] | 0.53 | 2.100 m | 0/24 | 0/14400 | 0/172800 | 0 | 307 | 26% / 24% |
| pillar_4 | merit_negative | 10.71 | [10.60, 10.81] | 0.50 | 2.100 m | 0/24 | 0/14400 | 0/172800 | 0 | 305 | 26% / 24% |
| pillar_4 | merit_wait | 10.72 | [10.63, 10.80] | 0.49 | 2.100 m | 0/24 | 0/14400 | 0/172800 | 0 | 319 | 26% / 24% |
| pillar_4 | round_robin | 10.59 | [10.47, 10.72] | 0.53 | 2.104 m | 0/24 | 0/14400 | 0/172800 | 0 | 336 | 25% / 25% |
| pillar_4 | none | 10.81 | [10.72, 10.90] | 0.53 | 2.105 m | 0/24 | 168/14400 | 6/172800 | 0 | 450 | - |
| pillar_6 | merit | 11.24 | [11.12, 11.35] | 1.39 | 2.100 m | 0/24 | 0/14400 | 0/432000 | 0 | 1036 | 18% / 15% |
| pillar_6 | merit_wait | 11.27 | [11.14, 11.40] | 1.47 | 2.101 m | 0/24 | 0/14400 | 0/432000 | 0 | 969 | 18% / 15% |
| pillar_6 | round_robin | 10.60 | [10.49, 10.70] | 1.11 | 2.100 m | 0/24 | 0/14400 | 0/432000 | 0 | 1267 | 17% / 17% |
| pillar_6 | none | 11.70 | [11.61, 11.80] | 1.51 | 2.101 m | 0/24 | 603/14400 | 185/432000 | 1 | 1501 | - |
| pillar_8 | merit | 10.64 | [10.53, 10.74] | 1.32 | 2.100 m | 0/24 | 0/14400 | 0/806400 | 0 | 2279 | 14% / 11% |
| pillar_8 | merit_negative | 10.52 | [10.34, 10.69] | 1.59 | 2.100 m | 0/24 | 0/14400 | 0/806400 | 0 | 2195 | 14% / 11% |
| pillar_8 | merit_wait | 11.02 | [10.86, 11.18] | 1.57 | 2.100 m | 0/24 | 0/14400 | 0/806400 | 0 | 2007 | 14% / 11% |
| pillar_8 | round_robin | 9.77 | [9.63, 9.90] | 1.35 | 2.101 m | 0/24 | 0/14400 | 0/806400 | 0 | 2806 | 12% / 12% |
| pillar_8 | none | 11.64 | [11.53, 11.76] | 1.65 | 2.100 m | 0/24 | 1100/14400 | 917/806400 | 0 | 2840 | - |
| open_10 | merit | 11.08 | [10.91, 11.24] | 1.06 | 2.100 m | 0/24 | 0/14400 | 0/1296000 | 0 | 2120 | 12% / 8% |
| open_10 | merit_negative | 10.88 | [10.75, 11.01] | 1.03 | 2.100 m | 0/24 | 0/14400 | 0/1296000 | 0 | 2026 | 12% / 8% |
| open_10 | merit_wait | 11.36 | [11.23, 11.49] | 0.88 | 2.100 m | 0/24 | 0/14400 | 0/1296000 | 0 | 1997 | 13% / 8% |
| open_10 | round_robin | 9.83 | [9.69, 9.98] | 1.07 | 2.100 m | 0/24 | 0/14400 | 0/1296000 | 0 | 2640 | 10% / 10% |
| open_10 | none | 12.11 | [12.00, 12.23] | 0.87 | 2.100 m | 0/24 | 997/14400 | 333/1296000 | 0 | 1985 | - |
| hallway_2 | merit | 11.65 | [11.47, 11.82] | 0.59 | 2.101 m | 0/24 | 0/14400 | 0/28800 | 0 | 158 | 50% / 50% |
| hallway_2 | round_robin | 11.65 | [11.47, 11.82] | 0.59 | 2.101 m | 0/24 | 0/14400 | 0/28800 | 0 | 158 | 50% / 50% |
| hallway_2 | none | 11.77 | [11.61, 11.93] | 0.59 | 2.100 m | 0/24 | 164/14400 | 12/28800 | 0 | 264 | - |
| four_agents_4 | merit | 5.19 | [4.96, 5.41] | 1.42 | 2.100 m | 0/24 | 0/14400 | 0/172800 | 0 | 376 | 27% / 24% |
| four_agents_4 | merit_negative | 4.60 | [4.42, 4.79] | 1.79 | 2.100 m | 0/24 | 0/14400 | 0/172800 | 0 | 369 | 32% / 16% |
| four_agents_4 | merit_wait | 5.26 | [5.05, 5.47] | 1.39 | 2.100 m | 0/24 | 0/14400 | 0/172800 | 0 | 365 | 28% / 23% |
| four_agents_4 | round_robin | 5.10 | [4.92, 5.29] | 1.36 | 2.100 m | 0/24 | 0/14400 | 0/172800 | 0 | 382 | 25% / 25% |
| four_agents_4 | none | 5.42 | [5.20, 5.63] | 1.38 | 2.101 m | 0/24 | 1481/14400 | 51/172800 | 0 | 559 | - |

**Definitions.** "Conflicting plans" counts cycles in which two robots' current plans come closer than 2.1 m at the same time, the condition of Theorem 1, checked directly in
every cycle. "Wrong beliefs" compares where a robot believed another one would be during a cycle with where that robot really was. "Min distance" is measured at the control steps.

### How to read the results

- **The safety guarantee held with any token.** All token variants have 0 conflicting plans, 0 wrong beliefs, and no run under 2.1 m in all scenes.
- **Without a token, plans conflict and robots can really collide.** Conflicts grow with the number of robots, the failure of Fig. 2. Before step 8, in one `pillar_6` run, two centers came within
  1.513 m, an overlap of the two 1.0 m radius robots. In the table above no run got under 2.1 m, but plans conflict in up to 1481 of 14400 cycles and wrong beliefs remain.
- **No coordination reaches the most goals, by being unsafe.** It changes all plans in every cycle. With a token, each robot waits for its turn. Its goal count is not a reason to use it.
- **Bids beat the fixed order when there are many robots.** On `open_10`, `merit` reaches 13% more goals than `round_robin`, 11.08 against 9.83, with intervals that do not overlap. This is the same
  direction as the paper's 12.5 to 15.1 (21%), and smaller. It also gains 6% on `pillar_6` and 9% on `pillar_8`.
- **With four robots, bids make no difference.** `pillar_4` and `four_agents_4` overlap with `round_robin`. The paper also found no gain with four robots with obstacles (11.9 and 11.9).
- **`merit_wait` adds a little.** It is above `merit` on `pillar_8` (11.02 against 10.64) and `open_10` (11.36 against 11.08), with intervals that do not overlap, and about equal elsewhere.

### Limits of this comparison

- **One bid per cycle.** See step 9 for bids sent more often, and only when better.
- **24 seeds on our own scenes.** The paper uses 12 ten-minute trials. `pillar_N` and `open_10` are not its scenes, and the two others are read from its figures.
- **Fixed samples per cycle.** All robots get 30 samples per cycle, not a time budget. See step 8 for other numbers on the paper's two scenes.

## Cooperative DMA-RRT: emergency stops (step 7)

Each robot marks nodes on its plan where it could stop safely, about every 4 s (Sec. 7.3.3), and shares them with its
waypoints. When the token holder's best path is blocked by a robot, it compares cutting its own path short with asking that
robot to stop at one of these nodes, and picks the lowest combined cost (Algorithm 7). A robot asked to stop gets
the token next, and cannot ask for a stop itself in that turn, which avoids cycles.

A robot can be asked to stop at any of these nodes, so the others must not pass through one after the robot could be there,
and must not stop there at all. Stop nodes therefore block space, and this turned out to be the main cost of the method.

### Variants

| Name | Tree | Plan | Bid |
|---|---|---|---|
| `cooperative_paper` | Ignores the other robots (line 4 of Algorithm 6) | Best path, cut short | Eq. 5 with that tree (line 23) |
| `cooperative` | Avoids the other robots when a node is added | Best path, cut short, or any free branch | Eq. 5 with that tree |
| `cooperative_wait` | Same as `cooperative` | Same | Adds the waiting time of `merit_wait` |
| `cooperative_negative` | Same as `cooperative` | Same | Can be negative, like `merit_negative` |

### Differences from the paper

In all variants:

- **Keeping the current plan is an option.** Otherwise the best path cut short can replace a better plan the robot already has.
- **A bid also carries the time of the sender's root.** A robot uses only stop nodes the other has not passed. The
  paper only says they have to be far enough ahead (Sec. 6.3), and this is the exact limit of our commit.
- **Passed stop nodes stop blocking.** The paper does not say this, but a robot cannot stop where it has already been.
- **The estop message is broadcast.** The paper sends it to the stopped robot, and the others learn from its next plan.
- **Other robots are checked while finding the last safe node.** The paper checks one robot and prunes for the others afterwards.

Only in `cooperative` and `cooperative_wait`:

- **The tree avoids the other robots, and blocked nodes are removed.** With a tree that ignores them, new samples attach to
  nearby blocked nodes, so branches around another robot barely grow.
- **Any free branch can be the plan.** The best path can stay blocked while another branch is free.

The paper's text does not mention these two problems. Its results may depend on details it does not give, so
they are limits of our reproduction, not a claim about the paper.

### Why `cooperative_paper` reaches so few goals

With a tree ignoring the others, a blocked robot keeps bidding about 1000 (the no-goal penalty of `cost_estimate`), wins the token, and
cannot use it. On `pillar_4` in 200 s runs before step 8, 93% of its token turns (3733 of 4000) left the plan unchanged, and two robots held most tokens. It still holds: the token shares of `cooperative_paper` in the new runs are up to 50% / 0%.

### Results

`compare_team`, 24 seeds, 600 s, goals per robot with the 95% interval.

| Scene | `merit` | `merit_wait` | `cooperative_paper` | `cooperative` | `cooperative_wait` |
|---|---|---|---|---|---|
| `pillar_2` | 10.73 [10.53, 10.92] | 10.73 [10.53, 10.92] | 5.04 [3.62, 6.46] | 10.88 [10.77, 10.98] | 10.88 [10.77, 10.98] |
| `pillar_4` | 10.69 [10.59, 10.79] | 10.72 [10.63, 10.80] | 2.42 [1.29, 3.55] | 10.24 [10.10, 10.38] | 10.27 [10.13, 10.42] |
| `pillar_6` | 11.24 [11.12, 11.35] | 11.27 [11.14, 11.40] | 0.11 [0.04, 0.18] | 9.56 [9.41, 9.71] | 9.87 [9.74, 10.00] |
| `pillar_8` | 10.64 [10.53, 10.74] | 11.02 [10.86, 11.18] | 0.02 [-0.01, 0.05] | 7.67 [7.52, 7.82] | 7.86 [7.68, 8.05] |
| `open_10` | 11.08 [10.91, 11.24] | 11.36 [11.23, 11.49] | 0.72 [0.60, 0.84] | 10.34 [10.21, 10.47] | 10.70 [10.56, 10.84] |
| `hallway_2` | 11.65 [11.47, 11.82] | 11.65 [11.47, 11.82] | 1.83 [1.27, 2.40] | 9.23 [8.09, 10.37] | 9.23 [8.09, 10.37] |
| `four_agents_4` | 5.19 [4.96, 5.41] | 5.26 [5.05, 5.47] | 0.50 [0.27, 0.73] | 4.46 [4.25, 4.66] | 4.47 [4.27, 4.66] |

With stop nodes every 16 s instead of 4 s:

| Scene | `cooperative` | `cooperative_wait` |
|---|---|---|
| `pillar_4` | 10.59 [10.45, 10.74] | 10.65 [10.53, 10.76] |
| `pillar_8` | 9.03 [8.82, 9.23] | 9.54 [9.28, 9.80] |
| `open_10` | 10.59 [10.42, 10.76] | 11.03 [10.83, 11.23] |
| `hallway_2` | 9.75 [8.81, 10.69] | 9.75 [8.81, 10.69] |
| `four_agents_4` | 4.66 [4.43, 4.88] | 4.00 [3.47, 4.53] |

Safety held in all of these runs: no conflicting plans, no wrong beliefs, no safety stops, no run under 2.1 m. The robot asked to stop is left out of the
belief check in that cycle, because the check takes beliefs just before the message; all such mismatches we inspected were of this kind. In a separate check at 200 s
on `pillar_4`, `pillar_8` and `open_10`, 22 seeds each, all stop nodes a robot could still use were safe against the others' plans (about 1.85 million checks, 0 unsafe).

### Where the loss comes from

300 s runs on `pillar_8`, 22 seeds, goals per robot with half of the 95% interval, in the same tree as `cooperative`. Measured before step 8, with the first bids and turning:

| Configuration | Goals |
|---|---|
| `merit` | 4.48 ± 0.20 |
| No stop nodes at all | 4.26 ± 0.25 |
| Stop nodes block space, but no robot can ask for a stop | 2.78 ± 0.17 |
| Full method, stop nodes every 4 s | 3.11 ± 0.17 |
| Full method, stop nodes every 16 s | 3.51 ± 0.24 |

### How to read the results

- **The paper's gain was not reproduced.** The paper reports 11.9 to 13.0 goals with four robots and obstacles. On `four_agents_4`, the closest scene we have, emergency stops lose
  14%, from 5.19 to 4.46. They are below `merit` in all scenes but `pillar_2`, where the two overlap. The fixes of step 8 did not change this.
- **Without stop nodes the method falls back to DMA-RRT**, as Sec. 6.4 says it should: 4.26 against 4.48, with overlapping intervals. This is our check that the rest of the implementation is sound.
- **Asking for a stop helps, and blocking space costs more.** On `pillar_8`, asking for stops gains 2.78 to 3.11, and the blocking loses 4.26 to 2.78.
- **Fewer stop nodes help where space is tight.** 16 s is clearly better on `pillar_4` and `pillar_8`, and overlaps in the other scenes. The exception is `cooperative_wait` on `four_agents_4`, which gets worse (4.00, with 1524 emergency stops
  against 917). We did not trace why.

### Limits of this comparison

- **The cost comparison includes the no-goal penalty.** How often an emergency stop trades one robot's goal for another's was not measured.
- **Fixed samples per cycle.** The paper plans within a time budget. See step 8 for other numbers.

## Checking the results against the paper (step 8)

The results of steps 6 and 7 looked worse than the paper's, so we looked for what in our code caused it. A video showed two robots standing still for 34 s and over 41 s after reaching a goal. There were two causes, both in our code:

1. **Negative bids.** A stopped robot bid less than 0 and lost to all robots bidding 0, so it got no token. See point 3 of the decisions for the bids.
2. **Turning in place did not start.** A robot facing a wall had a tree with a few useless nodes, so the rule "turn when the tree has only the root" did not apply. See [turning in place](#turning-in-place-not-in-the-paper).

With both fixed, the same run has longest stops of 4 s and 9 s for these two robots. Single robot results (`compare_execution`) did not change at all.

### Effect of the fixes

Goals per robot in 600 s, 24 seeds. "First version" is `merit` before step 8, "new turning" is the turning rule alone, and `merit` has both fixes.

| Scene | First version | New turning | `merit` | `round_robin` |
|---|---|---|---|---|
| `pillar_4` | 8.45 | 10.71 | 10.69 | 10.59 |
| `pillar_6` | 8.92 | - | 11.24 | 10.60 |
| `pillar_8` | 8.65 | 10.52 | 10.64 | 9.77 |
| `open_10` | 10.38 | 10.88 | 11.08 | 9.83 |
| `four_agents_4` | - | 4.60 | 5.19 | 5.10 |

- **Our earlier conclusion was wrong.** We wrote that bids as in eq. 5 are worse than the fixed order on the pillar map. That came from these two bugs. `merit` is now at least equal to `round_robin` in all scenes.
- **The turning rule explains most of the change on the pillar map.** Non-negative bids matter on `four_agents_4`, and a little on `open_10`.
- **The threshold does not matter much.** In 300 s runs on `pillar_4`, `pillar_8` and `open_10`, 22 seeds, `min_tree_growth` of 1, 2, 5, and 10 are close to each other. We use 2.

### Against the paper

| Result | Paper | Ours | Verdict |
|---|---|---|---|
| Ten robots, open map: `round_robin` to `merit` | 12.5 to 15.1 (+21%) | 9.83 to 11.08 (+13%) on `open_10` | Same direction, smaller gain, positions are our choice |
| Four robots with obstacles: `round_robin` to `merit` | 11.9 to 11.9 | 5.10 to 5.19, overlapping, on `four_agents_4` | Reproduced |
| Four robots with obstacles: `merit` to cooperative | 11.9 to 13.0 (+9%) | 5.19 to 4.46 (-14%), or 4.66 with stop nodes every 16 s | Not reproduced |
| Fig. 4: without emergency stops, two robots block each other in a hallway | Deadlock | 11.65 on `hallway_2`, against 11.77 without any coordination, cooperative 9.23 | Deadlock did not appear |

### Why emergency stops gain nothing here

**In our runs there is almost nothing for them to fix.** They are meant to solve deadlocks, and `merit` has few. On `hallway_2`, its longest stop was 5 s in all 22 of the 300 s runs below. On `four_agents_4`, even `none`, which never waits for a token, is only 4% above `merit` (5.42 against 5.19), less than the paper's gain of 9%.
`none` is not a strict upper limit, but this leaves little room, while stop nodes still block space.

We tried two explanations for the difference from the paper, in 300 s runs, and neither changes the result. Goals per robot:

| Samples per cycle (22 seeds) | `hallway_2`, `merit` | `hallway_2`, `cooperative` | `four_agents_4`, `merit` | `four_agents_4`, `cooperative` |
|---|---|---|---|---|
| 10 | 5.52 | 4.59 | 2.39 | 2.02 |
| 30 | 5.48 | 5.02 | 2.48 | 2.19 |
| 100 | 5.41 | 4.93 | 2.61 | 2.20 |
| 300 | 5.45 | 5.23 | 2.61 | 2.39 |

| Scale of the map (12 seeds) | `hallway_2`, `merit` | `hallway_2`, `cooperative` | `four_agents_4`, `merit` | `four_agents_4`, `cooperative` |
|---|---|---|---|---|
| 6 pixels = 1 m (hallway 4.0 m) | 5.54 | 5.29 | 2.42 | 2.19 |
| 7 pixels = 1 m (hallway 3.4 m) | 6.21 | 5.04 | 2.71 | 2.33 |
| 8 pixels = 1 m (hallway 3.0 m) | 7.21 | 4.00 | 2.81 | 2.54 |
| 9 pixels = 1 m (hallway 2.7 m) | 7.04 | 3.62 | - | - |

Smaller maps have shorter trips, so compare within a row, not between rows.

### Limits of this check

- **The reason for the missing deadlock is unknown.** The paper says narrow passages are hard for its planner, while ours gets through them even with 10 samples per cycle. Turning in place, which is not in the paper, may also help: step 10 shows that robots with `L1 = 1.5 m` get stuck without it.
- **The paper's absolute numbers are out of reach.** Its speed and map size are not given. `four_agents_4` is about 50 m between goals at 1 m/s, which limits it to about 10 goals, and reaches 5.19.
- **The maps are read from figures by eye**, and which goal each robot goes to first is our choice.
- **The loss table of step 7 was not repeated** with the fixes.
- **`cooperative_paper` still reaches few goals.** Non-negative bids do not change its problem, since blocked robots bid high, not low.
- **Robots rarely meet inside the hallway of `hallway_2`.** "Deadlock did not appear" is therefore a weak result: the later robot usually enters when the first has passed, so the deadlock of Fig. 4 was not really tested.

## Robots running their own cycles (step 9)

Until step 8, all robots ran one cycle at the same time, one bid per cycle, with the token passed once per cycle. `AsyncTeam` changes this, next to `Team`, which stays as it was: a fixed run on five scene and coordination pairs gives exactly the same numbers as before this step.

**What it does.** One process runs a virtual clock of 0.1 s. Each robot has its own 1 s cycle, starting at a random step:

1. At the start of its cycle a robot commits, as before.
2. During the cycle its samples are spread over the steps, so messages arrive while the tree grows.
3. One cycle after getting the token, a robot selects its plan and sends its waypoints and the winner, even in the middle of its own cycle. The plan starts at the robot's root, which is at or after this time, so what it sends is never in the past.

### What the paper says about bids

The paper is not consistent on this, and we only noticed it in this step:

| Where | Rule |
|---|---|
| Algorithm 4, lines 12-13 | Every robot without the token bids in every iteration |
| Sec. 5.3 | A bid is sent "when an agent without the token finds a better path", and bids are "much more frequent" than plans |

So one bid per cycle was already what Algorithm 4 says, except for the token holder bidding too. Note that n-1 bids per plan could also be what "much more frequent" means. `compare_async` therefore runs both readings.

### Decisions the paper leaves open

- **Old bids.** A robot's bid goes back to 0 when it sends a plan, since it just took its best path. This replaces our earlier rule of the token holder bidding too, which was added to stop two robots passing the token to each other.
- **Only when better.** A bid is sent only when it is higher than the last one this robot sent since its last plan. So bids that went down are not sent: a robot can win with an old bid and change nothing. This shows in "Plans that changed" below.
- **When a token is used.** One cycle after the robot gets it. We first used it at the end of the robot's current cycle, following the loop of Algorithm 4 (`async_*` below). That passes the token about 2 times per second, while Sec. 7.2.3 says the token takes about 10 s to go around ten robots, so about once per second. The `timed_*` variants match that, and are the default. The length of the paper's cycle is not given: our 1 s is what makes this match.
- **How often to check for a better path.** The paper gives no number. We use 0.2 s.
- **Emergency stops.** A robot also shares the time of its root at the start of each cycle, so that a stop node it has already passed is not used. This is an extra message, not in the paper.

### Checking that it is safe

A plan message can now arrive in the middle of another robot's cycle, so we checked the following at each control step: what each robot believes about each other robot, against where that robot really is, and whether current plans conflict after each plan message.

To check that these counters can fire, we broke the token rule on purpose on `pillar_8`, 120 s, one seed, with all robots changing their plans and messages arriving 1 s late: 188 of 834 checks had conflicts and 3787 of 67200 beliefs were wrong, against 0 and 0 with the token. The minimum distance stayed above the limit (2.110 m) in this broken run, so the minimum distance alone does not show that the rule holds.

### Results

Goals per robot in 600 s, 24 seeds. In all 75 settings of this step, plans never conflicted (0 of about 1.5 million checks), no belief was wrong (0 of about 370 million), and no run came under 2.1 m (lowest 2.100 m).

Names: `same` is all cycles starting together, `async` and `timed` are random phases with the token used as described above. `every` checks for a bid once per cycle, `fast` every 0.2 s, and `better` only sends a bid higher than the last one.

**Token timing**, with `fast_better` bids for `merit` (`Team` has its own bid per cycle). In brackets is the gain over `round_robin` in the same column:

| Scene | Synchronous `Team` (1 plan/s): `round_robin` / `merit` | `async` (about 2 plans/s): `round_robin` / `merit` | `timed` (1 plan/s): `round_robin` / `merit` |
|---|---|---|---|
| `open_10` | 9.83 / 11.08 (+13%) | 10.99 / 11.67 (+6%) | 9.78 / 11.10 (+13%) |
| `pillar_8` | 9.77 / 10.64 (+9%) | 10.98 / 11.47 (+4%) | 9.46 / 10.78 (+14%) |
| `pillar_4` | 10.59 / 10.69 (+1%) | 10.74 / 10.74 (0%) | 10.50 / 10.66 (+2%) |
| `four_agents_4` | 5.10 / 5.19 (+2%) | 5.34 / 5.43 (+2%) | 5.05 / 4.93 (-2%) |
| `hallway_2` | 11.65 / 11.65 (0%) | 11.69 / 11.69 (0%) | 11.46 / 11.46 (0%) |

**Bids**, `merit` with `timed` tokens. The last columns are bid messages per second for each robot:

| Scene | `every` | `fast` | `fast_better` | Messages: `every` | `fast` | `fast_better` |
|---|---|---|---|---|---|---|
| `open_10` | 10.93 | 11.11 | 11.10 | 0.90 | 4.50 | 0.09 |
| `pillar_8` | 10.60 | 10.78 | 10.78 | 0.87 | 4.37 | 0.15 |
| `pillar_4` | 10.66 | 10.64 | 10.66 | 0.75 | 3.75 | 0.12 |
| `four_agents_4` | 5.04 | 5.02 | 4.93 | 0.75 | 3.75 | 0.07 |
| `hallway_2` | 11.46 | 11.46 | 11.46 | 0.50 | 2.50 | 0.07 |

**Phases alone**, `same_every` against `timed_every`, both 1 plan/s:

| Scene | `round_robin`: `same` / `timed` | `merit`: `same` / `timed` |
|---|---|---|
| `open_10` | 9.65 / 9.78 | 11.34 / 10.93 |
| `pillar_8` | 9.53 / 9.46 | 10.99 / 10.60 |
| `pillar_4` | 10.58 / 10.50 | 10.74 / 10.66 |
| `four_agents_4` | 5.17 / 5.05 | 5.22 / 5.04 |
| `hallway_2` | 11.48 / 11.46 | 11.48 / 11.46 |

**Emergency stops**, with `fast_better` bids (`Team` has its own bid per cycle):

| Scene | Synchronous: `merit` / `cooperative` / `cooperative_paper` | `timed`: `merit` / `cooperative` / `cooperative_paper` | `async`: `merit` / `cooperative` / `cooperative_paper` |
|---|---|---|---|
| `open_10` | 11.08 / 10.34 / 0.72 | 11.10 / 11.28 / 0.69 | 11.67 / 11.62 / 0.72 |
| `pillar_8` | 10.64 / 7.67 / 0.02 | 10.78 / 8.53 / 0.05 | 11.47 / 8.86 / 0.05 |
| `pillar_4` | 10.69 / 10.24 / 2.42 | 10.66 / 10.44 / 1.47 | 10.74 / 10.68 / 1.28 |
| `four_agents_4` | 5.19 / 4.46 / 0.50 | 4.93 / 4.50 / 0.45 | 5.43 / 4.88 / 0.47 |
| `hallway_2` | 11.65 / 9.23 / 1.83 | 11.46 / 10.79 / 1.52 | 11.69 / 10.85 / 1.46 |

### How to read the results

- **The token rate matters more than the phases, the bid rate, or the bid rule.** With about 2 plans/s, `round_robin` gains most on the scenes with many robots (`open_10` 9.78 to 10.99), and the gain of bids over it falls. The likely reason is that the wait for a token is about half as long, but we did not test this on its own.
- **These three features do not explain the gap to the paper.** On the ten robot scene, the setting closest to the paper gains +13%, the same as the synchronous `Team`, against the paper's +21%.
- **Sending bids more often adds little.** `fast` is at most 0.18 above `every`, and the intervals overlap.
- **Sending only better bids changes nothing we can measure, and needs far fewer messages.** It sends 29 to 54 times fewer than `fast`. The old bids it leaves behind did not cost tokens: the share of plans that changed is about the same.
- **Random phases cost `merit` a little, and we do not know why.** `same_every` is above `timed_every` for `merit` on `open_10` and `pillar_8`, with intervals that do not overlap, and not for `round_robin`.
- **Emergency stops still gain nothing on the paper's scene.** `four_agents_4` goes from 4.93 to 4.50 (-9%), against the paper's +9%. `cooperative` is closer to `merit` than in the synchronous `Team` on most scenes, and above it on `open_10`, with overlapping intervals.
- **`cooperative_paper` stays low.** The timing does not solve its problem.

### Against the paper, after this step

| Result | Paper | Ours (`timed`, `fast_better`) | Verdict |
|---|---|---|---|
| Ten robots, open map: `round_robin` to `merit` | +21% | +13% on `open_10` | Same direction, smaller gain, same as step 8 |
| Four robots with obstacles: `round_robin` to `merit` | 0% | -2%, overlapping | Reproduced |
| Four robots with obstacles: `merit` to cooperative | +9% | -9% | Not reproduced |
| Fig. 4 deadlock | Deadlock | Robots rarely meet in the hallway | Not tested |

### Limits of this step

- **Not real parallelism.** It is one process with a virtual clock, not one computer per robot in real time. All robots get the same number of samples in each cycle and the network has no delay, so what this cannot show is the effect of computing time, jitter, or message delay.
- **Phases are random but fixed.** Cycles never drift against each other during a run.
- **In the videos, a robot turning in place gets its new yaw at the end of the cycle**, so it jumps once per second. Its position is right, and only its position matters to the others.

## The setup closest to the paper (step 10)

This step runs the team of step 9 with the parts of the paper we had replaced, and checks the code against Algorithms 1-8. No setup follows the paper in everything: the paper does not say how its robots leave a wall without turning in place, and part of Algorithm 1 is still open (see below).

| Option | Effect |
|---|---|
| `compare_async --paper_tree` | Grows the trees with `PaperCLRRT` |
| `compare_async --clip_each_wheel` | Each wheel limited on its own, as stated in Sec. 7.1 |
| `compare_async --no_turning` | Never uses the rule of turning in place |
| `compare_async --lookahead` | `L1`, 1.5 m by default |
| `demo_team --paper` | `PaperCLRRT` and each wheel limited on its own, with `L1 = 1.5 m` and the turning rule |

### A small `L1` does not match the paper

We first used `L1 = 0.3 m` to do without the turning rule, and called it the setup closest to the paper. **That was wrong.** Sec. 7.1 says the control law tracks the reference "with a smooth trajectory", and Figs. 5 and 7 show robots making wide U-turns. With `L1 = 0.3 m`, the robot turns on the spot when the look-ahead point is more than about 27° off its heading, and goes straight otherwise. The first CL-RRT of step 2 had curved trajectories because it used `L1 = 1.5 m` from the start, and that was the right look.

A robot following a line at 90° to its heading (`Lw = 2`, `v = 1`, `vmax = 2`, each wheel limited on its own):

| `L1` | Largest distance from the line | Distance moved while turning | Time turning |
|---|---|---|---|
| 1.5 m | 0.78 m | 6.67 m | 6.8 s |
| 0.6 m | 0.115 m | 2.42 m | 3.0 s |
| 0.3 m | 0.021 m | 1.10 m | 1.8 s |
| 0.15 m | 0.003 m | 0.44 m | 1.2 s |

The robot only turns on the spot if `L1` is about 0.67 m or less. Without the turning rule, a larger `L1` leaves robots stuck at walls, and a smaller one has no visible curves:

22 seeds, 300 s, no turning rule, each wheel limited on its own. Each cell has the goals per robot, and the longest stop in a run as median / max. The last row is the turning rule with the earlier wheel limit.

| `L1` | `hallway_2`, our tree | `hallway_2`, paper's tree | `four_agents_4`, our tree | `four_agents_4`, paper's tree |
|---|---|---|---|---|
| 1.5 m | 3.23, 284 / 294 s | 1.07, 294 / 294 s | 1.68, 243 / 286 s | 0.52, 284 / 290 s |
| 0.6 m | 5.39, 3 / 4 s | 4.55, 3 / 4 s | 2.60, 11 / 253 s | 1.58, 15 / 240 s |
| 0.3 m | 5.34, 3 / 4 s | 4.59, 3 / 6 s | 2.51, 10 / 31 s | 1.83, 11 / 17 s |
| 0.15 m | 5.07, 4 / 4 s | 4.41, 4 / 5 s | 2.36, 10 / 48 s | 1.80, 13 / 20 s |
| 1.5 m, turning rule | 5.45, 4 / 8 s | 4.25, 6 / 9 s | 2.49, 11 / 48 s | 1.69, 21 / 47 s |

**So we keep `L1 = 1.5 m` and the turning rule.** The rule is not in the paper, but it only acts on a stopped robot and does not change the trajectories in the tree, while a small `L1` changes all of them. We do not know how the paper's robots avoid getting stuck at a wall, since the paper does not discuss it.

### Algorithms 1-8 against the code

**Follows the paper:**

| Part | Code |
|---|---|
| Algorithm 1, lines 1-11 | `PaperCLRRT`: uniform samples, only the nearest node, stops at the first unsafe state, a node at the final reference |
| Algorithm 2, lines 6-18 | `CLRRTExecutor`: grow, select the best path, lazy check, select again |
| Algorithm 3 | `CLRRT.lazy_check` |
| Algorithms 4-5 | `AsyncTeam` and `Agent`: only the token holder changes its plan, broadcasts waypoints and the winner, the others keep their plans and bid. Ties are random |
| Eq. 5, 7-8, 11-12 | `compute_bid`, `best_node`, `SkidSteerModel`, `PurePursuit` |
| Algorithm 6, lines 4-20, as far as the differences below allow | `CooperativeAgent(as_in_paper=True)`: tree ignores the others, Algorithm 7 for the first blocker, stop nodes about every 4 s, the stopped robot gets the token |
| Algorithm 8 | `CooperativeAgent.receive_plan` |

**Differs from the paper, all of these in every setup unless noted:**

| # | Paper | Ours | Status |
|---|---|---|---|
| 1 | Algorithm 1, line 12: "for each feasible node N produced", while line 11 and Sec. 3.1 describe one node. Fig. 7(b) shows about 14 waypoints on one plan | `PaperCLRRT` adds one node per sample | Unclear in the paper, and our result looks different, see below. **Not tested** |
| 2 | Algorithm 2, lines 5 and 7: the tree grows from the state at `t + Δt` | The root moves to the first node the robot cannot avoid, which can be much later than `t + Δt` | Our decision, depends on 1 |
| 3 | Algorithm 2, line 6: grows for a fixed time | 30 samples per cycle | Our decision, so that runs can be repeated. Step 8 tests other numbers |
| 4 | Algorithm 4, line 2: the first token goes to a random robot | Always robot 0 | Simple to change, we expect little effect after the first seconds. Not tested |
| 5 | Algorithm 4, line 12: the bid is the difference of the costs | Not below 0, and set back to 0 after sending a plan | The first is our reading of eq. 5 and fixes starving robots (step 6), the second is our decision (step 9). Not in the paper |
| 6 | Sec. 7.2.2: the cost is travel time. How partial paths compare is not given | Adds 2 times the straight line time to the goal, and `NO_GOAL_PENALTY` | Our decision. Its effect on `cooperative_paper` is not tested |
| 7 | Sec. 5.3: the message has waypoints and the winner | Also the start state, its time, and where the reference starts | Needed because of 2 and of the turning rule |
| 8 | Not in the paper | Turning rule | See above |
| 9 | Algorithm 6, line 13: always takes the path, cut short if needed | Can keep the current plan, also with `as_in_paper=True` | Our decision, not tested without it |
| 10 | Algorithm 7, line 8: the estop message goes to the stopped robot | Sent to all robots with the plan | Our decision, the others only learn it earlier |
| 11 | Algorithm 6, line 4: the tree ignores the others | `cooperative` avoids them, and can use any branch. `cooperative_paper` does not | `cooperative` is ours by design (step 7) |

### Nodes per plan: the largest open difference

`four_agents_4`, 3 seeds, 120 s, each wheel limited on its own, `L1 = 1.5 m`, measured at every plan sent:

| | Fig. 7 of the paper, by eye | Paper's tree, `merit` | Our tree, `merit` |
|---|---|---|---|
| Nodes on a plan (median) | About 14 in Fig. 7(b) | 4 | 14 |
| Time of a segment, median / 90th percentile | - | 3.8 / 9.0 s | 2.1 / 2.9 s |
| Time the robot is committed, median / max | - | 2.6 / 32.4 s | 2.0 / 5.3 s |
| Nodes in the tree (median) | - | 38 | 340 |
| Emergency stop nodes on a plan (median, `cooperative`) | About 3 in Fig. 7(c) | 1 | 4 |

The figure is one snapshot, so this is not a measurement of the paper. Still, our reading of Algorithm 1 has few stop nodes to choose from and can commit a robot for over 30 s, which limits emergency stops. Whether more nodes per expansion fix this is not tested. `open_10` was not measured, because this run hit its time limit.

### Results

24 seeds, 600 s, `timed_fast_better`, the paper's tree, each wheel limited on its own, `L1 = 1.5 m`, and the turning rule. This is the setup closest to the paper, with the differences above.

| Scene | Coordination | Goals per robot | 95% interval | Longest stop in a run, median / max | Emergency stops |
|---|---|---|---|---|---|
| `hallway_2` | `round_robin` | 9.12 | [8.82, 9.43] | 6 / 11 s | 0 |
| `hallway_2` | `merit` | 9.12 | [8.82, 9.43] | 6 / 11 s | 0 |
| `hallway_2` | `cooperative` | 8.40 | [7.96, 8.83] | 21 / 56 s | 108 |
| `hallway_2` | `cooperative_paper` | 2.12 | [1.56, 2.69] | 461 / 569 s | 39 |
| `four_agents_4` | `round_robin` | 3.49 | [3.34, 3.64] | 22 / 87 s | 0 |
| `four_agents_4` | `merit` | 3.54 | [3.33, 3.76] | 21 / 49 s | 0 |
| `four_agents_4` | `cooperative` | 3.68 | [3.52, 3.84] | 41 / 106 s | 166 |
| `four_agents_4` | `cooperative_paper` | 0.90 | [0.68, 1.12] | 536 / 594 s | 63 |
| `open_10` | `round_robin` | 8.47 | [8.32, 8.62] | 19 / 30 s | 0 |
| `open_10` | `merit` | 9.12 | [8.92, 9.31] | 57 / 72 s | 0 |
| `open_10` | `cooperative` | 9.05 | [8.84, 9.25] | 99 / 160 s | 664 |
| `open_10` | `cooperative_paper` | 0.59 | [0.49, 0.69] | 600 / 600 s | 22 |

**Safety.** All 12 settings had 0 conflicts in about 173,000 plans, 0 wrong beliefs in about 60 million, and a minimum distance of 2.100 m between centers, which is the limit. `round_robin` and `merit` are identical on `hallway_2`, because with two robots the token can only go to the other robot.

For comparison, the same seeds without the turning rule and with `L1 = 0.3 m`, which we no longer consider close to the paper, on the paper's tree and then on ours:

| Scene | Coordination | Paper's tree | Our tree |
|---|---|---|---|
| `hallway_2` | `merit` | 9.73 [9.50, 9.96] | 11.21 [10.99, 11.43] |
| `hallway_2` | `cooperative` | 7.92 [7.00, 8.83] | 9.77 [9.00, 10.55] |
| `hallway_2` | `cooperative_paper` | 2.83 [1.91, 3.76] | 2.50 [1.81, 3.19] |
| `four_agents_4` | `round_robin` | 4.03 [3.89, 4.17] | 5.15 [4.94, 5.36] |
| `four_agents_4` | `merit` | 4.02 [3.91, 4.13] | 5.17 [4.94, 5.39] |
| `four_agents_4` | `cooperative` | 3.81 [3.69, 3.94] | 4.73 [4.54, 4.92] |
| `four_agents_4` | `cooperative_paper` | 0.59 [0.46, 0.73] | 0.50 [0.24, 0.76] |
| `open_10` | `round_robin` | 8.06 [7.86, 8.27] | 9.87 [9.72, 10.02] |
| `open_10` | `merit` | 8.91 [8.68, 9.15] | 11.35 [11.19, 11.52] |
| `open_10` | `cooperative` | 8.43 [8.18, 8.69] | 11.08 [10.93, 11.22] |
| `open_10` | `cooperative_paper` | 0.62 [0.53, 0.71] | 0.68 [0.54, 0.83] |

### How to read the results

- **`merit` gains less than in the paper.** +8% on `open_10`, with intervals that do not overlap.
- **Emergency stops are no longer clearly worse on the paper's scene.** `cooperative` is 4% above `merit` on `four_agents_4`, the paper's direction, but the intervals overlap, so this is not a measured gain. It is also not the method of the paper: see 11 in the table above.
- **`cooperative_paper`, the closest to Algorithm 6, still reaches very few goals**, with stops of up to 600 s.
- **On `hallway_2`, emergency stops still cost goals**, 9.12 to 8.40, and `merit` shows no deadlock.
- **The turning rule is one possible reason for the missing deadlock.** It frees a stopped robot, which the paper does not describe.

### Against the paper, after this step

| Result | Paper | Closest setup | Verdict |
|---|---|---|---|
| Ten robots, open map: `round_robin` to `merit` | +21% (12.5 to 15.1) | +8% (8.47 to 9.12) | Same direction, smaller gain |
| Four robots with obstacles: `round_robin` to `merit` | 0% (11.9 to 11.9) | +1% (3.49 to 3.54), overlapping | Reproduced |
| Four robots with obstacles: `merit` to cooperative | +9% (11.9 to 13.0) | +4% with `cooperative`, overlapping. -75% with `cooperative_paper` | Not reproduced |
| Fig. 4: DMA-RRT stuck at a hallway, cooperation solves it | Solved by emergency stops | No deadlock without them, fewer goals with them | Not reproduced |

### Why the robots take so long to reach their goals

The robots do reach their goals, but by a long way, even with no other robot on the map. One robot on `four_agents_4`, where the goals are 49.8 m apart in a straight line, 200 s, `L1 = 1.5 m`, 2 seeds per tree:

| Tree | Length of a trip | First plan reaching the goal | Plans that changed |
|---|---|---|---|
| Paper's | 75 to 83 m | 16 to 59 s | 18 to 20 of 199 |
| Ours | 70 to 77 m | 4 to 7 s | 7 to 16 of 199 |

The team then loses more on top of this. A short test with 8 seeds and 200 s, `merit`, and the goals scaled to 600 s, not comparable with the 600 s runs above because the first trip weighs more:

| Setting | `hallway_2` | `four_agents_4` |
|---|---|---|
| Paper's tree, 30 samples per cycle | 7.69 ± 0.67 | 2.72 ± 0.48 |
| Paper's tree, 300 samples per cycle | 10.12 ± 0.73 | 3.66 ± 0.33 |
| Paper's tree with a node every 2 m along a reference, 30 samples | 7.88 ± 0.92 | 3.47 ± 0.48 |
| The same with 300 samples | 10.12 ± 0.48 | 3.38 ± 0.56 |
| Our tree, 30 samples per cycle | 10.31 ± 0.37 | 4.69 ± 0.46 |

More samples help the paper's tree, and more nodes per expansion do not clearly help. The node variant lives in a test script, not in `dma_rrt/`. We did not test why the trips are long. Changing the weight of the time to the goal to 1 or 5 did not shorten them.

### Limits of this step

- **Difference 1 is only tested in the short run above, without emergency stops**, so we cannot yet say whether it explains the gap for emergency stops.
- **The map of `hallway_2` is read from Fig. 4 by eye**, and the paper gives no goal counts for it.
- **The paper does not give its speed, map size, or planning time per cycle**, so only the gains can be compared, not the goal counts.

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
- **An obstacle appearing on top of the robot is not handled.** The safety action stops the robot, but cannot move it away.
  In the inspected runs (ours, no turning), the obstacle's edge always appeared at least 1.454 m from the robot's center.
- **The new obstacle in the tests is known to the planner immediately.** There is no sensing, and no moving obstacle yet.
- **No timing guarantee.** Plans use a number of samples instead of a time budget, to be reproducible.
