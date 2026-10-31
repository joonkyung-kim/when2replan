# DMA-RRT

An implementation, built step by step for study, of:

> V. R. Desaraju and J. P. How, "Decentralized path planning for multi-agent teams with complex constraints",
> *Autonomous Robots*, 32:385-403, 2012. DOI: [10.1007/s10514-012-9275-2](https://doi.org/10.1007/s10514-012-9275-2)

The code is in [implementation/](implementation/). It is separate from the `navigation_stack_py` stack of this
repository and only imports its `MapHandler` to load maps and query obstacle distances. Nothing in the existing
stack imports this folder.

## The paper in short

Several robots with nonlinear dynamics plan paths in a shared environment, each on its own computer. If they all
change their plans at the same time, new plans can conflict even though each one looked safe when it was chosen.

- **CL-RRT** is the single-robot planner. It grows a tree by simulating the closed-loop system, so every path in
  the tree is dynamically feasible.
- **Merit-based token passing (DMA-RRT)** lets only the token holder change its plan, so one plan changes at a
  time. Each other robot bids its potential path improvement (`PPI = cost(current path) - cost(best path in its
  tree)`), and the token goes to the highest bid. The tree already exists, so computing the bid needs no extra planning.
- **Emergency stops (Cooperative DMA-RRT)** let a robot ask another one to stop early at a safe node, when that
  lowers the combined cost. This resolves deadlocks such as two robots facing each other in a narrow passage.

## Status

| Step | Content | Paper | Status |
|---|---|---|---|
| 0 | Basic RRT on an empty plane | | Done |
| 1 | Collision check on a map | | Done |
| 2 | Tree expansion with closed-loop simulation | Algorithm 1, eq. 11-12 | Done |
| 3 | Persistent tree, execution loop, lazy check | Algorithms 2-3 | Done |
| 4 | Collision check against time-parameterized obstacles | Sec. 7.1 | Not started |
| 5 | Round-robin token with two robots | Sec. 7.2.1 baseline | Not started |
| 6 | PPI bids (DMA-RRT) | Algorithms 4-5 | Not started |
| 7 | Emergency stops (Cooperative DMA-RRT) | Algorithms 6-8 | Not started |

See [implementation/README.md](implementation/README.md) for how to run the code, the design decisions, and the
comparison between the paper's CL-RRT and ours.

## Files not in git

The paper's PDF, generated videos, and figures in this folder are ignored by [.gitignore](.gitignore). Get the
paper from the DOI above. The videos can be generated again with the commands in the implementation README.
