"""
    Test scenes for the sixteen pillar map
"""

from __future__ import annotations

from typing import List
import numpy as np

from cl_rrt import CLRRT
from collision import CollisionChecker, MovingObstacle
from dynamics import PurePursuit

ROBOT_RADIUS = 1.0

# lines through the corridors of the map, crossing the way from (1, 1) to (24, 24)
CORRIDOR_LINES = [
    ((24.0, 6.3), (0.5, 6.3)),
    ((12.4, 24.0), (12.4, 0.5)),
    ((0.5, 18.7), (24.5, 18.7)),
]


def corridor_obstacles(checker: CollisionChecker, controller: PurePursuit) -> List[MovingObstacle]:
    """
    Robots with fixed plans driving through the corridors from time 0. They do not react to anything,
    and stay at the end of their lines
    """
    planner = CLRRT(checker, controller)
    obstacles = []
    for start, end in CORRIDOR_LINES:
        start_pos, end_pos = np.array(start), np.array(end)
        direction = end_pos - start_pos
        start_state = np.array([start_pos[0], start_pos[1], np.arctan2(direction[1], direction[0])])
        states = planner.simulate_waypoints(start_state, [end_pos])
        obstacles.append(MovingObstacle(states, 0.0, controller.dt, ROBOT_RADIUS))
    return obstacles
