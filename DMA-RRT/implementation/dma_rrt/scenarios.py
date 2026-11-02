"""
    Test scenes for the sixteen pillar map
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from dma_rrt.cl_rrt import CLRRT
from dma_rrt.collision import CollisionChecker, MovingObstacle, SIXTEEN_PILLAR_MAP
from dma_rrt.dynamics import PurePursuit

ROBOT_RADIUS = 1.0
OPEN_MAP = "DMA-RRT/implementation/maps/open_40/map.yaml"
HALLWAY_MAP = "DMA-RRT/implementation/maps/hallway/map.yaml"
FOUR_AGENTS_MAP = "DMA-RRT/implementation/maps/four_agents/map.yaml"

# lines through the corridors of the map, crossing the way from (1, 1) to (24, 24)
CORRIDOR_LINES = [
    ((24.0, 6.3), (0.5, 6.3)),
    ((12.4, 24.0), (12.4, 0.5)),
    ((0.5, 18.7), (24.5, 18.7)),
]


def pillar_swap(num_robots: int) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    """
    Robots swapping points on the border of the sixteen pillar map back and forth, in pairs on opposite sides:
    along the two diagonals first, then through the middle of the map
    :param num_robots: an even number up to 8
    :return: start state and goals of each robot
    """
    points = [
        (1.0, 1.0), (24.0, 24.0), (1.0, 24.0), (24.0, 1.0),
        (12.4, 1.0), (12.4, 24.0), (1.0, 12.5), (24.0, 12.5),
    ]
    yaws = [0.0, np.pi, 0.0, np.pi, 0.5 * np.pi, -0.5 * np.pi, 0.0, np.pi]
    start_states, goals = [], []
    for i in range(num_robots):
        opposite = i + 1 if i % 2 == 0 else i - 1
        start_states.append(np.array([points[i][0], points[i][1], yaws[i]]))
        goals.append([np.array(points[opposite]), np.array(points[i])])
    return start_states, goals


def open_map_scene() -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    """
    Ten robots on the open map, following the setup of Sec. 7.2 of the paper: five start at the bottom and five at
    the top, and all visit the same ten goals on a circle in order. The exact positions are not in the paper's text,
    so these are our choice. Goals follow each other across the map, not around the circle.
    :return: start state and goals of each robot
    """
    center = np.array([20.0, 20.0])
    angles = 2.0 * np.pi * np.arange(10) / 10.0
    circle = [center + 16.0 * np.array([np.cos(angle), np.sin(angle)]) for angle in angles]
    goal_list = [circle[(3 * k) % 10] for k in range(10)]

    start_states, goals = [], []
    for i in range(10):
        x = 8.0 + 6.0 * (i % 5)
        if i < 5:
            start_states.append(np.array([x, 2.0, 0.5 * np.pi]))
        else:
            start_states.append(np.array([x, 38.0, -0.5 * np.pi]))
        goals.append(goal_list[i:] + goal_list[:i])
    return start_states, goals


def _from_figure(frame: Tuple[float, float, float, float], x: float, y: float) -> np.ndarray:
    """
    Position in a map of maps/make_paper_maps.py from pixels of the paper's figure
    """
    left, _, _, bottom = frame
    return np.array([(x - left) / 6.0, (bottom - y) / 6.0])


def hallway_scene() -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    """
    Two robots on the map of Fig. 4 of the paper, which has a hallway too narrow to pass each other. The robots start
    as in the figure. In the figure each goes through once, here they then keep going back and forth
    """
    frame = (252.0, 93.0, 465.0, 211.0)
    goal_1 = _from_figure(frame, 273, 108)
    goal_2 = _from_figure(frame, 452, 129)
    robot_1 = _from_figure(frame, 432, 195)
    robot_2 = _from_figure(frame, 269, 132)
    start_states = [np.array([robot_1[0], robot_1[1], 0.5 * np.pi]), np.array([robot_2[0], robot_2[1], 0.0])]
    return start_states, [[goal_1, goal_2], [goal_2, goal_1]]


def four_agents_scene() -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    """
    Four robots on the map of Fig. 7(a) of the paper, starting as in the figure. All alternate between the same two
    goals (Sec. 7.3). Which goal each robot goes to first is not in the paper, here it is the far one
    """
    frame = (258.0, 75.0, 399.0, 403.0)
    top_goal = _from_figure(frame, 334, 82)
    bottom_goal = _from_figure(frame, 334, 381)
    starts = [(285, 85, np.pi), (355, 89, 0.5 * np.pi), (287, 387, 0.5 * np.pi), (356, 393, 0.0)]
    start_states, goals = [], []
    for i, (x, y, yaw) in enumerate(starts):
        position = _from_figure(frame, x, y)
        start_states.append(np.array([position[0], position[1], yaw]))
        goals.append([bottom_goal, top_goal] if i < 2 else [top_goal, bottom_goal])
    return start_states, goals


def make_scene(name: str) -> Tuple[List[np.ndarray], List[List[np.ndarray]], str]:
    """
    :param name: "open_10", "hallway_2", "four_agents_4", or "pillar_N" for N robots
    :return: start state of each robot, goals of each robot, map file
    """
    if name == "open_10":
        return (*open_map_scene(), OPEN_MAP)
    if name == "hallway_2":
        return (*hallway_scene(), HALLWAY_MAP)
    if name == "four_agents_4":
        return (*four_agents_scene(), FOUR_AGENTS_MAP)
    return (*pillar_swap(int(name.split("_")[1])), SIXTEEN_PILLAR_MAP)


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
