from typing import List
import fire
import numpy as np
import pygame
import os
import pickle
import subprocess
import glob
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
from navigation_stack_py.gym_env import NavigationStackEnv
from navigation_stack_py.common import MovingObstacle, MotionModel, AgentState


def rhu(number: float, decimals=0) -> int:
    return int(Decimal(number).quantize(Decimal("0"), rounding=ROUND_HALF_UP))


def signed_area(polygon: np.ndarray) -> float:
    polygon = np.asarray(polygon)
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def render_static_map(
    map_polygons: List[np.ndarray], size: tuple, scale: float, center: tuple, color: list
) -> pygame.Surface:
    """
    Render static map polygons (contours of occupied cells) to a transparent layer.
    Outer contours and hole contours have opposite orientation (outer: negative area),
    so obstacles are filled and holes inside them (e.g. rooms surrounded by walls) are cut out again.
    """
    transparent = (255, 0, 255)
    layer = pygame.Surface(size)
    layer.fill(transparent)
    layer.set_colorkey(transparent)

    # draw larger contours first so that nested contours are drawn on top of them
    for polygon in sorted(map_polygons, key=lambda p: abs(signed_area(p)), reverse=True):
        points = [node * scale + center for node in polygon]
        if len(points) == 1:
            # single occupied cell
            pygame.draw.circle(layer, color, points[0], 0.1 * scale)
        elif len(points) == 2:
            # one cell wide line
            pygame.draw.line(layer, color, points[0], points[1], max(2, rhu(0.1 * scale)))
        elif signed_area(polygon) <= 0:
            # outer contour of obstacle
            pygame.draw.polygon(layer, color, points, width=0)
        else:
            # hole inside obstacle: cut out and keep its boundary
            pygame.draw.polygon(layer, transparent, points, width=0)
            pygame.draw.polygon(layer, color, points, width=max(2, rhu(0.1 * scale)))

    return layer


def draw_robot(
    canvas: pygame.Surface,
    robot: AgentState,
    color: List[int],
    robot_radius: float,
    line_width: int,
    scale: float,
    center: tuple,
) -> None:
    """
    Draw robot body and wheels
    """
    # body
    pygame.draw.circle(
        surface=canvas,
        color=color,
        center=(
            rhu(robot.pos[0] * scale + center[0]),
            rhu(robot.pos[1] * scale + center[1]),
        ),
        radius=robot_radius * scale,
        width=line_width,
    )

    # heading line
    # start_pos = robot.pos
    # end_pos = robot.pos + robot_radius * np.array(
    #     [np.cos(robot.yaw), np.sin(robot.yaw)]
    # )
    # pygame.draw.line(
    #     surface=canvas,
    #     color=color,
    #     start_pos=(
    #         rhu(start_pos[0] * scale + center[0]),
    #         rhu(start_pos[1] * scale + center[1]),
    #     ),
    #     end_pos=(
    #         rhu(end_pos[0] * scale + center[0]),
    #         rhu(end_pos[1] * scale + center[1]),
    #     ),
    #     width=line_width,
    # )

    # wheels
    wheel_length = robot_radius * 0.4
    wheel_width = robot_radius * 0.2
    # left wheel
    lw_center = robot.pos + (robot_radius - wheel_width / 2.0) * np.array(
        [-np.sin(robot.yaw), np.cos(robot.yaw)]
    )
    # polygon points
    left_wheel = [
        lw_center
        + wheel_length * np.array([np.cos(robot.yaw), np.sin(robot.yaw)])
        + wheel_width * np.array([-np.sin(robot.yaw), np.cos(robot.yaw)]),
        lw_center
        + wheel_length * np.array([np.cos(robot.yaw), np.sin(robot.yaw)])
        - wheel_width * np.array([-np.sin(robot.yaw), np.cos(robot.yaw)]),
        lw_center
        - wheel_length * np.array([np.cos(robot.yaw), np.sin(robot.yaw)])
        - wheel_width * np.array([-np.sin(robot.yaw), np.cos(robot.yaw)]),
        lw_center
        - wheel_length * np.array([np.cos(robot.yaw), np.sin(robot.yaw)])
        + wheel_width * np.array([-np.sin(robot.yaw), np.cos(robot.yaw)]),
    ]

    pygame.draw.polygon(
        surface=canvas,
        color=color,
        points=[node * scale + center for node in left_wheel],
        width=0,
    )

    # right wheel
    rw_center = robot.pos + (robot_radius - wheel_width / 2.0) * np.array(
        [np.sin(robot.yaw), -np.cos(robot.yaw)]
    )
    right_wheel = [
        rw_center
        + wheel_length * np.array([np.cos(robot.yaw), np.sin(robot.yaw)])
        + wheel_width * np.array([np.sin(robot.yaw), -np.cos(robot.yaw)]),
        rw_center
        + wheel_length * np.array([np.cos(robot.yaw), np.sin(robot.yaw)])
        - wheel_width * np.array([np.sin(robot.yaw), -np.cos(robot.yaw)]),
        rw_center
        - wheel_length * np.array([np.cos(robot.yaw), np.sin(robot.yaw)])
        - wheel_width * np.array([np.sin(robot.yaw), -np.cos(robot.yaw)]),
        rw_center
        - wheel_length * np.array([np.cos(robot.yaw), np.sin(robot.yaw)])
        + wheel_width * np.array([np.sin(robot.yaw), -np.cos(robot.yaw)]),
    ]

    pygame.draw.polygon(
        surface=canvas,
        color=color,
        points=[node * scale + center for node in right_wheel],
        width=0,
    )


def draw_goal(
    canvas: pygame.Surface,
    goal: np.ndarray,
    color: List[int],
    goal_radius: float,
    scale: float,
    center: tuple,
) -> None:
    """
    Draw goal circle
    """
    pygame.draw.circle(
        surface=canvas,
        color=color,
        center=(
            rhu(goal[0] * scale + center[0]),
            rhu(goal[1] * scale + center[1]),
        ),
        radius=goal_radius / 2.0 * scale,
        width=2,
    )


def draw_trajectory(
    canvas: pygame.Surface,
    robot_traj: List[np.ndarray],
    replan_history: List[bool],
    color: List[int],
    replan_color: List[int],
    scale: float,
    center: tuple,
) -> None:
    """
    Draw robot trajectory, positions where the global path was replanned are drawn larger with replan_color
    """
    if robot_traj != [] and len(robot_traj) > 2:
        for i, pos in enumerate(robot_traj):
            if replan_history[i]:
                point_color = replan_color
                size = 0.2
            else:
                point_color = color
                size = 0.1
            pygame.draw.circle(
                surface=canvas,
                color=point_color,
                center=(
                    rhu(pos[0] * scale + center[0]),
                    rhu(pos[1] * scale + center[1]),
                ),
                radius=size * scale,
                width=0,
            )


def draw_replan_marker(
    canvas: pygame.Surface,
    robot: AgentState,
    color: List[int],
    robot_radius: float,
    line_width: int,
    scale: float,
    center: tuple,
) -> None:
    """
    Draw circle around the robot when the global path is replanned
    """
    pygame.draw.circle(
        surface=canvas,
        color=color,
        center=(
            rhu(robot.pos[0] * scale + center[0]),
            rhu(robot.pos[1] * scale + center[1]),
        ),
        radius=(robot_radius) * scale,
        width=line_width,
    )


def draw_paths(
    canvas: pygame.Surface,
    reference_path: np.ndarray,
    local_paths: List[np.ndarray],
    best_index: int,
    reference_path_color: List[int],
    local_path_color: List[int],
    best_path_color: List[int],
    scale: float,
    center: tuple,
) -> None:
    """
    Draw reference path (global path), all local path candidates, and the best local path
    """
    # reference path
    if reference_path is not None:
        for pos in reference_path:
            pygame.draw.circle(
                surface=canvas,
                color=reference_path_color,
                center=(
                    rhu(pos[0] * scale + center[0]),
                    rhu(pos[1] * scale + center[1]),
                ),
                radius=0.05 * scale,
                width=0,
            )

    # all paths
    if local_paths is not None:
        for path in local_paths:
            pygame.draw.lines(
                surface=canvas,
                color=local_path_color,
                points=[
                    (
                        rhu(pos[0] * scale + center[0]),
                        rhu(pos[1] * scale + center[1]),
                    )
                    for pos in path
                ],
                closed=False,
                width=rhu(0.03 * scale),
            )

        # best local path
        best_path = local_paths[best_index]
        pygame.draw.lines(
            surface=canvas,
            color=best_path_color,
            points=[
                (rhu(pos[0] * scale + center[0]), rhu(pos[1] * scale + center[1]))
                for pos in best_path
            ],
            closed=False,
            width=rhu(0.05 * scale),
        )


def main(
    traj_data_pkl: str, map_pkl: str = None, save_path: str = None, headless: bool = False
) -> None:
    """
    headless: render without opening a window and as fast as possible (for saving the video only)
    """
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # simulation settings
    robot_radius = 1.0  # meters
    line_width = 10  # robot line width
    goal_radius = 1.0  # meters
    control_time_interval = 0.1  # seconds

    # Window settings
    window_width = 1400  # pixels
    window_height = 1400  # pixels
    center = (150, 150)
    scale = 45  # pixels per meter
    render_fps = 10  # frames per second
    font_size = 100  # pixels
    font_location = (20, 20)  # pixels

    # colors
    black = [20, 20, 40]
    lightblue = [0, 120, 255]
    darkblue = [0, 40, 160]
    red = [255, 0, 0]  # robot in collision
    amber = [255, 190, 0]  # replan marker around robot
    # orange = [255, 100, 0]
    white = [255, 255, 255]
    # blue = [0, 0, 255]
    lightgrey = [230, 230, 230]

    # tableu colors
    green = [44, 150, 44]
    lightgreen = [152, 223, 138]
    darkgreen = [20, 90, 20]
    orange = [255, 127, 14]
    olive = [188, 189, 34]
    blue = [31, 119, 180]
    # other robots in multi robot run (robot, local path candidates, best local path), indexed by robot id
    other_robot_colors = [
        ([148, 103, 189], [197, 176, 213], [90, 50, 130]),  # purple
        ([140, 86, 75], [196, 156, 148], [85, 50, 40]),  # brown
        ([227, 119, 194], [247, 182, 210], [160, 50, 130]),  # pink
        ([23, 190, 207], [158, 218, 229], [10, 115, 130]),  # cyan
        ([40, 40, 140], [160, 160, 220], [20, 20, 80]),  # navy
        ([0, 128, 110], [130, 200, 190], [0, 75, 65]),  # teal
        ([110, 110, 110], [190, 190, 190], [55, 55, 55]),  # grey
        ([140, 20, 80], [215, 150, 180], [80, 10, 45]),  # wine
    ]

    # get directory path of traj_data_pkl
    dir = os.path.dirname(traj_data_pkl)
    name = os.path.basename(traj_data_pkl)

    if map_pkl is None:
        map_pkl = os.path.join(dir, "map.pkl")

    if save_path is None:
        save_path = os.path.join(dir, name.replace(".pkl", ".mp4"))

    # Prepare video writer: pipe raw frames to ffmpeg to encode H.264 (opencv-python has no H.264 encoder)
    out = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "{}x{}".format(window_width, window_height),
            "-r", str(render_fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", save_path,
        ],
        stdin=subprocess.PIPE,
    )

    with open(traj_data_pkl, "rb") as f:
        # load data
        traj_data = pickle.load(f)

    with open(map_pkl, "rb") as f:
        # load map as list of polygon
        map_polygons = pickle.load(f)

    pygame.init()
    pygame.display.set_caption("Navigation")
    pygame.display.init()
    window = pygame.display.set_mode((window_width, window_height))
    font = pygame.font.Font(None, font_size)

    clock = pygame.time.Clock()
    canvas = pygame.Surface((window_width, window_height))
    canvas.fill(white)
    map_layer = render_static_map(
        map_polygons, (window_width, window_height), scale, center, black
    )

    robot_traj = []
    replan_history = []
    # trajectories of the other robots (multi robot run), indexed by the order in data["other_robots"]
    other_robot_trajs = defaultdict(list)
    other_replan_histories = defaultdict(list)

    length = len(traj_data)
    for i in range(length):
        canvas.fill(white)

        data = traj_data[i]

        scan = data["scan"]
        # inflation area for scans
        for point in scan:
            pygame.draw.circle(
                surface=canvas,
                color=lightgrey,
                center=(
                    rhu(point[0] * scale + center[0]),
                    rhu(point[1] * scale + center[1]),
                ),
                radius=robot_radius * scale,
                width=0,
            )
        # inflation area for static map
        for polygon in map_polygons:
            # edge
            edges = [
                (polygon[i], polygon[(i + 1) % len(polygon)])
                for i in range(len(polygon))
            ]
            interpolated_points = []
            for edge in edges:
                interpolated_points += list(
                    np.linspace(edge[0], edge[1], num=10, endpoint=False)
                )
            for point in interpolated_points:
                pygame.draw.circle(
                    surface=canvas,
                    color=lightgrey,
                    center=(
                        rhu(point[0] * scale + center[0]),
                        rhu(point[1] * scale + center[1]),
                    ),
                    radius=robot_radius * scale,
                    width=0,
                )

        # goal
        draw_goal(canvas, data["goal"], green, goal_radius, scale, center)
        for j, other_robot in enumerate(data.get("other_robots", [])):
            robot_color = other_robot_colors[other_robot.get("robot_id", j)][0]
            draw_goal(canvas, other_robot["goal"], robot_color, goal_radius, scale, center)

        # draw map
        canvas.blit(map_layer, (0, 0))

        robot = data["robot_state"]
        robot_traj.append(robot.pos)
        if data["is_replan"]:
            replan_history.append(True)
        else:
            replan_history.append(False)

        # draw robot trajectory
        draw_trajectory(canvas, robot_traj, replan_history, lightblue, green, scale, center)

        # draw robot, red if in collision
        ego_color = red if data.get("is_collision", False) else green
        draw_robot(canvas, robot, ego_color, robot_radius, line_width, scale, center)

        # if replan, draw amber circle
        if data["is_replan"]:
            draw_replan_marker(canvas, robot, amber, robot_radius, line_width, scale, center)

        # draw obstacles
        for obstacle in data["obstacles"]:
            pos = obstacle.pos
            if obstacle.motion_model == MotionModel.POINT_MASS_MODEL:
                color = blue
            else:
                color = orange
            pygame.draw.circle(
                surface=canvas,
                color=color,
                center=(
                    rhu(pos[0] * scale + center[0]),
                    rhu(pos[1] * scale + center[1]),
                ),
                radius=obstacle.size[0] * scale,
                # width=line_width,
            )

        # draw other robots in the same way as the robot (multi robot run)
        for j, other_robot in enumerate(data.get("other_robots", [])):
            robot_color, local_path_color, best_path_color = other_robot_colors[
                other_robot.get("robot_id", j)
            ]
            other_robot_trajs[j].append(other_robot["robot_state"].pos)
            other_replan_histories[j].append(other_robot["is_replan"])
            draw_trajectory(
                canvas,
                other_robot_trajs[j],
                other_replan_histories[j],
                robot_color,
                robot_color,
                scale,
                center,
            )
            draw_paths(
                canvas,
                other_robot["global_path"],
                other_robot["local_path_list"],
                other_robot["local_path_best_index"],
                robot_color,
                local_path_color,
                best_path_color,
                scale,
                center,
            )
            draw_robot(
                canvas,
                other_robot["robot_state"],
                red if other_robot.get("is_collision", False) else robot_color,
                robot_radius,
                line_width,
                scale,
                center,
            )
            if other_robot["is_replan"]:
                draw_replan_marker(
                    canvas,
                    other_robot["robot_state"],
                    amber,
                    robot_radius,
                    line_width,
                    scale,
                    center,
                )

        # scan points
        for point in scan:
            pygame.draw.circle(
                surface=canvas,
                color=olive,
                center=(
                    rhu(point[0] * scale + center[0]),
                    rhu(point[1] * scale + center[1]),
                ),
                radius=0.2 * scale,
                width=0,
            )

        # reference path and local paths
        draw_paths(
            canvas,
            data["global_path"],
            data["local_path_list"],
            data["local_path_best_index"],
            green,
            lightgreen,
            darkgreen,
            scale,
            center,
        )

        window.blit(canvas, (0, 0))

        # time stamp
        time = i * control_time_interval
        text = font.render("Time: " + str(round(time, 1)) + " s", True, (0, 0, 0))
        window.blit(text, font_location)

        pygame.event.pump()
        pygame.display.update()

        # save image
        screen_array = pygame.surfarray.array3d(window)
        screen_array = screen_array.transpose([1, 0, 2])
        out.stdin.write(np.ascontiguousarray(screen_array).tobytes())

        if not headless:
            clock.tick(render_fps)

    out.stdin.close()
    if out.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the video: {}".format(save_path))

    pygame.display.quit()
    pygame.quit()


def run_all(dir: str, headless: bool = False):
    # get all pkl file path in the dir recursively without map.pkl
    pkl_files = glob.glob(dir + "/**/*.pkl", recursive=True)

    # remove map.pkl
    pkl_files = [pkl_file for pkl_file in pkl_files if "map.pkl" not in pkl_file]

    for pkl_file in pkl_files:
        print("Processing: ", pkl_file)
        main(pkl_file, headless=headless)


if __name__ == "__main__":
    fire.Fire()
    # fire.Fire(main)
    # fire.Fire(run_all)
