"""
    Skid-steer vehicle (eq. 11) and pure pursuit steering (eq. 12) of the DMA-RRT paper

    state: (x, y, yaw), yaw is positive counterclockwise
    NOTE: delta_v is v_right - v_left here, so that a positive delta_v turns counterclockwise.
    The paper writes it as v_left - v_right with a minus sign in eq. 12.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class SkidSteerModel:
    def __init__(self, wheel_base: float = 2.0, max_wheel_speed: float = 2.0) -> None:
        self.wheel_base = wheel_base
        self.max_wheel_speed = max_wheel_speed

    def saturate(self, v: float, delta_v: float) -> Tuple[float, float]:
        """
        Slow down until both wheels are within the limit, the curvature of the motion stays the same
        """
        max_speed = abs(v) + 0.5 * abs(delta_v)
        if max_speed <= self.max_wheel_speed:
            return v, delta_v
        scale = self.max_wheel_speed / max_speed
        return scale * v, scale * delta_v

    def step(self, state: np.ndarray, v: float, delta_v: float, dt: float) -> np.ndarray:
        x, y, yaw = state
        yaw_rate = delta_v / self.wheel_base
        mid_yaw = yaw + 0.5 * yaw_rate * dt
        return np.array(
            [
                x + v * np.cos(mid_yaw) * dt,
                y + v * np.sin(mid_yaw) * dt,
                wrap_angle(yaw + yaw_rate * dt),
            ]
        )


class PurePursuit:
    def __init__(
        self,
        model: SkidSteerModel,
        speed: float = 1.0,
        lookahead: float = 1.5,
        anchor: float = 0.0,
        dt: float = 0.1,
    ) -> None:
        """
        :param model: vehicle model
        :param speed: speed command [m/s]
        :param lookahead: distance to the look-ahead point on the reference (L1) [m]
        :param anchor: distance of the anchor point from the axle (l_a) [m]
        :param dt: control interval [s]
        """
        self.model = model
        self.speed = speed
        self.lookahead = lookahead
        self.anchor = anchor
        self.dt = dt

    def control(self, state: np.ndarray, target_pos: np.ndarray) -> Tuple[float, float]:
        """
        Compute (v, delta_v) to steer toward target_pos, within the wheel speed limit
        """
        to_target = target_pos - state[:2]
        eta = wrap_angle(np.arctan2(to_target[1], to_target[0]) - state[2])
        # sin(eta) goes back to 0 behind the robot, so turn at the limit there instead of going straight
        eta = max(min(eta, 0.5 * np.pi), -0.5 * np.pi)
        delta_v = (
            self.speed
            * self.model.wheel_base
            * np.sin(eta)
            / (0.5 * self.lookahead + self.anchor * np.cos(eta))
        )
        return self.model.saturate(self.speed, delta_v)

    def lookahead_point(
        self, position: np.ndarray, from_pos: np.ndarray, to_pos: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Get the look-ahead point on the line from from_pos to to_pos
        :return: look-ahead point, and whether the robot has passed the end of the line
        """
        line = to_pos - from_pos
        length = np.linalg.norm(line)
        direction = line / length
        progress = np.dot(position - from_pos, direction)
        cross_track = np.linalg.norm(position - (from_pos + progress * direction))
        ahead = np.sqrt(max(self.lookahead**2 - cross_track**2, 0.0))
        target_progress = max(min(progress + ahead, length), 0.0)
        return from_pos + target_progress * direction, progress >= length

    def max_segment_steps(self, from_pos: np.ndarray, to_pos: np.ndarray) -> int:
        """
        Number of steps to give up tracking a line, the robot can circle a point it cannot reach forever
        """
        length = np.linalg.norm(to_pos - from_pos)
        return int(max(20.0, 2.0 * length / self.speed) / self.dt)

    def simulate_segment(
        self, state: np.ndarray, from_pos: np.ndarray, to_pos: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Simulate the closed loop system tracking the line from from_pos to to_pos
        :param state: initial state
        :return: states after the initial state size: (N, 3), and whether the end of the line was reached
        """
        states = []
        current = state
        for _ in range(self.max_segment_steps(from_pos, to_pos)):
            target_pos, is_passed = self.lookahead_point(current[:2], from_pos, to_pos)
            if is_passed:
                return np.array(states).reshape(-1, 3), True
            v, delta_v = self.control(current, target_pos)
            current = self.model.step(current, v, delta_v, self.dt)
            states.append(current)
        return np.array(states).reshape(-1, 3), False
