"""
Simple simulation where balls travel in straight lines in a 2D space,
and bounce off of each other in an elastic manner.
"""

import dataclasses
from typing import Tuple

import cv2
import numpy as np

from grund.abstract import GrundEnv
from grund.util.spaces import ContinuousActionSpace, ObservationSpace


def _as_opencv_coordinates(np_vector: np.ndarray) -> Tuple[int, int]:
    return tuple(np_vector[::-1].astype(int).tolist())


@dataclasses.dataclass
class DodgeConfig:
    dt: float
    simulation_width: int
    simulation_height: int
    num_enemy_balls: int
    ball_radius: int
    ball_velocity: int


@dataclasses.dataclass
class DodgeStepResult:
    observation: np.ndarray
    reward: float
    done: bool


def _get_colliding_balls(ball_positions: np.ndarray, ball_radius: int) -> np.ndarray:
    deltas = ball_positions[:, None, :] - ball_positions[None, :, :]
    distances = np.sqrt(np.sum(deltas**2, axis=-1))
    np.fill_diagonal(distances, np.inf)

    close_balls = distances < 2 * ball_radius
    return np.unique(np.where(close_balls)[0])


def _get_colliding_pairs(ball_positions: np.ndarray, ball_radius: int) -> np.ndarray:
    deltas = ball_positions[:, None, :] - ball_positions[None, :, :]
    distances_squared = np.sum(deltas**2, axis=-1)
    radii_sum_squared = (2 * ball_radius) ** 2

    colliding = np.where(distances_squared < radii_sum_squared)
    colliding_pairs = np.column_stack(colliding)

    # Remove duplicates and self-collisions
    colliding_pairs = colliding_pairs[colliding_pairs[:, 0] < colliding_pairs[:, 1]]

    return colliding_pairs


class Dodge(GrundEnv):
    def __init__(self, config: DodgeConfig):
        super().__init__()
        self.cfg = config
        self.num_balls = self.cfg.num_enemy_balls + 1
        self.observation_space = ObservationSpace((self.num_balls, 4))
        self.action_space = ContinuousActionSpace(
            shape=(2,), minima=np.array([-2.0]), maxima=np.array([2.0])
        )
        self.ball_positions = np.zeros([self.num_balls, 2], dtype=int)
        self.ball_velocities = np.zeros(self.num_balls + 1, dtype=float)
        self.player_ball_idx = self.num_balls - 1

    def reset(self):
        w = self.cfg.ball_radius
        max_x, max_y = self.cfg.simulation_width - w, self.cfg.simulation_height - w
        ball_positions = np.random.randint(
            low=w,
            high=[max_x - 1, max_y - 1],
            size=(self.num_balls, 2),
        )
        ball_angles = np.random.uniform(
            low=0.0,
            high=2 * np.pi,
            size=self.num_balls,
        )
        ball_directions = np.stack([np.cos(ball_angles), np.sin(ball_angles)], axis=1)

        for iteration in range(1, 6):
            colliding_balls = _get_colliding_balls(
                ball_positions=ball_positions,
                ball_radius=self.cfg.ball_radius,
            )
            if len(colliding_balls) == 0:
                break
            ball_positions[colliding_balls] = np.random.randint(
                low=w,
                high=[max_x - 1, max_y - 1],
                size=(colliding_balls.size, 2),
            )
        else:
            raise RuntimeError(
                f"Couldn't place balls in {iteration} iterations. Config: {self.cfg}"
            )

        self.ball_positions = ball_positions
        self.ball_velocities = ball_directions * self.cfg.ball_velocity
        self.ball_velocities[-1, :] = 0.0

    def step(self, action: np.ndarray) -> DodgeStepResult:
        # 1. Update Positions
        enemy_ball_velocities = self.ball_velocities[:-1] * self.cfg.dt
        player_ball_velocity = self.ball_velocities[-1:] + action[None, :]
        new_ball_positions = self.ball_positions + np.concatenate(
            [enemy_ball_velocities, player_ball_velocity]
        )

        # 2. Handle Border Collisions
        x_max = self.cfg.simulation_width - self.cfg.ball_radius - 1
        hit_left_right = np.logical_or(
            new_ball_positions[:, 0] <= self.cfg.ball_radius,
            new_ball_positions[:, 0] > x_max,
        )
        y_max = self.cfg.simulation_height - self.cfg.ball_radius - 1
        hit_top_bottom = np.logical_or(
            new_ball_positions[:, 1] <= self.cfg.ball_radius,
            new_ball_positions[:, 1] > y_max,
        )

        self.ball_velocities[hit_left_right, 0] = -self.ball_velocities[
            hit_left_right, 0
        ]  # Reflect horizontal direction
        self.ball_velocities[hit_top_bottom, 1] = -self.ball_velocities[
            hit_top_bottom, 1
        ]  # Reflect vertical direction

        # Adjust positions for balls that hit the walls
        new_ball_positions[hit_left_right, 0] = np.clip(
            new_ball_positions[hit_left_right, 0],
            self.cfg.ball_radius,
            x_max,
        )
        new_ball_positions[hit_top_bottom, 1] = np.clip(
            new_ball_positions[hit_top_bottom, 1],
            self.cfg.ball_radius,
            y_max,
        )

        # 3. Handle Ball Collisions
        colliding_pairs = _get_colliding_pairs(
            ball_positions=new_ball_positions,
            ball_radius=self.cfg.ball_radius,
        )
        done = False
        for ball_1_idx, ball_2_idx in colliding_pairs:
            if ball_1_idx == self.player_ball_idx or ball_2_idx == self.player_ball_idx:
                done = True
                break
            pos_1, pos_2 = (
                new_ball_positions[ball_1_idx],
                new_ball_positions[ball_2_idx],
            )
            vel_1, vel_2 = (
                self.ball_velocities[ball_1_idx],
                self.ball_velocities[ball_2_idx],
            )

            # Calculate the normal and tangential vectors of the collision
            collision_vector = pos_2 - pos_1
            normal_vector = collision_vector / np.linalg.norm(collision_vector)
            tangent_vector = np.array([-normal_vector[1], normal_vector[0]])

            # Project the velocities onto the normal and tangential vectors
            vel_1_normal = np.dot(vel_1, normal_vector)
            vel_1_tangent = np.dot(vel_1, tangent_vector)
            vel_2_normal = np.dot(vel_2, normal_vector)
            vel_2_tangent = np.dot(vel_2, tangent_vector)

            # Calculate new normal velocities (since the balls have the same mass, they exchange velocities)
            vel_1_normal, vel_2_normal = vel_2_normal, vel_1_normal

            # Convert the normal and tangential velocities back to the original coordinate system
            vel_1_new = vel_1_normal * normal_vector + vel_1_tangent * tangent_vector
            vel_2_new = vel_2_normal * normal_vector + vel_2_tangent * tangent_vector

            # Update the velocities
            self.ball_velocities[ball_1_idx] = vel_1_new
            self.ball_velocities[ball_2_idx] = vel_2_new

        self.ball_positions = new_ball_positions
        # assert np.isclose(np.linalg.norm(self.ball_velocities, axis=1).mean(), self.cfg.initial_ball_velocity)
        print(
            "Total system velocity:",
            np.linalg.norm(self.ball_velocities, axis=-1).mean(),
        )
        return DodgeStepResult(
            observation=np.stack([self.ball_positions, self.ball_velocities], axis=1),
            reward=1.0,
            done=done,
        )

    def render(self, mode: str = "human"):
        canvas = np.zeros(
            (self.cfg.simulation_width, self.cfg.simulation_height, 3), dtype="uint8"
        )
        WHITE = (255, 255, 255)
        RED = (0, 0, 255)
        for pos in self.ball_positions[:-1]:
            pos_image_space = _as_opencv_coordinates(pos)
            canvas = cv2.circle(
                canvas, pos_image_space, self.cfg.ball_radius, WHITE, thickness=-1
            )
        canvas = cv2.circle(
            canvas,
            _as_opencv_coordinates(self.ball_positions[-1]),
            self.cfg.ball_radius,
            RED,
            thickness=-1,
        )
        return canvas
