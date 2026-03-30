import dataclasses
from typing import Tuple, Literal, Any

import cv2
import gymnasium as gym
import numpy as np

from grund.abstract import GrundEnv
from grund.util import movement
from grund.reskiv.entities import EnemyBall, PlayerBall, Square


_types = [PlayerBall, Square, EnemyBall]


@dataclasses.dataclass
class ReskivConfig:
    canvas_shape: Tuple[int, int] = (400, 500)
    frames_per_second: int = 25
    initial_number_of_enemies: int = 1
    player_radius: int = 10
    enemy_radius: int = 5
    target_size: int = 10
    player_color: Tuple[int, int, int] = (127, 127, 127)
    enemy_color: Tuple[int, int, int] = (255, 0, 0)
    target_color: Tuple[int, int, int] = (63, 63, 63)
    enemy_speed: int = 5
    player_speed: int = 7
    observation_type: Literal["rgb", "coords"] = "rgb"
    time_limit: int = -1


class Reskiv(GrundEnv):
    def __init__(self, config: ReskivConfig = ReskivConfig()):
        super().__init__()
        self.cfg = config
        self._step_counter = -1
        self.canvas_shape = list(config.canvas_shape) + [3]
        self._canvas = np.zeros(self.canvas_shape, dtype="uint8")  # type: np.ndarray

        self.player = PlayerBall(
            config.canvas_shape,
            config.player_color,
            config.player_radius,
            config.player_speed,
        )
        self.square = Square(
            config.canvas_shape, config.target_color, config.target_size
        )
        self.enemies = []

        self.mean_dist = np.min(np.array(config.canvas_shape)) / 2.0

        self.action_space = gym.spaces.Discrete(5)
        if config.observation_type == "rgb":
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=self.canvas_shape, dtype="uint8"
            )
        else:
            self.observation_space = gym.spaces.Box(low=0, high=3, shape=[3, 3])
        self.movement_vectors = movement.get_movement_vectors(num_directions=5)

        self.renderer = None

    def spawn_enemy(self):
        enemy = EnemyBall(
            self.cfg.canvas_shape,
            self.cfg.enemy_color,
            self.cfg.enemy_radius,
            self.cfg.enemy_speed,
        )
        while enemy.distance(self.player) < self.mean_dist:
            enemy.teleport()
        self.enemies.append(enemy)

    def make_obs_rgb(self):
        self._canvas *= 0
        self._canvas = self.player.draw(self._canvas)
        self._canvas = self.square.draw(self._canvas)
        for enemy in self.enemies:
            self._canvas = enemy.draw(self._canvas)

    def make_obs_coord(self):
        coords_array = np.array([self.player.coords, self.square.coords] + [e.coords for e in self.enemies])
        types_array = np.array([
            _types.index(type(obj)) for obj in [self.player, self.square] + self.enemies
        ])[..., np.newaxis]
        coords_array = coords_array / np.array(self.canvas_shape[:2])
        obs = np.concatenate([coords_array, types_array], axis=-1) # shape: n x 3
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        self.player.teleport()
        self.square.teleport()
        self.enemies = []
        for _ in range(self.cfg.initial_number_of_enemies):
            self.spawn_enemy()
        self._step_counter = -1
        if self.cfg.observation_type == "rgb":
            self.make_obs_rgb()
            return self._canvas, {}
        else:
            return self.make_obs_coord(), {}

    def step(self, action: int):
        term = False
        trunc = False
        info = {"num_enemies": len(self.enemies)}
        reward = -0.1
        self._step_counter += 1
        movement_vector = self.movement_vectors[action]
        self.player.move(movement_vector)

        if self.player.touches(self.square):
            reward += 10.0
            self.square.teleport()
            self.spawn_enemy()

        for e in self.enemies:
            e.move()
            if self.player.touches(e):
                term = True
                reward = -1.0
                break

        if (self.cfg.time_limit > -1) and (self._step_counter >= self.cfg.time_limit):
            trunc = True

        if self.cfg.observation_type == "rgb":
            self.make_obs_rgb()
            obs = self._canvas
        else:
            obs = self.make_obs_coord()
        return obs, reward, term, trunc, info

    def render(self, mode: str = "human"):
        if mode == "human":
            cv2.imshow("REskiv", self._canvas)
            cv2.waitKey(1000 // self.cfg.frames_per_second)
        elif mode == "rgb_array":
            return self._canvas
        else:
            raise NotImplementedError
