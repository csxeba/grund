from typing import Tuple

import numpy as np
import cv2
import gym

from ..abstract import GrundEnv
from .entities import EnemyBall, PlayerBall, Square
from ..util import movement


class ReskivConfig:

    def __init__(self,
                 canvas_shape=(128, 128),
                 frames_per_second: int = 25,
                 initial_number_of_enemies: int = 1,
                 player_radius: int = 4,
                 enemy_radius: int = 2,
                 target_size: int = 3,
                 player_color: Tuple[int, int, int] = (127, 127, 127),
                 enemy_color: Tuple[int, int, int] = (255, 0, 0),
                 target_color: Tuple[int, int, int] = (63, 63, 63),
                 enemy_speed: int = 5,
                 player_speed: int = 7):

        self.canvas_shape = list(canvas_shape)
        self.frames_per_second = frames_per_second
        self.initial_number_of_enemies = initial_number_of_enemies
        self.player_radius = player_radius
        self.enemy_radius = enemy_radius
        self.target_size = target_size
        self.player_color = player_color
        self.enemy_color = enemy_color
        self.target_color = target_color
        self.enemy_speed = enemy_speed
        self.player_speed = player_speed


class REskiv(GrundEnv):

    def __init__(self, config: ReskivConfig):
        super().__init__()
        self.cfg = config

        self.canvas_shape = config.canvas_shape + [3]
        self._canvas = np.zeros(self.canvas_shape, dtype="uint8")  # type: np.ndarray

        self.player = PlayerBall(config.canvas_shape, config.player_color, config.player_radius, config.player_speed)
        self.square = Square(config.canvas_shape, config.target_color, config.target_size)
        self.enemies = []

        self.mean_dist = np.min(np.array(config.canvas_shape)) / 2.

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.canvas_shape, dtype="uint8")
        self.movement_vectors = movement.get_movement_vectors(num_directions=5)

        self.renderer = None

    def spawn_enemy(self):
        enemy = EnemyBall(self.cfg.canvas_shape, self.cfg.enemy_color, self.cfg.enemy_radius, self.cfg.enemy_speed)
        while enemy.distance(self.player) < self.mean_dist:
            enemy.teleport()
        self.enemies.append(enemy)

    def draw(self):
        self._canvas *= 0
        self._canvas = self.player.draw(self._canvas)
        self._canvas = self.square.draw(self._canvas)
        for enemy in self.enemies:
            self._canvas = enemy.draw(self._canvas)

    def reset(self):
        self.player.teleport()
        self.square.teleport()
        self.enemies = []
        for _ in range(self.cfg.initial_number_of_enemies):
            self.spawn_enemy()
        self.draw()
        return self._canvas

    def step(self, action: int):
        done = False
        info = {}
        reward = 0.
        movement_vector = self.movement_vectors[action]
        self.player.move(movement_vector)

        if self.player.touches(self.square):
            reward = 5.
            self.square.teleport()
            self.spawn_enemy()

        for e in self.enemies:
            e.move()
            if self.player.touches(e):
                done = True
                break

        self.draw()
        return self._canvas, reward, done, info

    def render(self, mode: str = "human"):
        if mode == "human":
            cv2.imshow("REskiv", self._canvas)
            cv2.waitKey(1000 // self.cfg.frames_per_second)
        elif mode == "rgb_array":
            return self._canvas
        else:
            raise NotImplementedError
