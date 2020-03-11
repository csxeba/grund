from typing import List

import gym
import numpy as np
import cv2

from .entities import EnemyBall, PlayerBall, Square
from ..util import movement


class ReskivConfig:

    def __init__(self,
                 canvas_shape=(128, 128),
                 initial_number_of_enemies=1,
                 player_radius=4,
                 enemy_radius=2,
                 target_size=3,
                 player_color=(127, 127, 127),
                 enemy_color=(0, 0, 255),
                 target_color=(63, 63, 63),
                 enemy_speed=5,
                 player_speed=7):

        self.canvas_shape = list(canvas_shape)
        self.initial_number_of_enemies = initial_number_of_enemies
        self.player_radius = player_radius
        self.enemy_radius = enemy_radius
        self.target_size = target_size
        self.player_color = player_color
        self.enemy_color = enemy_color
        self.target_color = target_color
        self.enemy_speed = enemy_speed
        self.player_speed = player_speed


class Reskiv(gym.Env):

    def __init__(self, config: ReskivConfig):
        super().__init__()
        self.cfg = config

        self.canvas_shape = config.canvas_shape + [3]
        self.canvas = np.zeros(self.canvas_shape, dtype="uint8")  # type: np.ndarray

        self.player = PlayerBall(config.canvas_shape, config.player_color, config.player_radius, config.player_speed)
        self.square = Square(config.canvas_shape, config.target_color, config.target_size)
        self.enemies = []

        self.score = 0
        self.mean_dist = np.array(config.canvas_shape).min() / 2.

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.canvas_shape, dtype="uint8")
        self.movement_vectors = movement.get_movement_vectors(num_directions=5)

    def spawn_enemy(self):
        enemy = EnemyBall(self.cfg.canvas_shape, self.cfg.enemy_color, self.cfg.enemy_radius, self.cfg.enemy_speed)
        while enemy.distance(self.player) < self.mean_dist:
            enemy.teleport()
        self.enemies.append(enemy)

    def draw(self):
        self.canvas *= 0
        self.canvas = self.player.draw(self.canvas)
        self.canvas = self.square.draw(self.canvas)
        for enemy in self.enemies:
            self.canvas = enemy.draw(self.canvas)

    def reset(self):
        self.player.teleport()
        self.square.teleport()
        self.enemies = []
        for _ in range(self.cfg.initial_number_of_enemies):
            self.spawn_enemy()
        self.score = 0
        self.draw()
        return self.canvas

    def step(self, action: int):
        done = False
        info = {}
        reward = -0.1
        movement_vector = self.movement_vectors[action]
        self.player.move(movement_vector)

        if self.player.touches(self.square):
            self.score += 1
            reward = 10. * self.score
            self.square.teleport()
            self.spawn_enemy()

        for e in self.enemies:
            e.move()
            if self.player.touches(e):
                done = True
                break

        self.draw()
        return self.canvas, reward, done, info

    def render(self, mode='human'):
        cv2.imshow("Reskiv Canvas", self.canvas)
        return cv2.waitKey(1000 // 25)
