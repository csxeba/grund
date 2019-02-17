from typing import List

import numpy as np

from .entities import EnemyBall, PlayerBall, Square
from ..util.abstract import EnvironmentBase


class ReskivConfig:

    def __init__(self,
                 canvas_shape=(128, 128, 3),
                 initial_number_of_enemies=1,
                 player_radius=4,
                 enemy_radius=2,
                 target_size=3,
                 player_color=(127, 127, 127),
                 enemy_color=(0, 0, 255),
                 target_color=(63, 63, 63),
                 enemy_speed=5,
                 player_speed=7):

        self.canvas_shape = canvas_shape
        self.initial_number_of_enemies = initial_number_of_enemies
        self.player_radius = player_radius
        self.enemy_radius = enemy_radius
        self.target_size = target_size
        self.player_color = player_color
        self.enemy_color = enemy_color
        self.target_color = target_color
        self.enemy_speed = enemy_speed
        self.player_speed = player_speed


class Reskiv(EnvironmentBase):

    def __init__(self, config: ReskivConfig):
        super().__init__()
        self.cfg = config
        self.canvas = np.zeros(config.canvas_shape, dtype="uint8")  # type: np.ndarray
        self.player = None  # type: PlayerBall
        self.square = None  # type: Square
        self.enemies = None  # type: List[EnemyBall]
        self.score = False
        self.mean_dist = np.array(self.cfg.canvas_shape).min() / 2.

    def spawn_enemy(self):
        enemy = EnemyBall(np.array(self.cfg.canvas_shape[:2]).astype(float),
                          self.cfg.enemy_color, self.cfg.enemy_radius, self.cfg.enemy_speed)
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
        cshape = np.array(self.cfg.canvas_shape[:2])
        self.player = PlayerBall(cshape, self.cfg.player_color, self.cfg.player_radius, self.cfg.player_speed)
        self.square = Square(cshape, self.cfg.target_color, self.cfg.target_size)
        self.enemies = []
        for _ in range(self.cfg.initial_number_of_enemies):
            self.spawn_enemy()
        self.score = 0
        self.draw()
        return self.canvas

    def step(self, action):
        done = False
        reward = 0.
        info = {}
        self.player.move(action)

        if self.player.touches(self.square):
            self.score += 1
            reward = 1.
            self.square.teleport()
            self.spawn_enemy()

        for e in self.enemies:
            e.move()
            if self.player.touches(e):
                done = True
                break

        self.draw()
        return self.canvas, reward, done, info

    @property
    def neurons_required(self):
        return self.state_shape, self.action_space_shape

    @property
    def state_shape(self):
        return self.canvas.shape

    @property
    def action_space_shape(self):
        return 2,
