from typing import List

import numpy as np
import cv2

from ..util.abstract import EnvironmentBase


def d(pt1, pt2):
    return np.square(np.sum(np.square(pt1 - pt2)))


class Entity:

    def __init__(self, size: int, color: float, canvas_shape: np.ndarray):
        self.coords = np.empty(2, dtype="float32")
        self.color = color
        self.size = size
        self.limits = canvas_shape

    @property
    def quantized(self):
        pos = np.round(self.coords).astype(int)
        pos = np.clip(pos, [0, 0], self.limits)
        return pos

    def step(self, dxy):
        self.coords += dxy
        self.coords = np.clip(self.coords, [0, 0], self.limits)


class Glide(EnvironmentBase):

    def __init__(self,
                 target_size: int = 3,
                 player_size: int = 3,
                 target_color: float = 1.,
                 player_color: float = 0.5,
                 render_shape: List[int, int] = (24, 24)):

        self.state = np.empty(render_shape, dtype="float32")
        self.state_shape = np.array(render_shape)
        self.player = Entity(player_size, player_color, self.state_shape)
        self.target = Entity(target_size, target_color, self.state_shape)

    def touching(self):
        return d(self.player.coords, self.target.coords) < (self.player.size + self.target.size)

    def render(self):
        self.state[:] = 0.
        self.state = cv2.circle(self.state, self.player.quantized, radius=self.player.size, color=self.player.color)
        self.state = cv2.circle(self.state, self.target.quantized, radius=self.target.size, color=self.target.color)

    def reset(self) -> np.ndarray:
        while 1:
            self.player.coords = np.random.uniform(0, 1, size=2).astype("float32")
            self.target.coords = np.random.uniform(0, 1, size=2).astype("float32")
            if not self.touching():
                break
        self.render()
        return self.state

    def step(self, action: np.ndarray):
        self.player.step(action)
        self.render()
        if self.touching():
            reward = 1.
            done = True
        else:
            reward = 0.
            done = False
        return self.state, reward, done, {}
