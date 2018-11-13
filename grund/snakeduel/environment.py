import numpy as np

from ..util.abstract import EnvironmentBase
from .classes import Snake, Food


class SnakeEnv(EnvironmentBase):

    def __init__(self, size=(400, 300)):
        self.size = np.array(size)
        self.canvas = np.zeros(size, dtype=int)
        self.snakes = []
        self.food = None
        self.actions = (0, 1, 2, 3)

    def _construct_new_food(self):

        coords = (np.random.uniform(size=2) * self.size).astype(int)
        while np.any(self.snakes[0].body == coords):
            coords = (np.random.uniform(size=2) * self.size).astype(int)
        self.food = Food(coords)

    def escaping(self, snake: Snake):
        return np.any(snake.coords < 0) or np.any(snake.coords >= self.size)

    def draw(self):
        self.canvas[tuple(self.food.coords)] = self.food.color
        for snake in self.snakes:
            self.canvas[tuple(snake.coords)] = snake.color

    def step(self, action):
        s = self.snakes[0]  # type: Snake
        s.move(action)
        if self.escaping(s) or s.suicide:
            return self.canvas, 0., 1
        if np.all(self.snakes[0].coords == self.food.coords):
            return self.canvas, 1., 1
        self.draw()
        return self.canvas, 0., 0

    def reset(self):
        mid = self.size // 2
        coords1 = mid.copy()
        coords1[0] //= 2
        coords2 = mid.copy()
        coords2[0] += mid[0] // 2
        self.canvas = np.zeros(self.size)
        self.snakes = [Snake(coords=coords1)]
        self._construct_new_food()
        self.draw()
        return self.canvas

    @property
    def neurons_required(self):
        return self.size, len(self.actions)
