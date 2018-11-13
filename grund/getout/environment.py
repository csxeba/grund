import numpy as np

from ..util.abstract import EnvironmentBase
from ..util.movement import get_movement_vectors


class Entity:

    def __init__(self, coords, color=1):
        self.coords = coords  # type: np.ndarray
        self.color = color

    def move(self, vector):
        self.coords += np.array(vector)

    def touches(self, other):
        return np.all(self.coords == other.coords)


class GetOut(EnvironmentBase):

    def __init__(self, size):
        self.exit = None  # type: Entity
        self.player = None  # type: Entity
        self.size = np.array(size)
        self.canvas = np.zeros(size)
        self.actions = get_movement_vectors(9)
        self.steps = 0

    @property
    def neurons_required(self):
        return tuple(self.size), len(self.actions)

    def escaping(self):
        return np.any(self.player.coords < 0) or \
               np.any(self.player.coords > self.size-1)

    def draw(self):
        self.canvas = np.zeros(self.size)
        self.canvas[tuple(self.player.coords)] = self.player.color
        self.canvas[tuple(self.exit.coords)] = self.exit.color

    def reset(self):
        self.steps = 0

        xcoords = (np.random.uniform(size=2) * self.size).astype(int)
        rndax = int(np.random.uniform() < 0.5)
        xcoords[rndax] = np.random.choice([0, (self.size-1)[rndax]])
        pcoords = self.size // 2

        self.exit = Entity(xcoords, color=-1)
        self.player = Entity(pcoords, color=1)

        self.draw()
        return self.canvas

    def step(self, action):
        self.steps += 1
        self.player.move(self.actions[action])
        reward = 0.
        done = False
        esc = self.escaping()
        if esc:
            reward = -1.
            done = True
        if self.player.touches(self.exit):
            reward = +1.
            done = True
        if self.steps > np.prod(self.size) * 2:
            reward = 0.
            done = True
        if not esc:
            self.draw()
        return self.canvas, reward, done
