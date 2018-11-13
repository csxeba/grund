import numpy as np

from .config import MatchConfig


ZERO = np.zeros(2)
ZERO.flags["WRITEABLE"] = False


class Entity:

    def __init__(self, start_position, matchconfig: MatchConfig):
        self.fsize = matchconfig.canvas_size
        self.start_position = start_position.copy()
        self._last_position = None
        self._position = start_position
        self.velocity = ZERO.copy()
        self.friction = None
        self.max_velocity = None
        self.radius = None
        print("Created {} @ {}".format(self.__class__.__name__, self._position))

    @property
    def position(self):
        return tuple(self._position.astype(int))

    @position.setter
    def position(self, position):
        self._last_position = self._position
        self._position = np.array(position, dtype=float)

    @property
    def last_position(self):
        return self._last_position

    def step(self, vector=ZERO):
        self.velocity += vector
        self.velocity *= self.friction
        vnorm = np.linalg.norm(self.velocity) + 1e-7
        if vnorm > self.max_velocity:
            self.velocity = self.velocity / vnorm * self.max_velocity
        self._position += self.velocity
        hangover = np.logical_or(self._position < self.radius, self._position >= self.fsize - self.radius)
        self.velocity[hangover] *= -0.25
        self._position = np.clip(self._position, self.radius, self.fsize-self.radius)

    def touches(self, other):
        d = np.linalg.norm(self._position - other._position)
        return d <= self.radius + other.radius

    def reset(self, position=None):
        self._position = position if position is not None else self.start_position
        self.velocity = ZERO.copy()


class Ball(Entity):

    def __init__(self, start_position: np.ndarray, matchconfig: MatchConfig):
        super().__init__(start_position, matchconfig)
        self.goal = 0
        self.friction = matchconfig.ball_friction
        self.max_velocity = matchconfig.ball_max_velocity
        self.radius = matchconfig.ball_pixel_radius

    def reset(self, position=None):
        super().reset(position)
        self.goal = 0

    def step(self, vector=ZERO):
        super().step(vector)
        if self._position[1] <= self.radius:
            self.goal = 1
        elif self._position[1] >= self.fsize[1] - self.radius:
            self.goal = -1
        else:
            return


class Player(Entity):
    next_id = 0

    def __init__(self, start_position, matchconfig: MatchConfig, side):
        super().__init__(start_position, matchconfig)
        self.ID = self.__class__.next_id
        self.team = side
        self.radius = matchconfig.players_pixel_radius
        self.max_velocity = matchconfig.players_max_velocity
        self.friction = matchconfig.players_friction[self.ID]
        self.__class__.next_id += 1
