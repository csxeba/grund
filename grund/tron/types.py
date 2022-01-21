from typing import NamedTuple

import numpy as np


class COLORS:
    RED = (255, 0, 0)
    DARK_RED = (128, 0, 0)
    BLUE = (0, 0, 255)
    DARK_BLUE = (0, 0, 128)
    BLACK = (0, 0, 0)


class SEMANTIC_CLASSES:
    BACKGROUND = 0
    EGO = 1
    ENEMY = 2
    BLOCKED = 3
    NO_SEMANTIC_CLASSES = 4


class ASPECT:
    PLAYER_1 = "player_1"
    PLAYER_2 = "player_2"
    HUMAN = "human"
    MULTIASPECT = "multiaspect"


class Coordinate(NamedTuple):

    x: int
    y: int

    def numpy(self):
        return np.array([self.x, self.y])

    def copy(self):
        return Coordinate(self.x, self.y)

    @property
    def T(self):
        return Coordinate(self.y, self.x)

    def __add__(self, other: "Coordinate"):
        assert isinstance(other, self.__class__)
        x, y = self.x + other.x, self.y + other.y
        return Coordinate(x, y)

    def __eq__(self, other: "Coordinate"):
        assert isinstance(other, self.__class__)
        return self.x == other.x and self.y == other.y


class DIRECTIONS:

    UP = Coordinate(0, -1)
    DOWN = Coordinate(0, 1)
    LEFT = Coordinate(-1, 0)
    RIGHT = Coordinate(1, 0)
    NOOP = Coordinate(0, 0)

    VALID_DIRECTIONS = [UP, DOWN, LEFT, RIGHT, NOOP]

    UP_SELECTOR = 0
    DOWN_SELECTOR = 1
    LEFT_SELECTOR = 2
    RIGHT_SELECTOR = 3
    NOOP_SELECTOR = 4

    SELECTORS = [UP_SELECTOR, DOWN_SELECTOR, LEFT_SELECTOR, RIGHT_SELECTOR, NOOP_SELECTOR]

    OPPOSITES = {UP: DOWN,
                 DOWN: UP,
                 LEFT: RIGHT,
                 RIGHT: LEFT,
                 NOOP: NOOP}


class TRONAction(NamedTuple):

    player_1: int
    player_2: int

    @classmethod
    def create(cls,
               player_1_selector: int = DIRECTIONS.NOOP_SELECTOR,
               player_2_selector: int = DIRECTIONS.NOOP_SELECTOR):

        if player_1_selector not in DIRECTIONS.SELECTORS:
            raise RuntimeError(f"Invalid action selector for player 1: {player_1_selector}")
        if player_2_selector not in DIRECTIONS.SELECTORS:
            raise RuntimeError(f"Invalid action selector for player 2: {player_2_selector}")

        return cls(player_1=player_1_selector, player_2=player_2_selector)


class TRONObservation(NamedTuple):

    player_1: np.ndarray
    player_2: np.ndarray
    render_aspect: np.ndarray


class TRONReward(NamedTuple):

    player_1: float
    player_2: float


class StepResult(NamedTuple):

    observation: TRONObservation
    reward: TRONReward
    done: bool
    info: dict
