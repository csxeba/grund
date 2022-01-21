import dataclasses
from typing import Tuple, NamedTuple, Set

import numpy as np

from grund import abstract
from grund.util import spaces
from .types import Coordinate, TRONAction, TRONObservation, TRONReward, StepResult
from .types import COLORS, SEMANTIC_CLASSES, DIRECTIONS


@dataclasses.dataclass
class TRONConfig:

    shape_xy: Tuple[int, int] = (96, 96)
    player_1_color: Tuple[int, int, int] = COLORS.RED
    player_1_blocked_color: Tuple[int, int, int] = COLORS.DARK_RED
    player_2_color: Tuple[int, int, int] = COLORS.BLUE
    player_2_blocked_color: Tuple[int, int, int] = COLORS.DARK_BLUE
    background_color: Tuple[int, int, int] = COLORS.BLACK


@dataclasses.dataclass
class TRONPlayer:

    position_xy: Coordinate
    previous_position_xy: Coordinate
    direction_xy: Coordinate
    path: Set[Coordinate]
    observation_aspect: np.ndarray
    reward_aspect: float

    @classmethod
    def create(cls, canvas_shape: Tuple[int, int]):

        observation_aspect = np.full(
            shape=[canvas_shape[0], canvas_shape[1], SEMANTIC_CLASSES.NO_SEMANTIC_CLASSES],
            fill_value=SEMANTIC_CLASSES.BACKGROUND,
            dtype="uint8",
        )
        return cls(
            position_xy=Coordinate(0, 0),
            previous_position_xy=Coordinate(0, 0),
            direction_xy=Coordinate(0, 0),
            path={Coordinate(0, 0)},
            observation_aspect=observation_aspect,
            reward_aspect=0.,
        )

    def reset(self, position: Coordinate, direction: Coordinate):
        self.observation_aspect[0, :] = SEMANTIC_CLASSES.BLOCKED
        self.observation_aspect[-1, :] = SEMANTIC_CLASSES.BLOCKED
        self.observation_aspect[:, 0] = SEMANTIC_CLASSES.BLOCKED
        self.observation_aspect[:, -1] = SEMANTIC_CLASSES.BLOCKED
        self.observation_aspect[1:-1, 1:-1] = SEMANTIC_CLASSES.BACKGROUND

        self.position_xy = position
        self.previous_position_xy = position + DIRECTIONS.OPPOSITES[direction]
        self.direction_xy = direction
        self.path = {self.previous_position_xy.copy()}
        self.reward_aspect = 0.

    def update_aspect(self, enemy: "TRONPlayer"):
        self.observation_aspect[self.previous_position_xy.T] = SEMANTIC_CLASSES.BLOCKED
        self.observation_aspect[enemy.previous_position_xy.T] = SEMANTIC_CLASSES.BLOCKED
        self.observation_aspect[self.position_xy.T] = SEMANTIC_CLASSES.EGO
        self.observation_aspect[enemy.position_xy.T] = SEMANTIC_CLASSES.ENEMY

    def step(self, action: int):

        action_vec = DIRECTIONS.VALID_DIRECTIONS[action]

        if action_vec not in DIRECTIONS.VALID_DIRECTIONS:
            raise RuntimeError(f"Invalid step direction: {action_vec}")
        if action_vec in [self.direction_xy, DIRECTIONS.OPPOSITES[self.direction_xy]]:
            action_vec = DIRECTIONS.NOOP
        if action_vec != DIRECTIONS.NOOP:
            self.direction_xy = action_vec
        self.previous_position_xy = self.position_xy
        self.position_xy += self.direction_xy
        self.path.add(self.previous_position_xy.copy())


class TRONEnvironment(abstract.GrundEnv):

    def __init__(self, config: TRONConfig):
        super().__init__()
        self.cfg = config
        self.size_xy = np.array(config.shape_xy[::-1])
        self.player_1 = TRONPlayer.create(canvas_shape=config.shape_xy)
        self.player_2 = TRONPlayer.create(canvas_shape=config.shape_xy)
        self._render_aspect = np.empty([config.shape_xy[0], config.shape_xy[1], 3], dtype="uint8")
        self.action_space = spaces.DiscreetActionSpace(actions=len(DIRECTIONS.VALID_DIRECTIONS))
        self.observation_space = spaces.ObservationSpace(
            shape=list(config.shape_xy) + [SEMANTIC_CLASSES.NO_SEMANTIC_CLASSES])
        self.done = None

    def _is_player_dead(self, player: TRONPlayer, enemy: TRONPlayer) -> bool:
        coords_np = player.position_xy.numpy()
        dead = np.any(np.logical_or(coords_np < 0, coords_np >= self.size_xy))
        dead = dead or player.position_xy in player.path
        dead = dead or player.position_xy in enemy.path
        return dead

    def _update_aspects(self):
        self._render_aspect[self.player_1.previous_position_xy.T] = self.cfg.player_1_blocked_color
        self._render_aspect[self.player_2.previous_position_xy.T] = self.cfg.player_2_blocked_color
        self._render_aspect[self.player_1.position_xy.T] = self.cfg.player_1_color
        self._render_aspect[self.player_2.position_xy.T] = self.cfg.player_2_color
        self.player_1.update_aspect(enemy=self.player_2)
        self.player_2.update_aspect(enemy=self.player_1)

    def _reset_render_aspect(self):
        self._render_aspect[:] = 0
        self._render_aspect[self.player_1.position_xy.T] = self.cfg.player_1_color
        self._render_aspect[self.player_2.position_xy.T] = self.cfg.player_2_color

    def reset(self) -> TRONObservation:
        self.player_1.reset(position=Coordinate(self.cfg.shape_xy[1] // 3, self.cfg.shape_xy[0] // 2),
                            direction=DIRECTIONS.UP)
        self.player_2.reset(position=Coordinate((self.cfg.shape_xy[1] * 2) // 3, self.cfg.shape_xy[0] // 2),
                            direction=DIRECTIONS.UP)
        self._reset_render_aspect()
        self.done = False
        return TRONObservation(self.player_1.observation_aspect, self.player_2.observation_aspect, self._render_aspect)

    def step(self, action: TRONAction) -> StepResult:

        if not isinstance(action, TRONAction):
            raise RuntimeError(f"Step function must receive an instance of Action. Got: {type(action)}")

        if self.done is not False:
            raise RuntimeError("Environment must be reset")

        self.player_1.step(action.player_1)
        self.player_2.step(action.player_2)

        dead_players = [self._is_player_dead(self.player_1, self.player_2),
                        self._is_player_dead(self.player_2, self.player_1)]

        if all(dead_players) or not any(dead_players):
            self.player_1.reward_aspect = 0.
            self.player_2.reward_aspect = 0.
        elif dead_players[0]:
            self.player_1.reward_aspect = -1.
            self.player_2.reward_aspect = 1.
        elif dead_players[1]:
            self.player_1.reward_aspect = 1.
            self.player_2.reward_aspect = -1.
        else:
            assert False, "All cases should've been handled"

        self._update_aspects()

        observation = TRONObservation(self.player_1.observation_aspect,
                                      self.player_2.observation_aspect,
                                      self._render_aspect)
        reward = TRONReward(self.player_1.reward_aspect, self.player_2.reward_aspect)
        self.done = any(dead_players)
        info = {}

        return StepResult(observation, reward, self.done, info)

    @property
    def canvas(self) -> np.ndarray:
        return self._render_aspect
