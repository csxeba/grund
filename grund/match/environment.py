import numpy as np

from .config import MatchConfig, Side, ObservationType, LearningType
from .entities import Ball, Player
from .operation import handle_player_collision, handle_kick
from .observation import MatchObservationMaker
from ..util.abstract import EnvironmentBase
from ..util.spaces import ObservationSpace, DiscreetActionSpace
from ..util.screen import CV2Screen
from ..util.movement import get_movement_vectors


class Match(EnvironmentBase):

    def __init__(self, config: MatchConfig):
        self.cfg = config
        self.shuffle = config.random_initialization
        self.canvas_size = np.array(config.canvas_size, dtype=int)
        self.observation_factory = MatchObservationMaker(config)
        self.finished = False
        self.ball = None
        self.players = None
        self.observation_space = None
        self.action_space = None
        self.screen = CV2Screen()
        self.movements = get_movement_vectors(5)
        self._next_player_id = -1
        self._create_all_entities()

    @property
    def _player_id(self):
        self._next_player_id += 1
        return self._next_player_id

    def _create_all_entities(self):
        x_th = self.canvas_size[0] / (self.cfg.players_per_side + 1)
        y_th = self.canvas_size[1] / 4
        self.ball = Ball(start_position=self.canvas_size / 2, matchconfig=self.cfg)
        self.players = [Player(start_position=np.array([x_th * i, y_th]),
                               matchconfig=self.cfg,
                               side=Side.FRIEND,
                               ID=self._player_id) for i in range(1, self.cfg.players_per_side + 1)]
        self.players += [Player(start_position=np.array([x_th * i, y_th * 3]),
                                matchconfig=self.cfg,
                                side=Side.ENEMY,
                                ID=self._player_id) for i in range(1, self.cfg.players_per_side + 1)]
        self.observation_space = ObservationSpace(self.observation_factory.observation_shape)
        self.action_space = DiscreetActionSpace(actions=np.arange(len(self.movements)))

    def _get_random_coordinates(self):
        return np.random.uniform(0, self.canvas_size, size=2)

    def _get_random_actions(self):
        actions = np.random.randint(0, self.action_space.n, size=len(self.players))
        return actions

    def _get_no_actions(self):
        actions = np.full(len(self.players), 4, dtype=int)
        return actions

    def _random_reset(self):
        self.players[0].reset(self._get_random_coordinates())
        for i, player in enumerate(self.players[1:]):
            player.reset(self._get_random_coordinates())
            while 1:
                for other in self.players[:i] + [self.ball]:
                    if player.touches(other):
                        player.reset(self._get_random_coordinates())
                        break
                else:
                    break

    def reset(self):
        self.ball.reset(self.ball.start_position)
        self.finished = False
        if self.shuffle:
            self._random_reset()
            return self.get_state()
        for player in self.players:
            player.reset()
        return self.get_state()

    def get_canvas_size(self):
        return tuple(self.canvas_size) + (3,)

    def get_state(self):
        team1, team2 = self.teams
        if self.cfg.observation_type == ObservationType.VECTOR:
            observation = self.observation_factory.get_numeric_observation(self.ball, team1, team2)
        else:
            observation = self.observation_factory.get_pixel_observation(self.ball, team1, team2)
        return observation

    def get_reward_ball_offset(self):
        return ((self.canvas_size[0] - self.ball.position[0]) / self.canvas_size[0]) - 0.5

    def get_reward_score(self):
        return self.ball.goal

    def step(self, actions):
        if self.cfg.learning_type == LearningType.SINGLE_AGENT:
            action_template = self._get_no_actions()
            action_template[0] = actions
            actions = action_template

        b_movement = []
        positions = {"ball": self.ball.position}
        for ctrl, player in zip(actions, self.players):
            positions[player.ID] = player.position
            vector = self.movements[ctrl]
            player.step(vector)
            if player.touches(self.ball):
                kick_vector = handle_kick(player, self.ball)
                b_movement.append(kick_vector)
                player.position = positions[player.ID]

        for i, player1 in enumerate(self.players[:-1], start=1):
            for player2 in self.players[i:]:
                if player1.touches(player2):
                    handle_player_collision(player1, player2)
                    player1.position = positions[player1.ID]
                    player2.position = positions[player2.ID]

        self.ball.step(np.sum(b_movement, axis=0))
        if self.ball.goal and not self.finished:
            self.finished = True
        return self.get_state(), self.get_reward_score(), self.finished, {}

    def render(self):
        team1, team2 = self.teams
        obs = self.observation_factory.get_pixel_observation(self.ball, team1, team2)
        self.screen.blit(obs[0])

    @property
    def teams(self):
        team_split = len(self.players) // 2
        return self.players[:team_split], self.players[team_split:]
