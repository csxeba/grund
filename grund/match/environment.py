import numpy as np

from .config import MatchConfig, Side
from .entities import Ball, Player
from .operation import handle_player_collision, handle_kick
from .visualization import MatchVisualizer
from ..util.abstract import EnvironmentBase


class Match(EnvironmentBase):

    def __init__(self, config: MatchConfig):
        self.cfg = config
        self.shuffle = config.random_initialization
        self.canvas_size = np.array(config.canvas_size, dtype=int)
        self.visualizer = MatchVisualizer(config)
        self.finished = False
        self.ball = None
        self.players = None
        self._aggregate_entities()

    def _aggregate_entities(self):
        x_th = self.canvas_size[0] / (self.cfg.players_per_side + 1)
        y_th = self.canvas_size[1] / 4
        self.ball = Ball(start_position=self.canvas_size / 2, matchconfig=self.cfg)
        self.players = [Player(start_position=np.array([x_th * i, y_th]),
                               matchconfig=self.cfg,
                               side=Side.FRIEND) for i in range(1, self.cfg.players_per_side + 1)]
        self.players += [Player(start_position=np.array([x_th * i, y_th * 3]),
                                matchconfig=self.cfg,
                                side=Side.ENEMY) for i in range(1, self.cfg.players_per_side + 1)]

    def _get_random_coordinates(self):
        return np.random.uniform(0, self.canvas_size, size=2)

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
        ppside = len(self.players) // 2
        return self.visualizer.draw(self.ball, self.players[:ppside], self.players[ppside:])

    def get_reward_ball_offset(self):
        return ((self.canvas_size[0] - self.ball.position[0]) / self.canvas_size[0]) - 0.5

    def get_reward_score(self):
        return self.ball.goal

    def step(self, controls):
        b_movement = []
        positions = {"ball": self.ball.position}
        for ctrl, player in zip(controls, self.players):
            positions[player.ID] = player.position
            player.step(ctrl)
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
        return self.get_state(), self.get_reward_score(), self.finished

    @property
    def neurons_required(self):
        input_shape = tuple(self.canvas_size) + (3,)
        output_shape = (2,)
        return input_shape, output_shape

    def random_state(self, batch_size):
        shuff = self.shuffle
        self.shuffle = True
        states = []
        rewards = []
        for _ in range(batch_size):
            self.reset()
            while 1:
                self.ball.reset(self._get_random_coordinates())
                for player in self.players:
                    if self.ball.touches(player):
                        break
                else:
                    break
            states.append(self.get_state()[0])
            rewards.append(self.get_reward_ball_offset())
        self.shuffle = shuff
        return np.array(states), np.array(rewards)
