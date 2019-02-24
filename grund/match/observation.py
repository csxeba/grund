from typing import List

import numpy as np
import cv2

from .config import MatchConfig, ObservationType, LearningType
from .entities import Ball, Player


class MatchObservationMaker:

    def __init__(self, matchconfig: MatchConfig):
        self.canvas_size = tuple(matchconfig.canvas_size)
        self.cfg = matchconfig
        if self.cfg.observation_type == ObservationType.PIXEL:
            self.observation_shape = self.canvas_size
        elif self.cfg.observation_type == ObservationType.VECTOR:
            self.observation_shape = ((4 + self.cfg.players_per_side * 2 * 2 * 2),)
        else:
            assert False

    def get_pixel_observation(self, ball, team1, team2):
        template = np.zeros(self.canvas_size, dtype="uint8")
        for player in team1:
            template = cv2.circle(template, player.position[::-1], player.radius, 1, 2)
        for player in team2:
            template = cv2.circle(template, player.position[::-1], player.radius, 2, 2)
        perspective_base = np.zeros(self.canvas_size + (3,), dtype="uint8")
        perspective_base = cv2.circle(perspective_base, ball.position[::-1], ball.radius, self.cfg.ball_color, 2)

        canvases = []
        for player in team1:
            perspective = perspective_base.copy()
            perspective[template == 1] = self.cfg.friend_color
            perspective[template == 2] = self.cfg.enemy_color
            perspective = cv2.circle(perspective, player.position[::-1], player.radius // 2,
                                     self.cfg.ego_player_indicator_color)
            canvases.append(perspective)
        for player in team2:
            perspective = perspective_base.copy()
            perspective[template == 2] = self.cfg.friend_color
            perspective[template == 1] = self.cfg.enemy_color
            perspective = cv2.circle(perspective, player.position[::-1], player.radius // 2,
                                     self.cfg.ego_player_indicator_color)
            canvases.append(perspective[:, ::-1, :])
        return np.array(canvases)

    def get_numeric_observation(self, ball: Ball, team1: List[Player], team2: List[Player]):
        """
        Make an observation which consist of:
        - 2 coordinates: ball
        - 2 velocities: ball
        - 2 coordinates: ego
        - 2 velocities: ego
        - N-1_ego_team * 2 coordinates
        - N-1_enemy_team * 2 coordinates
        - N_ego_team * 2 velocities
        - N_enemy_team * 2 velocities
        # 3: ego communication
        # N-1 * 3: team communication
        """
        n_team = len(team1)
        n_players = n_team*2

        canvas_size = np.array(self.canvas_size)

        coord_observation = np.empty((n_players, 2+n_players*2, 2))
        coord_observation[:, 0] = ball.position / canvas_size
        coord_observation[:, 1] = ball.velocity

        ego_team = {1: team1, 2: team2}
        enemy_team = {1: team2, 2: team1}

        for i, player in enumerate(team1 + team2):
            coord_observation[i, 2] = player.position / canvas_size
            coord_observation[i, 3] = player.velocity
            for j, mate in enumerate(p for p in ego_team[player.team] if p.ID != player.ID):
                idx = j * 2
                coord_observation[i, 4+idx] = mate.position / canvas_size
                coord_observation[i, 4+idx+1] = mate.velocity
            for j, enemy in enumerate(enemy_team[player.team]):
                idx = j * 2
                coord_observation[i, 4+n_team+idx] = enemy.position / canvas_size
                coord_observation[i, 4+n_team+idx+1] = enemy.velocity
            if self.cfg.learning_type == LearningType.SINGLE_AGENT:
                break

        if self.cfg.learning_type == LearningType.SINGLE_AGENT:
            return coord_observation[0].ravel()
        elif self.cfg.learning_type == LearningType.MULTI_AGENT:
            return coord_observation.reshape(n_players, -1)
        else:
            assert False
