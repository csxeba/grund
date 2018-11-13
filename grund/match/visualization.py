import numpy as np
import cv2

from .config import MatchConfig


class MatchVisualizer:

    def __init__(self, matchconfig: MatchConfig):
        self.canvas_size = tuple(matchconfig.canvas_size)
        self.cfg = matchconfig

    def draw(self, ball, team1, team2):
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
