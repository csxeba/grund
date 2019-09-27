import numpy as np


class Color:
    WHITE = (255, 255, 255)
    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)


class Side:
    FRIEND = 1
    ENEMY = 2


class ObservationType:
    PIXEL = "pixel"
    VECTOR = "vector"


class LearningType:
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"


class ActionSpaceType:
    DISCREETE = "discreete"
    CONTINUOUS = "continuous"


class MatchConfig:

    OBSERVATION_TYPE_PIXEL = "pixel"
    OBSERVATION_TYPE_VECTOR = "vector"
    LEARNING_TYPE_SINGLE_AGENT = "single_agent"
    LEARNING_TYPE_MULTI_AGENT = "multi_agent"

    def __init__(self, canvas_size, players_per_side,
                 observation_type=ObservationType.VECTOR,
                 learning_type=LearningType.MULTI_AGENT,
                 action_space_type=ActionSpaceType.DISCREETE,
                 random_initialization=True,
                 frameskip=None,
                 ball_pixel_radius: int=5,
                 ball_max_velocity: float=20.,
                 ball_friction: float=0.9,
                 ball_color_rgb=Color.WHITE,
                 players_pixel_radius: int=8,
                 players_max_velocity: float=5,
                 players_friction: float=0.9,
                 enemy_color_rgb=Color.YELLOW,
                 friend_color_rgb=Color.CYAN,
                 ego_player_indicator_color_rgb=Color.MAGENTA,
                 communication_allowed=False
                 ):

        self.canvas_size = canvas_size
        self.observation_type = observation_type
        self.learning_type = learning_type
        self.action_space_type = action_space_type
        self.players_per_side = players_per_side
        self.random_initialization = random_initialization
        self.frameskip = frameskip
        self.ball_pixel_radius = ball_pixel_radius
        self.players_pixel_radius = players_pixel_radius
        self.ball_max_velocity = ball_max_velocity
        self.players_max_velocity = players_max_velocity
        self.ball_friction = ball_friction
        self.players_friction = players_friction
        self.ball_color = ball_color_rgb
        self.enemy_color = enemy_color_rgb
        self.friend_color = friend_color_rgb
        self.ego_player_indicator_color = ego_player_indicator_color_rgb
        self.communication_allowed = communication_allowed  # TODO: code this
        self._sanitize()

    def _sanitize(self):
        self.canvas_size = np.array(self.canvas_size[::-1])
        if not isinstance(self.players_per_side, int):
            raise RuntimeError("players_per_side must be an integer (for now) :)")
        if any(map(lambda c: len(c) != 3, [
            self.ball_color, self.enemy_color, self.friend_color, self.ego_player_indicator_color
        ])):
            raise RuntimeError("Color parameters must be a 3-tuple (RGB)")
        else:
            self.ball_color, self.enemy_color, self.friend_color, self.ego_player_indicator_color = map(
                tuple, [self.ball_color, self.enemy_color, self.friend_color, self.ego_player_indicator_color]
            )
        if type(self.players_friction) in [int, float]:
            self.players_friction = [self.players_friction for _ in range(self.players_per_side*2)]
        if self.communication_allowed:
            print("Communication is not yet implemented :(")
        if self.frameskip is None or self.frameskip < 1:
            self.frameskip = 1
