"""
Simple simulation where balls travel in straight
"""

import dataclasses

from ..abstract import GrundEnv


@dataclasses.dataclass
class DodgeConfig:
    canvas_width: int
    canvas_height: int
    starting_num_balls: int


class Dodge(GrundEnv):

    def __init__(self, config: DodgeConfig):
        self.cfg = config

    def step(self, action: np.ndarray):

