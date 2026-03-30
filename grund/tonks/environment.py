import dataclasses
from typing import List, Optional

import numpy as np

from ..abstract import GrundEnv
from ..util import operation


@dataclasses.dataclass
class TonksConfig:
    map_width: int = 640
    map_height: int = 640
    num_tonks: int = 2
    tonks_width: int = 16
    tonks_height: int = 8
    tonks_mass: float = 1.0
    gravity: float = 1.0
    delta_time: float = 1.0


@dataclasses.dataclass
class Particle:
    type_id: int
    position: np.ndarray


class TonksEnvironment(GrundEnv):
    def __init__(self, config: TonksConfig) -> None:
        super().__init__()
        self.config = config
        self.tonks_static_info = np.zeros(config.num_tonks, 5)  # x0,y0,w,h,theta
        self.tonks_dynamic_info = np.zeros(config.num_tonks, 4)  # vx,vy,m,g
        self.tonks_dynamic_info[:, 2] = config.tonks_mass
        self.tonks_dynamic_info[:, 3] = config.gravity
        self.particles: List[np.ndarray] = []
        self.gridmap = np.zeros((config.map_height, config.map_width), dtype="uint8")

    def _add_tonk(self, position_x0y0: Optional[np.ndarray] = None) -> None:
        TRIES = 100
        tonks_wh = np.array([self.config.tonks_width, self.config.tonks_height])
        all_tonks_x0y0x1y1 = np.concatenate(
            [
                self.tonks_static_info[:, :2],
                self.tonks_static_info[:, :2] + self.tonks_static_info[:, 2:4],
            ],
            axis=1,
        )
        if position_x0y0 is not None:
            tonk_x0y0x1y1 = np.concatenate(
                [position_x0y0, position_x0y0 + tonks_wh], axis=0
            )
            if any(operation.boxes_overlap(tonk_x0y0x1y1, all_tonks_x0y0x1y1)):
                raise RuntimeError(
                    f"Cannot add tonk to {position_x0y0}. Reason is collision position"
                )
        max_w = self.config.map_width - self.config.tonks_width - 1
        max_h = self.config.map_height - self.config.tonks_height - 1
        for trial_no in range(TRIES):
            position_x0y0 = np.random.randint([0, 0], [max_w, max_h], size=1)
            tonk_x0y0x1y1 = np.concatenate(
                [position_x0y0, position_x0y0 + tonks_wh], axis=0
            )
            if any(operation.boxes_overlap(tonk_x0y0x1y1, all_tonks_x0y0x1y1)):
                continue
            else:
                break
        else:
            raise RuntimeError(f"Couldn't generate new tonk after {TRIES} tries.")

    def step(self, actions: np.ndarray) -> None:
        """
        :param actions: np.ndarray[float], shape: [n,

        :return:
        """
        # acceleration_xy = acceleration_xy.copy()
        # Update acceleration y with gravity (m * g)
        # acceleration_xy[:, 1] += self.tonks_dynamic_info[:, 4] * self.tonks_dynamic_info[:, 5]
        # Update velocity with acceleration
        # self.tonks_dynamic_info[:, 0:2] += acceleration_xy * self.config.delta_time  # Update velocity
        # Update position with velocity
        # new_box_x0y0 = self.tonks_static_info[:, 0:2] + self.tonks_dynamic_info[:, 0:2] * self.config.delta_time
        # self.tonks_static_info[:, 0:2] += self.tonks_dynamic_info[:, 0:2]

        # Perform boundary check
        # new_box_x1y1 = new_box_x0y0 + self.tonks_static_info[:, 2:4]
        # out_of_bounds = np.logical_or(new_box_x1y1 < 0, new_box_x0y0 >= self.gridmap.shape)
        # assert len(out_of_bounds) == len(new_box_x0y0)

        # if any(out_of_bounds):
        #     reflect_velocity = np.where(out_of_bounds, 1.0, -0.75)
        #     self.tonks_dynamic_info[:, 2:4] *= reflect_velocity
        # else:
        #     self.tonks_static_info[:, :2] = new_box_x0y0

    def render(self, mode: str = "human"): ...
