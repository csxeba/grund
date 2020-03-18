import numpy as np
import gym


class GrundEnv(gym.Env):

    canvas_error = NotImplementedError(
        "Please implement the .canvas property or set the _canvas attribute with an array!")

    def __init__(self):
        self._canvas = None
        self.renderer = None

    @property
    def canvas(self) -> np.ndarray:
        if self._canvas is None:
            raise self.canvas_error
        if not isinstance(self._canvas, np.ndarray):
            raise self.canvas_error
        if self._canvas.ndim not in (2, 3):
            raise self.canvas_error
        return self._canvas

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode: str = "human"):
        return self.canvas
