from typing import Optional, Tuple

import numpy as np


class ObservationSpace:
    def __init__(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape


class DiscreetActionSpace:
    def __init__(self, actions):
        if isinstance(actions, int):
            actions = np.arange(actions)
        self.n = len(actions)
        self.actions = actions
        self.action_indices = np.arange(self.n)

    def action_onehot(self, action):
        result = np.zeros(self.n)
        result[np.squeeze(np.argwhere(self.actions == action))] = 1
        return result

    def sample(self):
        return np.random.choice(self.actions)


class ContinuousActionSpace:
    def __init__(
        self,
        shape: Tuple[int, ...],
        minima: np.ndarray,
        maxima: np.ndarray,
    ):
        self.shape = shape
        self.minima = minima
        self.maxima = maxima

    def sample(self, n: Optional[int] = None) -> np.ndarray:
        if n is None:
            return np.random.uniform(low=self.minima, high=self.maxima, size=self.shape)
        else:
            return np.random.uniform(
                low=self.minima, high=self.maxima, size=(n,) + self.shape
            )
