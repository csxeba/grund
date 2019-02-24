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
