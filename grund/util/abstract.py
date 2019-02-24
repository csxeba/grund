class EnvironmentBase:

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
