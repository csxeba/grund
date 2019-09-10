import numpy as np


def sum_normalize(vector):
    return vector / vector.sum()


def decorrelate(matrix):
    u, s, v = np.linalg.svd(matrix)
    m = v @ matrix
    assert np.all(np.cov(m) == np.eye(len(matrix)))
    return m


class CellMass:

    species_counter = 0

    def __init__(self, mass, transformation, species):
        self.mass = mass
        self.transformation = transformation
        self.species = species

    @classmethod
    def create_species(cls, state_size, mass=None, transformation=None):
        if transformation is None:
            transformation = decorrelate(np.random.randn(state_size, state_size))
            transformation[np.diag_indices_from(transformation)] = 1
        if mass is None:
            mass = np.random.uniform(0, 0.5)
        cls.species_counter += 1
        return cls(mass, transformation, cls.species_counter)

    def step(self, environment_state):
        local_state = environment_state * self.mass
        products = local_state @ self.transformation
        environment_state += products
        return environment_state


class Environment:

    def __init__(self, num_components):
        self.num_components = num_components
        self.state = sum_normalize(np.random.uniform(0, 1, size=num_components))
        self.cell = CellMass.create_species(num_components, mass=0.1)

    def step(self, gauge, delta_t):
        drain = self.state * delta_t
        self.state -= drain
        self.state = self.cell.step(self.state)
        self.state += gauge * delta_t


def simulate():
    from matplotlib import pyplot as plt

    NUM_COMPONENTS = 6
    STEPS = 1000

    env = Environment(num_components=NUM_COMPONENTS)
    gauge1 = sum_normalize(np.random.uniform(0, 1, size=NUM_COMPONENTS))
    gauge2 = sum_normalize(np.random.uniform(0, 1, size=NUM_COMPONENTS))

    volume = []

    for step in range(STEPS):
        gauge = gauge1 if step < 500 else gauge2
        env.step(gauge, delta_t=0.1)
        volume.append(np.linalg.norm(env.state))

    x = np.arange(1, STEPS+1)
    plt.plot(x, volume)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    simulate()
