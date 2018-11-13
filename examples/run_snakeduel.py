import numpy as np

from brainforge import BackpropNetwork
from brainforge.layers import DenseLayer, Flatten
from brainforge.reinforcement import PG

from grund.util.screen import CV2Screen


def simulation(game, render=False):
    inshape, outshape = game.neurons_required
    ann = BackpropNetwork(input_shape=np.prod(inshape), layerstack=[
        Flatten(),
        DenseLayer(180, activation="tanh"),
        DenseLayer(outshape, activation="softmax")
    ], cost="cxent", optimizer="rmsprop")
    agent = PG(ann, outshape)
    screen = CV2Screen(scale=20)

    episode = 1

    while 1:
        print()
        print(f"Episode {episode}")
        canvas = game.reset()
        if render:
            screen.blit(canvas)
        step = 0
        done = 0
        reward = None
        while not done:
            action = agent.sample(canvas, reward)
            canvas, reward, done = game.step(action)
            if render:
                screen.blit(canvas)
            step += 1
            # print(f"\rStep: {step}", end="")
        cost = agent.accumulate(canvas, reward)
        print(f" Steps taken: {step:>4}, {'died' if reward == 0 else 'ate!'}, cost = {cost:.4f}")
        if episode % 10 == 0:
            print("Updating!")
        if episode % 1000 == 0:
            render = True
        episode += 1


if __name__ == '__main__':
    from grund import snakeduel
    environment = snakeduel.SnakeEnv((10, 10))
    simulation(environment, render=False)
    print()
