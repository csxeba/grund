import numpy as np

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Input, Add, LeakyReLU
from keras.utils import plot_model

from brainforge.reinforcement import DQN, AgentConfig
from brainforge.reinforcement.experience import Experience

from grund.match import Match, MatchConfig
from grund.util.movement import get_movement_vectors


PLAYERS_PER_SIDE = 2
NUM_MOVEMENT_DIRECTIONS = 5

movement_vectors = get_movement_vectors(NUM_MOVEMENT_DIRECTIONS)

# Instantiate the environment
env_cfg = MatchConfig(
    canvas_size=[240, 120],
    players_per_side=2,
    random_initialization=True,
    players_friction=0.9,
    ball_friction=0.95
)
env = Match(env_cfg)

# Build a neural RL agent
ann_input_shape, ann_output_shape = env.neurons_required

ann_input = Input(ann_input_shape)
x = Conv2D(16, (5, 5), strides=(2, 2))
x = Dense(ann_output_shape[0], activation="softmax")(x)
ann = Model(ann_input, x)
ann.compile("adam", "categorical_crossentropy")
plot_model(ann)

agent_cfg = AgentConfig(
    batch_size=32,
    replay_memory=Experience(limit=10000, mode="drop", downsample=4)
)
agent = DQN(
    network=ann,
    num_actions=NUM_MOVEMENT_DIRECTIONS,
    agentconfig=agent_cfg
)

# Matrix holding the actions of every player
controls = np.zeros((env_cfg.players_per_side * 2, 2))

running_absolute_reward = None
episodes = 1
loss = 0.

while 1:
    print("Episode", episodes, "loss: {:.4f}".format(loss))
    canvas = env.reset()
    rewards = np.zeros(PLAYERS_PER_SIDE*2)
    done = False
    step = 0

    for step in range(1, 401):
        canvases, rewards, done = env.step(controls)
        actions = agent.sample_multiple(canvas, rewards)
        controls = movement_vectors[actions]
        absolute_reward = np.mean(np.sum(rewards))
        if running_absolute_reward is None:
            running_absolute_reward = absolute_reward
        else:
            running_absolute_reward *= 0.99
            running_absolute_reward += absolute_reward * 0.01
        print("\rStep {:>3} {:.4f}".format(step, running_absolute_reward))

    agent.accumulate_multiple(canvas, rewards)
    loss = agent.learn_batch(batch_size=1024)

