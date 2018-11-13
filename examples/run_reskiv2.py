import numpy as np

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Input, Conv2D, Activation, PReLU
from keras import backend as K, optimizers

from grund.reskiv2 import Reskiv, ReskivConfig
from grund.util.movement import get_movement_vectors


class ReplayMemory:

    def __init__(self):
        self.states = np.zeros([0, 128, 128, 3])
        self.action_onehots = np.zeros([0, 5])
        self.discounted_rewards = np.zeros([0])

    def remember(self, states, onehot, rewards):
        self.states = np.concatenate([self.states, states])
        self.action_onehots = np.concatenate([self.action_onehots, onehot])
        self.discounted_rewards = np.concatenate([self.discounted_rewards, rewards])
        assert len(self.states) == len(self.action_onehots) == len(self.discounted_rewards)
        if len(states) > 40000:
            self.states = self.states[-40000:]
            self.action_onehots = self.action_onehots[-40000:]
            self.discounted_rewards = self.discounted_rewards[-40000:]

    def current_num_updates(self, batch_size):
        return len(self.states) // batch_size

    def batch_generator(self, batch_size=32, infinite=False):
        while 1:
            iter_start = np.random.randint(0, batch_size)
            arg = np.arange(iter_start, len(self.states) - batch_size - 1, batch_size)
            np.random.shuffle(arg)
            for i, start in enumerate(arg):
                end = start + batch_size
                states = self.states[start:end]
                action_onehots = self.action_onehots[start:end]
                discounted_rewards = self.discounted_rewards[start:end]
                yield states, action_onehots, discounted_rewards
            if not infinite:
                break


class PolicyGradient:

    def __init__(self, model: Model, discount_factor: float, replay_memory: ReplayMemory,
                 num_actions: int):
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.model = model
        self.actions = np.arange(self.num_actions)
        self.action_labels = np.eye(self.num_actions)
        self.states = []
        self.action_onehots = []
        self.rewards = []
        self.replay = replay_memory
        self.train_function = self._custom_train_function()

    def _custom_train_function(self):
        softmaxes = self.model.output
        action_onehots = K.placeholder((None, self.model.output_shape[0]), name="action_onehot")
        discounted_rewards = K.placeholder((None,), name="discounted_rewards")
        action_probabilities = K.sum(softmaxes * action_onehots, axis=1)
        action_log_probs = -K.log(action_probabilities)

        loss = K.sum(action_log_probs * discounted_rewards)

        updates = self.model.optimizer.get_updates(loss, self.model.trainable_weights)
        return K.function(inputs=[self.model.input,
                                  action_onehots,
                                  discounted_rewards],
                          outputs=[loss],
                          updates=updates)

    def reset(self):
        self.states = []
        self.action_onehots = []
        self.rewards = []

    def sample(self, state, current_reward):
        self.states.append(state)
        self.rewards.append(current_reward)
        softmax = self.model.predict(self.preprocess(state)[None, ...])[0]
        action = np.random.choice(self.actions, p=softmax)
        self.action_onehots.append(self.action_labels[action])
        return action

    def discount_rewards2(self, R):
        discounted_r = np.zeros_like(R)
        running_add = R[-1]
        for t, r in enumerate(R[::-1]):
            running_add += self.discount_factor * r
            discounted_r[t] = running_add
        discounted_r[0] = R[-1]
        discounted_r = discounted_r[::-1]
        return discounted_r

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    @staticmethod
    def calculate_advantage(discounted_rewards, onehots):
        advantage = onehots * discounted_rewards
        return advantage

    def accumulate(self, final_reward):
        R = np.array(self.rewards[1:] + [final_reward])
        if self.discount_factor > 0.:
            R = self.discount_rewards(R)
        S = np.array(self.states)
        onehots = np.array(self.action_onehots)
        self.replay.remember(S, onehots, R)
        self.reset()

    def train_model(self, batch_site=32, epochs=1):
        num_updates = self.replay.current_num_updates(batch_site)
        for e in range(1, epochs+1):
            print("Epoch {}/{}".format(epochs, e))
            batch_stream = self.replay.batch_generator(batch_site)
            losses = []
            for i, (states, action_onehots, discounted_rewards) in enumerate(batch_stream, start=1):
                states = self.preprocess(states)
                # loss = self.train_function([states, action_onehots, discounted_rewards])
                loss = self.model.train_on_batch(states, action_onehots*discounted_rewards[..., None])
                losses.append(loss)
                print("\rTraining {:>7.2%} Batch {}/{} loss {:.4f}"
                      .format(i/num_updates, num_updates, i, np.mean(losses)), end="")
                break
            print()

    @staticmethod
    def preprocess(canvas):
        return canvas / 255.


def build_net(inshape, num_movements):
    model = Sequential([Flatten(input_shape=inshape),
                        Dense(64, activation="relu"),
                        Dense(32, activation="relu"),
                        Dense(num_movements, activation="softmax")])
    model.compile(optimizer=optimizers.Adam(1e-4), loss="categorical_crossentropy")
    return model


def build_cnn(inshape, num_movements):
    inputs = Input(inshape)  # 128
    x = Conv2D(16, (3, 3), strides=2, padding="same")(inputs)  # 64
    x = PReLU()(x)
    x = Conv2D(16, (3, 3), strides=2, padding="same")(x)  # 32
    x = PReLU()(x)
    x = Conv2D(32, (3, 3), strides=2, padding="same")(x)  # 16
    x = PReLU()(x)
    x = Conv2D(32, (3, 3), strides=2, padding="same")(x)  # 8
    x = PReLU()(x)
    x = Conv2D(64, (3, 3), strides=2, padding="same")(x)  # 4
    x = PReLU()(x)
    x = Conv2D(num_movements, (4, 4), strides=1, padding="valid")(x)
    x = Flatten()(x)
    x = Activation("softmax")(x)
    model = Model(inputs, x)
    model.compile("adam", "categorical_crossentropy")
    return model


def main():
    DIRECTIONS = 5
    MAX_STEP = 1000

    movements = get_movement_vectors(DIRECTIONS)

    cfg = ReskivConfig(canvas_shape=[128, 128, 3])
    env = Reskiv(cfg)

    inp, out = env.neurons_required
    ann = build_cnn(inp, DIRECTIONS)
    target = build_cnn(inp, DIRECTIONS)
    target.set_weights(ann.get_weights())

    replay_memory = ReplayMemory()
    agent = PolicyGradient(ann, num_actions=DIRECTIONS, discount_factor=0.99, replay_memory=replay_memory)
    reward = 0.
    episode = 1
    while 1:
        total_reward = 0
        canvas = env.reset()
        print("\nEPISODE", episode)

        for step in range(1, MAX_STEP+1):
            print("\rStep {}/{}".format(MAX_STEP, step), end="")
            action = agent.sample(canvas, reward)
            controls = movements[action]
            canvas, reward, done, info = env.step(controls)
            total_reward += reward
            if done:
                break
        print()
        print("TOTAL REWARD: {:.2f}".format(total_reward))
        agent.accumulate(reward)
        agent.train_model(batch_site=480, epochs=1)
        episode += 1

        # if episode % 10 == 0:
        #     print("Transferring weights to target network!")
        #     updated_weights = []
        #     for W_actor, W_target in zip(ann.get_weights(), target.get_weights()):
        #         updated_weights.append(W_actor * 0.1 + W_target * 0.9)
        #     ann.set_weights(updated_weights)
        #     target.set_weights(updated_weights)
        #     target.save("reskiv_target_model.h5")


if __name__ == '__main__':
    main()
