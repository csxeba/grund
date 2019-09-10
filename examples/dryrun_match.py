import numpy as np
from grund.match import Match, MatchConfig

cfg = MatchConfig(canvas_size=(128, 128), players_per_side=2, random_initialization=False)
env = Match(cfg)

num_steps = []
rewards = []

for i in range(1, 101):
    env.reset()
    done = False
    steps = 0
    reward = 0
    while not done:
        # print()
        # print("BALL COORD:", state[:2])
        # print("BALL VELOC:", state[2:4])
        # print("EGO  COORD:", state[4:6])
        # print("EGO  VELOC:", state[6:8])
        # print("MATE COORD:", state[8:10])
        # print("MATE VELOC:", state[10:12])
        # print("ENM1 COORD:", state[12:14])
        # print("ENM1 VELOC:", state[14:16])
        # print("ENM2 COORD:", state[16:18])
        # print("ENM2 VELOC:", state[18:20])
        state, reward, done, info = env.step(np.random.randint(0, 5, size=4))
        steps += 1
    rewards.append(reward)
    num_steps.append(steps)
    print("\rProgress {:.0%}".format(i/100), end="")

print()
print("*"*50)
print("STEPS mean:", np.mean(num_steps))
print("STEPS stdv:", np.std(num_steps))
print("STEPS min :", np.min(num_steps))
print("STEPS max :", np.max(num_steps))
print("-"*50)
print("REWDS mean:", np.mean(rewards))
print("REWDS stdv:", np.std(rewards))
print("REWDS min :", np.min(rewards))
print("REWDS max :", np.max(rewards))
