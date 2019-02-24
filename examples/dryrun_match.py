import numpy as np
from grund.match import Match, MatchConfig

cfg = MatchConfig(canvas_size=(150, 150), players_per_side=2)
env = Match(cfg)

while 1:
    state = env.reset()
    done = False
    while not done:
        state = state[0]
        print()
        print("BALL COORD:", state[:2])
        print("BALL VELOC:", state[2:4])
        print("EGO  COORD:", state[4:6])
        print("EGO  VELOC:", state[6:8])
        print("MATE COORD:", state[8:10])
        print("MATE VELOC:", state[10:12])
        print("ENM1 COORD:", state[12:14])
        print("ENM1 VELOC:", state[14:16])
        print("ENM2 COORD:", state[16:18])
        print("ENM2 VELOC:", state[18:20])
        state, reward, done, info = env.step(np.random.randint(0, 5, size=4))
        env.render()
