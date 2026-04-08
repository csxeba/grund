import numpy as np

ALL_DIRECTIONS = np.array(
    [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
)
NW, N, NE, W, NOOP, E, SW, S, SE = ALL_DIRECTIONS


def get_movement_vectors(num_directions: int):
    mapping = {
        9: ALL_DIRECTIONS,
        8: np.stack([NW, N, NE, W, E, SW, S, SE]),
        5: np.stack([W, N, E, S, NOOP]),
        4: np.stack([W, N, E, S]),
    }
    if num_directions not in mapping:
        raise ValueError("Can only handle 9, 8, 5 or 4 directions!")
    return mapping[num_directions]


class MovementTranslator:

    def __init__(self, continuous: bool):
        if continuous:
            self.mapping = {
                -1: NOOP,
                83: E,
                81: W,
                82: N,
                84: S,
                27: None,
                119: N,
                115: S,
                97: W,
                100: E,
            }
        else:
            self.mapping = {
                -1: 4,
                83: 2,
                81: 0,
                82: 1,
                84: 3,
                27: None,
                119: 1,
                115: 3,
                97: 0,
                100: 2,
            }


    def translate(self, opencv_keypress):
        return self.mapping[opencv_keypress]
