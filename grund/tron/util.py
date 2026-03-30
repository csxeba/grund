from typing import Tuple

from .types import Coordinate

NEIGHBORHOOD = [
    Coordinate(0, 1),
    Coordinate(1, 0),
    Coordinate(0, -1),
    Coordinate(-1, 0),
]


def generate_random_starting_positions(canvas_shape: Tuple[int, int], boundary: int):
    x0 = boundary
    x1 = canvas_shape[0] - boundary - 1
    y0 = boundary
    y1 = canvas_shape[1] - boundary - 1
    c1 = Coordinate.random(x0, x1, y0, y1)
    c2 = None
    c1_neighborhood = {c1 + n for n in NEIGHBORHOOD}
    c1_neighborhood.add(c1)
    while c2 is None:
        c = Coordinate.random(x0, x1, y0, y1)
        if c in c1_neighborhood:
            continue
        c2 = c

    return c1, c2
