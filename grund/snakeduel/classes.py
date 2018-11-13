from ..util.movement import get_movement_vectors


class Snake:

    next_color = 1

    def __init__(self, coords):
        self.coords = coords
        self.color = self.next_color
        self.next_color += 1
        self.body = []
        self.direction = 1
        self.vectors = {1: [0, 1], 3: [0, -1],
                        2: [1, 0], 4: [-1, 0],
                        0: [0, 0]}
        self.opposite = {1: 3, 3: 1, 2: 4, 4: 2, 0: None}

    def move(self, direction):
        invalid_direction = any([
            direction == self.direction,
            self.opposite[direction] == self.direction,
            not direction
        ])
        if invalid_direction:
            direction = self.direction
        self.direction = direction
        self.body.append(tuple(self.coords))
        newcoords = self.coords + self.vectors[direction]
        self.coords = newcoords

    @property
    def suicide(self):
        return tuple(self.coords) in self.body


class Food:

    color = -1

    def __init__(self, coords=None):
        self.coords = coords
