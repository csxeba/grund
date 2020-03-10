import numpy as np
import cv2


class _EntityBase:

    def __init__(self, canvas_shape, color, size, coords=None):
        self.canvas_shape = canvas_shape
        self._coords = np.array([-50., -50.])
        self.color = color
        self.size = size
        if coords is not None:
            self.teleport(coords)

    @property
    def coords(self):
        return tuple(self._coords.astype(int))

    def adjust_coordinates(self):
        self._coords[self._coords < self.size] = self.size
        toobig = self._coords > (self.canvas_shape - self.size)
        self._coords[toobig] = self.canvas_shape[toobig] - self.size

    def draw(self, screen):
        return cv2.circle(screen, self.coords[::-1], self.size, self.color, -1)

    def move(self, dvec):
        self._coords += dvec.astype(int)
        self.adjust_coordinates()

    def distance(self, other):
        return np.linalg.norm(self._coords - other.coords)

    def touches(self, other):
        return self.distance(other) <= (self.size + other.size)

    def escaping(self):
        r = self.size
        P = self._coords
        return np.any(P <= r) or np.any(P >= self.canvas_shape - r)

    def teleport(self, destination=None):
        if destination is None:
            destination = (np.random.uniform(0.05, 0.95, 2) * self.canvas_shape).astype(int)
        self._coords = destination
        self.adjust_coordinates()


class EnemyBall(_EntityBase):

    def __init__(self, canvas_shape, color, size, speed):
        super().__init__(canvas_shape, color, size)
        self.speed = speed
        self.hori = np.random.uniform() < 0.5
        self._move_generator = self._automove()
        self.move()
        self.teleport()

    def _automove(self):
        d = np.array([self.speed]*2)
        d[int(self.hori)] = 0
        if np.random.uniform() < 0.5:
            d *= -1
        while 1:
            if self.escaping():
                d *= -1
            yield d

    def move(self, dvec=None):
        super().move(next(self._move_generator))


class Square(_EntityBase):

    def __init__(self, game, color, size):
        super().__init__(game, color, size*2)
        self.teleport()

    def draw(self, screen):
        adjust = self.size // 2
        rect_xx = tuple(self._coords - adjust)
        rect_yy = tuple(self._coords + adjust)
        return cv2.rectangle(screen, rect_xx[::-1], rect_yy[::-1], self.color, -1)


class PlayerBall(_EntityBase):

    def __init__(self, canvas_shape, color, size, speed):
        super().__init__(canvas_shape, color, size)
        self.speed = speed
        self.teleport()

    def move(self, dvec):
        super().move(dvec=dvec*self.speed)
