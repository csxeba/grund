import cv2


class CV2Screen:

    def __init__(self, fps=30, scale=1, interpolation="INTER_NEAREST"):
        self.scale = scale
        self.fps = fps
        self.interpolation = getattr(cv2, interpolation)

    def blit(self, canvas):
        if self.scale != 1:
            canvas = cv2.resize(canvas, tuple(s*self.scale for s in canvas.shape[:2])[::-1],
                                interpolation=self.interpolation)
        cv2.imshow("CV2Screen", canvas[..., ::-1])
        key = cv2.waitKey(1000 // self.fps)
        return key
