import numpy as np

from sparcity.dev import typechecker, valchecker


class Coord:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ):
        typechecker(x, np.ndarray, "x")
        typechecker(y, np.ndarray, "y")
        typechecker(z, np.ndarray, "z")
        valchecker(len(x.shape) == 1)
        valchecker(len(y.shape) == 1)
        valchecker(len(z.shape) == 2)
        valchecker(x.size == z.shape[1])
        valchecker(y.size == z.shape[0])
        self.x = x
        self.y = y
        self.z = z
        self.x_2d, self.y_2d = np.meshgrid(x, y)
        self.shape = z.shape

    def __call__(self) -> dict:
        return dict(X=self.x_2d, Y=self.y_2d, Z=self.z)
