"""
Abstract class for generator
"""
from abc import abstractmethod

import numpy as np

from ._coord import Coord


class Generator(Coord):
    """
    Abstract class for generator

    Methods
    -------
    __init__(self, **kwargs) -> None:
        initialize attributes listed below.

    __call__() -> Dict[str, np.ndarray]:
        returns {'X': self.x_2d, 'Y': self.y_2d, 'Z': self.z}
        The return value is designed for surfaceplot of matplotlib.

    Attributes
    ----------
    x: np.ndarray
        x coordinates (longitude) for the designated area.
        Given by `xrange` argument.

    y: np.ndarray
        y coordinates (latitude) for the designated area.
        Given by `yrange` argument.

    z: np.ndarray
        z values (e.g., altitude) for the designated area.
        Generated from given functions.

    x_2d: np.ndarray
        x coordinates (longitude) corresponding to z

    y_2d: np.ndarray
        y coordinates (latitude) corresponding to z

    shape: List[int]
        shape of z
    """
    @abstractmethod
    def __init__(self, xrange, yrange, **kwargs) -> None:
        """
        Parameters
        ----------
        xrange: np.ndarray
            x-values to calculate

        yrange: np.ndarray
            y-values to calculate

        kwargs: Any
            keyword arguments to control the function of z
        """
        self.x = xrange
        self.y = yrange
        # actual implementation of self.z is flexible
        self.z = np.empty((self.y.size, self.x.size))
        self.x_2d, self.y_2d = np.meshgrid(self.x, self.y)
        self.shape = self.z.shape

    @abstractmethod
    def __call__(self):
        """
        Returns
        -------
        CoordinateDict: Dict[str, np.ndarray]
            {'X': self.x_2d, 'Y': self.y_2d, 'Z': self.z}
        """
        return dict(X=self.x, Y=self.y, Z=self.z)
