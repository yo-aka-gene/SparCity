"""
Simulatory-data Generator for quadratic function
"""
from typing import Union

import numpy as np

from sparcity.core import Generator
from sparcity.dev import typechecker


class QuadraticGenerator(Generator):
    """
    Class for quadratic function generator

    .. math:: f(x, y) = a(x - x_1)^2 + b(y - y_1)^2 + c(x - x_2)(y - y_2)

    Methods
    -------
    __init__(a, b, c, x1, x2, y1, y2, xrange, yrange) -> None:
        initialize attributes listed below.

    __call__() -> Dict[str, np.ndarray]:
        returns {'X': self.x_2d, 'Y': self.y_2d, 'Z': self.z}
        The return value is designed for surfaceplot of matplotlib.

    Attributes
    ----------
    x: np.ndarray
        x coordinates (longitude) for the designated area.
        Given by `xrange` argument (default is `np.linspace(0, 100, 1000)`).

    y: np.ndarray
        y coordinates (latitude) for the designated area.
        Given by `yrange` argument (default is `np.linspace(0, 100, 1000)`).

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
    def __init__(
        self,
        a: Union[float, int] = 1,
        b: Union[float, int] = 1,
        c: Union[float, int] = 1,
        x1: Union[float, int] = 50,
        x2: Union[float, int] = 50,
        y1: Union[float, int] = 30,
        y2: Union[float, int] = 30,
        xrange: np.ndarray = np.linspace(0, 100, 1000),
        yrange: np.ndarray = np.linspace(0, 100, 1000)
    ) -> None:
        """
        Parameters
        ----------
        a: Union[float, int], default: 1
            coeff for :math:`(x - x_1)^2`

        b: Union[float, int], default: 1
            coeff for :math:`(y - y_1)^2`

        c: Union[float, int], default: 1
            coeff for :math:`(x - x_2)(y - y_2)`

        x1: Union[float, int], default: 50
            centering factor in :math:`(x - x_1)^2`

        x2: Union[float, int], default: 50
            centering factor in :math:`(x - x_2)(y - y_2)`

        y1: Union[float, int], default: 30
            centering factor in :math:`(y - y_1)^2`

        y2: Union[float, int], default: 30
            centering factor in :math:`(x - x_2)(y - y_2)`

        xrange: numpy.ndarray, default: numpy.linspace(0, 100, 1000)
            x-values to calculate

        yrange: numpy.ndarray, default: numpy.linspace(0, 100, 1000)
            y-values to calculate

        Examples
        --------
        >>> import numpy as np
        >>> from sparcity.dataset import QuadraticGenerator
        >>> model1 = QuadraticGenerator()
        >>> np.all(model1.x == np.linspace(0, 100, 1000))
        True
        >>> np.all(model1.y == np.linspace(0, 100, 1000))
        True
        >>> x_2d, y_2d = np.meshgrid(model1.x, model1.y)
        >>> np.all(model1.x_2d == x_2d)
        True
        >>> np.all(model1.y_2d == y_2d)
        True
        >>> z1 = (x_2d - 50) ** 2 + (y_2d - 30) ** 2 + (x_2d - 50) * (y_2d - 30)
        >>> np.all(model1.z == z1)
        True
        >>> model1.shape
        (1000, 1000)
        >>> a, b, c = (2, 3, 4)
        >>> x1, x2, y1, y2 = (10, 20, 30, 40)
        >>> model2 = QuadraticGenerator(a, b, c, x1, x2, y1, y2)
        >>> np.all(model2.x == np.linspace(0, 100, 1000))
        True
        >>> np.all(model2.y == np.linspace(0, 100, 1000))
        True
        >>> z2 = a * (x_2d - x1) ** 2 + b * (y_2d - y1) ** 2 + c * (x_2d - x2) * (y_2d - y2)
        >>> np.all(model2.z == z2)
        True
        >>> model2.shape
        (1000, 1000)
        """
        typechecker(a, (float, int), "a")
        typechecker(b, (float, int), "b")
        typechecker(c, (float, int), "c")
        typechecker(x1, (float, int), "x1")
        typechecker(x2, (float, int), "x2")
        typechecker(y1, (float, int), "y1")
        typechecker(y2, (float, int), "y2")
        typechecker(xrange, np.ndarray, "xrange")
        typechecker(yrange, np.ndarray, "yrange")

        self.x = xrange
        self.y = yrange
        self.x_2d, self.y_2d = np.meshgrid(self.x, self.y)
        self.z = a * (self.x_2d - x1) ** 2 + \
            b * (self.y_2d - y1) ** 2 + c * (self.x_2d - x2) * (self.y_2d - y2)
        self.shape = self.z.shape

    def __call__(self):
        """
        Returns
        -------
        CoordinateDict: Dict[str, np.ndarray]
            {'X': self.x_2d, 'Y': self.y_2d, 'Z': self.z}

        Examples
        --------
        >>> import numpy as np
        >>> from sparcity.dataset import QuadraticGenerator
        >>> model = QuadraticGenerator()
        >>> area_dict = model()
        >>> np.all(area_dict["X"] == model.x_2d)
        True
        >>> np.all(area_dict["Y"] == model.y_2d)
        True
        >>> np.all(area_dict["Z"] == model.z)
        True
        """
        return dict(X=self.x_2d, Y=self.y_2d, Z=self.z)
