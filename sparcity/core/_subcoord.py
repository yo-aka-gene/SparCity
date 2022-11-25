"""
Class of Subareas' Coordinates
"""
import numpy as np

from sparcity.dev import typechecker, valchecker
from ._coord import Coord


class SubCoord(Coord):
    """
    Class of Subareas' Coordinates.
    Expected to be generated via sparcity.subarea function.

    Methods
    -------
    __init__(
        x: numpy.ndarray,
        y: numpy.ndarray,
        z: numpy.ndarray,
        locx: numpy.ndarray,
        locy: numpy.ndarray
    ) -> None:
        initialize attributes listed below.

    __call__() -> Dict[str, np.ndarray]:
        returns {'X': self.x_2d, 'Y': self.y_2d, 'Z': self.z}
        The return value is designed for surfaceplot of matplotlib.

    Attributes
    ----------
    x: np.ndarray
        x coordinates (longitude) for the designated area.
        Expected to be a 1D-array.

    y: np.ndarray
        y coordinates (latitude) for the designated area.
        Expected to be a 1D-array.

    z: np.ndarray
        z values (e.g., altitude) for the designated area.
        Expected to be a 2D-array (for ver.0.1.0).

    x_2d: np.ndarray
        x coordinates (longitude) corresponding to z.
        Expected to be a 2D-array (for ver.0.1.0).

    y_2d: np.ndarray
        y coordinates (latitude) corresponding to z.
        Expected to be a 2D-array (for ver.0.1.0).

    shape: List[int]
        shape of z

    loc: Tuple[numpy.ndarray]
    """
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        locx: np.ndarray,
        locy: np.ndarray
    ):
        """
        Parameters
        ----------
        x: numpy.ndarray
            x coordinates (longitude) for the designated area.
            Expected to be a 1D-array.

        y: numpy.ndarray
            y coordinates (latitude) for the designated area.
            Expected to be a 1D-array.

        z: np.ndarray
            z values (e.g., altitude) for the designated area.
            Expected to be a 2D-array (for ver.0.1.0).

        locx: numpy.ndarray
            location (regarding x-values) in the original matrix

        locy: numpy.ndarray
            location (regarding y-values) in the original matrix

        Examples
        --------
        >>> import numpy as np
        >>> from sparcity import Coord, subarea
        >>> from sparcity.core import SubCoord
        >>> from sparcity.dev import is_subarray
        >>> x = np.arange(5)
        >>> y = np.arange(4)
        >>> z = np.arange(20).reshape(4, 5)
        >>> field = Coord(x=x, y=y, z=z)
        >>> area = subarea(x=0.3, y=0.5, length=2, field=field)
        >>> isinstance(area, Coord)
        True
        >>> isinstance(area, SubCoord)
        True
        >>> is_subarray(area.x, x)
        True
        >>> is_subarray(area.y, y)
        True
        >>> area.z
        array([[ 0,  1,  2],
               [ 5,  6,  7],
               [10, 11, 12]])
        >>> x_2d, y_2d = np.meshgrid(area.x, area.y)
        >>> np.all(area.x_2d == x_2d)
        True
        >>> np.all(area.y_2d == y_2d)
        True
        >>> area.shape
        (3, 3)
        >>> np.all(field.z[area.loc] == area.z.ravel())
        True
        """
        typechecker(x, np.ndarray, "x")
        typechecker(y, np.ndarray, "y")
        typechecker(z, np.ndarray, "z")
        typechecker(locx, np.ndarray, "locx")
        typechecker(locy, np.ndarray, "locy")
        valchecker(len(x.shape) == 1)
        valchecker(len(y.shape) == 1)
        valchecker(len(z.shape) == 2)
        valchecker(len(locx.shape) == 1)
        valchecker(len(locy.shape) == 1)
        valchecker(x.size == z.shape[1])
        valchecker(y.size == z.shape[0])
        valchecker(locx.size * locy.size == z.ravel().size)
        self.x = x
        self.y = y
        self.z = z
        self.x_2d, self.y_2d = np.meshgrid(x, y)
        self.shape = z.shape
        self.loc = (
            np.tile(locy, locx.size).reshape(locx.size, locy.size).T.ravel(),
            np.tile(locx, locy.size)
        )

    def __call__(self) -> dict:
        """
        Returns
        -------
        CoordinateDict: Dict[str, np.ndarray]
            {'X': self.x_2d, 'Y': self.y_2d, 'Z': self.z}

        Examples
        --------
        >>> import numpy as np
        >>> from sparcity import Coord, subarea
        >>> x = np.linspace(0, 10, 100)
        >>> y = np.linspace(0, 20, 150)
        >>> np.random.seed(123)
        >>> z = np.random.randn(150, 100)
        >>> field = Coord(x=x, y=y, z=z)
        >>> area = subarea(x=5, y=6, length=2, field=field)
        >>> area_dict = area()
        >>> np.all(area_dict["X"] == area.x_2d)
        True
        >>> np.all(area_dict["Y"] == area.y_2d)
        True
        >>> np.all(area_dict["Z"] == area.z)
        True
        """
        return dict(X=self.x_2d, Y=self.y_2d, Z=self.z)
