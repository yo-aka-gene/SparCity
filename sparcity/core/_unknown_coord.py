"""
Class of Unknown Coordinates
"""
import numpy as np

from sparcity.dev import typechecker, valchecker
from ._coord import Coord


class UnknownCoord(Coord):
    """
    Class of Coordinates of unknown z values (altitudes)

    Methods
    -------
    __init__(x: np.ndarray, y:np.ndarray) -> None:
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
        z values (e.g., altitude) for the designated area are set as numpy.nan.
        Expected to be a 2D-array (for ver.0.1.0).

    x_2d: np.ndarray
        x coordinates (longitude) corresponding to z.
        Expected to be a 2D-array (for ver.0.1.0).

    y_2d: np.ndarray
        y coordinates (latitude) corresponding to z.
        Expected to be a 2D-array (for ver.0.1.0).

    shape: List[int]
        shape of z
    """
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray
    ):
        """
        Parameters
        ----------
        x: np.ndarray
            x coordinates (longitude) for the designated area.
            Expected to be a 1D-array.

        y: np.ndarray
            y coordinates (latitude) for the designated area.
            Expected to be a 1D-array.

        Examples
        --------
        >>> import numpy as np
        >>> from sparcity import UnknownCoord
        >>> x = np.linspace(0, 10, 100)
        >>> y = np.linspace(0, 20, 150)
        >>> x_2d, y_2d = np.meshgrid(x, y)
        >>> area = UnknownCoord(x=x, y=y)
        >>> np.all(area.x == x)
        True
        >>> np.all(area.y == y)
        True
        >>> np.all(np.isnan(area.z))
        True
        >>> np.all(area.x_2d == x_2d)
        True
        >>> np.all(area.y_2d == y_2d)
        True
        >>> area.shape == (150, 100)
        True
        """
        typechecker(x, np.ndarray, "x")
        typechecker(y, np.ndarray, "y")
        valchecker(len(x.shape) == 1)
        valchecker(len(y.shape) == 1)
        self.x = x
        self.y = y
        self.z = np.array(
            [np.nan for v in np.arange(y.size * x.size)]
        ).reshape(y.size, x.size)
        self.x_2d, self.y_2d = np.meshgrid(x, y)
        self.shape = self.z.shape

    def __call__(self) -> dict:
        """
        Returns
        -------
        CoordinateDict: Dict[str, np.ndarray]
            {'X': self.x_2d, 'Y': self.y_2d, 'Z': self.z}

        Examples
        --------
        >>> import numpy as np
        >>> from sparcity import UnknownCoord
        >>> x = np.linspace(0, 10, 100)
        >>> y = np.linspace(0, 20, 150)
        >>> area = UnknownCoord(x=x, y=y)
        >>> area_dict = area()
        >>> np.all(area_dict["X"] == area.x_2d)
        True
        >>> np.all(area_dict["Y"] == area.y_2d)
        True
        >>> np.all(np.isnan(area_dict["Z"]))
        True
        """
        return dict(X=self.x_2d, Y=self.y_2d, Z=self.z)
