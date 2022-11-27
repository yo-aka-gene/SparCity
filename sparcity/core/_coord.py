"""
Class of Coordinates
"""
import numpy as np

from sparcity.dev import is_monotonical_increasing, typechecker, valchecker


class Coord:
    """
    Class of Coordinates

    Methods
    -------
    __init__(x: np.ndarray, y:np.ndarray, z:np.ndarray) -> None:
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
    """
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
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

        z: np.ndarray
            z values (e.g., altitude) for the designated area.
            Expected to be a 2D-array (for ver.0.1.0).

        Examples
        --------
        >>> import numpy as np
        >>> from sparcity import Coord
        >>> x = np.linspace(0, 10, 100)
        >>> y = np.linspace(0, 20, 150)
        >>> np.random.seed(123)
        >>> z = np.random.randn(150, 100)
        >>> x_2d, y_2d = np.meshgrid(x, y)
        >>> area = Coord(x=x, y=y, z=z)
        >>> np.all(area.x == x)
        True
        >>> np.all(area.y == y)
        True
        >>> np.all(area.z == z)
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
        typechecker(z, np.ndarray, "z")
        valchecker(x.ndim == 1)
        valchecker(y.ndim == 1)
        valchecker(z.ndim == 2)
        valchecker(x.size == z.shape[1])
        valchecker(y.size == z.shape[0])
        valchecker(is_monotonical_increasing(x))
        valchecker(is_monotonical_increasing(y))
        self.x = x
        self.y = y
        self.z = z
        self.x_2d, self.y_2d = np.meshgrid(x, y)
        self.shape = z.shape

    def __call__(self) -> dict:
        """
        Returns
        -------
        CoordinateDict: Dict[str, np.ndarray]
            {'X': self.x_2d, 'Y': self.y_2d, 'Z': self.z}

        Examples
        --------
        >>> import numpy as np
        >>> from sparcity import Coord
        >>> x = np.linspace(0, 10, 100)
        >>> y = np.linspace(0, 20, 150)
        >>> np.random.seed(123)
        >>> z = np.random.randn(150, 100)
        >>> area = Coord(x=x, y=y, z=z)
        >>> area_dict = area()
        >>> np.all(area_dict["X"] == area.x_2d)
        True
        >>> np.all(area_dict["Y"] == area.y_2d)
        True
        >>> np.all(area_dict["Z"] == area.z)
        True
        """
        return dict(X=self.x_2d, Y=self.y_2d, Z=self.z)
