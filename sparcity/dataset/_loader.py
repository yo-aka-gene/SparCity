"""
data loading functions
"""
from typing import Union
import numpy as np

from sparcity.core import Coord
from sparcity.dev import typechecker
from .data import oyama


def load_oyama(
    as_Coord: bool = False
) -> Union[np.ndarray, Coord]:
    """
    function to load oyama data

    Parameters
    ----------
    as_Coord: bool, default: False
        if True, it returns the data as numpy.ndarray

    Returns
    -------
    CoordinateData: Union[numpy.ndarray, Coord]
        if as_Coord == True, it returns data as Coord;
        otherwise, as numpy.ndarray

    Notes
    -----
    1 in oyama data is equivalent to 10 meters

    Examples
    --------
    >>> import numpy as np
    >>> from sparcity.dataset import load_oyama
    >>> from sparcity.core import Coord
    >>> numpy_data = load_oyama()
    >>> isinstance(numpy_data, np.ndarray)
    True
    >>> numpy_data.shape
    (750, 1125)
    >>> coordinate = load_oyama(as_Coord=True)
    >>> isinstance(coordinate, Coord)
    True
    >>> np.all(coordinate.x == np.arange(1125))
    True
    >>> np.all(coordinate.y == np.arange(750))
    True
    >>> np.all(coordinate.z == numpy_data)
    True
    """
    typechecker(as_Coord, bool, "as_Coord")
    if as_Coord:
        x = np.arange(oyama.shape[1])
        y = np.arange(oyama.shape[0])
        return Coord(x=x, y=y, z=oyama)
    else:
        return oyama
