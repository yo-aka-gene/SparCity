"""
function to crop a square-shaped area
"""
from typing import Union

import numpy as np

from ._coord import Coord
from sparcity.dev import typechecker, valchecker


def subarea(
    x: Union[float, int],
    y: Union[float, int],
    length: Union[float, int],
    field: Coord
) -> Coord:
    """
    A function that crops a squared-shaped area
    centering the designated point (x, y)

    Parameters
    ----------
    x: Union[float, int]
        x-coordinate (longitude) for the center

    y: Union[float, int]
        y-coordinate (latitude) for the center

    length: Union[float, int]
        length of a side in the square
        expected to be positive while being smaller than each side in the original field.

    field: Coord
        Coord class of the original field

    Returns
    -------
    Subarea: Coord
        Coord class of the cropped subarea.
        The center position will be automatically formatted to adjust the original field.

    Examples
    --------
    >>> import numpy as np
    >>> from sparcity import Coord, subarea
    >>> x = np.arange(5)
    >>> y = np.arange(4)
    >>> z = np.arange(20).reshape(4, 5)
    >>> field = Coord(x=x, y=y, z=z)
    >>> area1 = subarea(x=0.3, y=0.5, length=2, field=field)
    >>> area1.x
    array([0, 1, 2])
    >>> area1.y
    array([0, 1, 2])
    >>> area1.z
    array([[ 0,  1,  2],
           [ 5,  6,  7],
           [10, 11, 12]])
    >>> area2 = subarea(x=3, y=2, length=2, field=field)
    >>> area2.x
    array([2, 3, 4])
    >>> area2.y
    array([1, 2, 3])
    >>> area2.z
    array([[ 7,  8,  9],
           [12, 13, 14],
           [17, 18, 19]])
    """
    typechecker(x, (float, int), "x")
    typechecker(y, (float, int), "y")
    typechecker(length, (float, int), "length")
    typechecker(field, Coord, "field")
    valchecker(
        0 < length < min(
            field.x.max() - field.x.min(),
            field.y.max() - field.y.min()
        )
    )

    diff = length / 2
    if x <= field.x.min() + diff:
        locx = np.where(field.x <= field.x.min() + length)[0]
    elif x >= field.x.max() - diff:
        locx = np.where(field.x >= field.x.max() - length)[0]
    else:
        locx = np.where((x - diff <= field.x) & (field.x <= x + diff))[0]

    if y <= field.y.min() + diff:
        locy = np.where(field.y <= field.y.min() + length)[0]
    elif y >= field.y.max() - diff:
        locy = np.where(field.y >= field.y.max() - length)[0]
    else:
        locy = np.where((y - diff <= field.y) & (field.y <= y + diff))[0]

    return Coord(
        x=field.x[locx],
        y=field.y[locy],
        z=field.z[:, locx][locy, :]
    )
