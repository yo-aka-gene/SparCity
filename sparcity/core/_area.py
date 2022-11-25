"""
function to crop a square-shaped area
"""
from itertools import product
from typing import List, Union

import numpy as np

from ._coord import Coord
from sparcity.dev import is_subarray, typechecker, valchecker


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


def patch(
        list_of_Coord: List[Coord],
        field: Coord
) -> Coord:
    """
    A function to concatenate discrete subareas.
    Use this function to make TestData class for discrete areas

    Parameters
    ----------
    list_of_Coord: List[Coord]
        list of subareas in field

    field: Coord
        Coord class of the original field.
        Required for get information about overall (x, y) values.
        Use UnknownCoord instead if the correct z values are unknown.

    Returns
    -------
    DiscreteArea: Coord
        Coord class of discrete area. The z values are taken over from the subareas
        while the rest z values missing in the subareas are filled with numpy.nan.

    Examples
    --------
    >>> import numpy as np
    >>> from sparcity import Coord, patch, UnknownCoord
    >>> x1 = np.array([0, 1, 2])
    >>> y1 = np.array([3, 4, 5])
    >>> z1 = np.array([[ 6,  7,  8], [ 9, 10, 11], [12, 13, 14]])
    >>> area1 = Coord(x=x1, y=y1, z=z1)
    >>> x2 = np.array([2, 3])
    >>> y2 = np.array([5, 6])
    >>> z2 = np.array([[14, 15], [16, 17]])
    >>> area2 = Coord(x=x2, y=y2, z=z2)
    >>> x = np.unique(np.append(x1, x2))
    >>> x
    array([0, 1, 2, 3])
    >>> y = np.unique(np.append(y1, y2))
    >>> y
    array([3, 4, 5, 6])
    >>> field = UnknownCoord(x=x, y=y)
    >>> patched = patch([area1, area2], field)
    >>> np.all(patched.x == x)
    True
    >>> np.all(patched.y == y)
    True
    >>> patched.z
    array([[ 6.,  7.,  8., nan],
           [ 9., 10., 11., nan],
           [12., 13., 14., 15.],
           [nan, nan, 16., 17.]])
    >>> patched.shape
    (4, 4)
    """
    typechecker(list_of_Coord, list, "list_of_Coord")
    typechecker(field, Coord, "field")
    for i, v in enumerate(list_of_Coord):
        typechecker(v, Coord, f"list_of_Coord[{i}]")
        valchecker(
            is_subarray(v.x, field.x),
            f"list_of_Coord[{i}] expected to be a subarea of field"
        )
        valchecker(
            is_subarray(v.y, field.y),
            f"list_of_Coord[{i}] expected to be a subarea of field"
        )

    directsum = np.concatenate(
        [
            np.array([
                (*v, subarea.z.ravel()[i]) for i, v in enumerate(
                    product(subarea.y, subarea.x)
                )
            ]) for subarea in list_of_Coord
        ],
        axis=0
    )
    summed_coo, idx = np.unique(directsum[:, :2], axis=0, return_index=True)
    summed_z = directsum[idx, 2]
    summed_coo = np.array([str(tuple(v)) for v in summed_coo])

    z = np.array([
        summed_z[
            np.where(summed_coo == str(coo))[0].item()
        ] if str(coo) in summed_coo else np.nan for coo in product(
            field.y, field.x
        )
    ]).reshape(*field.shape)

    return Coord(
        x=field.x,
        y=field.y,
        z=z
    )
