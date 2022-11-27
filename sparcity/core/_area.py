"""
function to crop a square-shaped area
"""
from itertools import product
from typing import List, Union

import numpy as np
from tqdm import tqdm

from sparcity.dev import get_subarray_loc, is_subarray, typechecker, valchecker
from ._coord import Coord
from ._subcoord import SubCoord


def subarea(
    x: Union[float, int],
    y: Union[float, int],
    length: Union[float, int],
    field: Coord
) -> SubCoord:
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
    Subarea: SubCoord
        SubCoord class of the cropped subarea.
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

    return SubCoord(
        x=field.x[locx],
        y=field.y[locy],
        z=field.z[:, locx][locy, :],
        locx=locx,
        locy=locy
    )


def patch(
        list_of_Coord: List[Coord],
        field: Coord,
        logging: bool = False
) -> Coord:
    """
    A function to concatenate discrete subareas.
    Use this function to make TestData class for discrete areas

    Parameters
    ----------
    list_of_Coord: List[Coord]
        list of SubCoord in field

    field: Coord
        Coord class of the original field.
        Required for get information about overall (x, y) values.
        Use UnknownCoord instead if the correct z values are unknown.

    logging: bool, default: False
        Set True to show progress bar

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
    typechecker(logging, bool, "logging")
    is_all_subcoord = True
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
        is_all_subcoord &= isinstance(v, SubCoord)

    z = np.zeros(field.shape)
    z = np.where(z == 0, np.nan, np.nan)

    if is_all_subcoord:
        if logging:
            for v in tqdm(
                list_of_Coord,
                desc="Referring Data Location",
                total=len(list_of_Coord)
            ):
                z[v.loc] = v.z.ravel()
        else:
            for v in list_of_Coord:
                z[v.loc] = v.z.ravel()

    else:
        if logging:
            for v in tqdm(
                list_of_Coord,
                desc="Referring Data Location",
                total=len(list_of_Coord)
            ):
                m_x = np.where(field.x == v.x.min())[0].item()
                M_x = np.where(field.x == v.x.max())[0].item()
                m_y = np.where(field.y == v.y.min())[0].item()
                M_y = np.where(field.y == v.y.max())[0].item()
                locx = np.arange(m_x, M_x + 1)
                locy = np.arange(m_y, M_y + 1)
                loc = (
                    np.tile(locy, locx.size).reshape(locx.size, locy.size).T.ravel(),
                    np.tile(locx, locy.size)
                )
                z[loc] = v.z.ravel()
        else:
            for v in list_of_Coord:
                m_x = np.where(field.x == v.x.min())[0].item()
                M_x = np.where(field.x == v.x.max())[0].item()
                m_y = np.where(field.y == v.y.min())[0].item()
                M_y = np.where(field.y == v.y.max())[0].item()
                locx = np.arange(m_x, M_x + 1)
                locy = np.arange(m_y, M_y + 1)
                loc = (
                    np.tile(locy, locx.size).reshape(locx.size, locy.size).T.ravel(),
                    np.tile(locx, locy.size)
                )
                z[loc] = v.z.ravel()

    return Coord(
        x=field.x,
        y=field.y,
        z=z
    )


def cleave(
    field: Coord,
    n_x: Union[int, np.int64] = 1,
    n_y: Union[int, np.int64] = 1
) -> List[SubCoord]:
    """
    function to cleave fields into subareas

    Parameters
    ----------
    fileds: Coord
        original field

    n_x: Union[int, np.int64], defualt = 1
        Number of blocks in x (longitude)

    n_y: Union[int, np.int64], defualt = 1
        Number of blocks in y (latitude)

    Returns
    -------
    Blocks: List[SubCoord]
        list of cleaved blocks (SubCoord)

    Examples
    --------
    >>> import numpy as np
    >>> from sparcity.core import Coord, cleave, SubCoord
    >>> x = np.arange(4)
    >>> y = np.arange(3)
    >>> z = np.arange(12).reshape(3, 4) + 20
    >>> field = Coord(x=x, y=y, z=z)
    >>> lst_subcoord = cleave(field, 2, 3)
    >>> len(lst_subcoord)
    6
    >>> isinstance(lst_subcoord[0], SubCoord)
    True
    >>> lst_subcoord[0].shape
    (1, 2)
    >>> lst_subcoord[1].x
    array([0, 1])
    >>> lst_subcoord[1].y
    array([1])
    >>> lst_subcoord[1].z
    array([[24, 25]])
    """
    typechecker(field, Coord, "field")
    typechecker(n_x, (int, np.int64), "n_x")
    typechecker(n_y, (int, np.int64), "n_y")
    valchecker(n_x > 0)
    valchecker(n_y > 0)
    valchecker(field.x.size >= n_x)
    valchecker(field.y.size >= n_y)
    return [
        SubCoord(
            x=x,
            y=y,
            z=field.z[
                :, get_subarray_loc(x, field.x)
            ][
                get_subarray_loc(y, field.y), :
            ],
            locx=get_subarray_loc(x, field.x),
            locy=get_subarray_loc(y, field.y)
        ) for x, y in product(*[
            np.split(field.x, n_x),
            np.split(field.y, n_y)
        ])
    ]


def subarea_in_grid(
    x: Union[float, int],
    y: Union[float, int],
    length: Union[float, int],
    grid: SubCoord,
    field: Coord
) -> SubCoord:
    """
    A function that crops a squared-shaped area in grids
    centering the designated point (x, y).
    Expected to be called in sampler classes.

    Parameters
    ----------
    x: Union[float, int]
        x-coordinate (longitude) for the center

    y: Union[float, int]
        y-coordinate (latitude) for the center

    length: Union[float, int]
        length of a side in the square
        expected to be positive while being smaller than each side in the original field.

    grid: SubCoord
        SubCoord of designated grid

    field: Coord
        Coord class of the original field

    Returns
    -------
    Subarea: SubCoord
        SubCoord class of the cropped subarea.
        The center position will be automatically formatted to adjust the original field.

    Examples
    --------
    >>> import numpy as np
    >>> from sparcity import Coord, subarea
    >>> from sparcity.core import cleave, subarea_in_grid
    >>> x = np.arange(8)
    >>> y = np.arange(8)
    >>> z = np.arange(64).reshape(8, 8) + 100
    >>> field = Coord(x=x, y=y, z=z)
    >>> grid = cleave(field, 2, 2)[1]
    >>> grid.x
    array([0, 1, 2, 3])
    >>> grid.y
    array([4, 5, 6, 7])
    >>> grid.z
    array([[132, 133, 134, 135],
           [140, 141, 142, 143],
           [148, 149, 150, 151],
           [156, 157, 158, 159]])
    >>> subarea = subarea_in_grid(x=1, y=5, length=2, grid=grid, field=field)
    >>> subarea.x
    array([0, 1, 2])
    >>> subarea.y
    array([4, 5, 6])
    >>> subarea.z
    array([[132, 133, 134],
           [140, 141, 142],
           [148, 149, 150]])
    """
    typechecker(x, (float, int), "x")
    typechecker(y, (float, int), "y")
    typechecker(length, (float, int), "length")
    typechecker(grid, SubCoord, "grid")
    typechecker(field, Coord, "field")
    valchecker(
        0 < length < min(
            grid.x.max() - grid.x.min(),
            grid.y.max() - grid.y.min()
        )
    )

    diff = length / 2
    if x <= grid.x.min() + diff:
        locx = np.where((grid.x.min() <= field.x) & (field.x <= grid.x.min() + length))[0]
    elif x >= grid.x.max() - diff:
        locx = np.where((grid.x.max() >= field.x) & (field.x >= grid.x.max() - length))[0]
    else:
        locx = np.where((x - diff <= field.x) & (field.x <= x + diff))[0]

    if y <= grid.y.min() + diff:
        locy = np.where((grid.y.min() <= field.y) & (field.y <= grid.y.min() + length))[0]
    elif y >= grid.y.max() - diff:
        locy = np.where((grid.y.max() >= field.y) & (field.y >= grid.y.max() - length))[0]
    else:
        locy = np.where((y - diff <= field.y) & (field.y <= y + diff))[0]

    return SubCoord(
        x=field.x[locx],
        y=field.y[locy],
        z=field.z[:, locx][locy, :],
        locx=locx,
        locy=locy
    )
