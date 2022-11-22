"""
function for random choice of the center coordination
"""
from typing import Tuple

import numpy as np

from sparcity.core import Coord
from sparcity.dev import typechecker
from ._random import seedfixer


def random_coord(
    field: Coord,
    seed: int
) -> Tuple[float]:
    """
    random choice of (x, y) coordinate

    Parameters
    ----------
    field: Coord
        original field to choose the random coordinate from

    seed: int
        random seed

    Returns
    -------
    (x, y): Tuple[float]
        tuple of floats (length == 2)

    Examples
    --------
    >>> import numpy as np
    >>> import sparcity as spc
    >>> x, y = np.arange(3), np.arange(5)
    >>> z = np.zeros((5, 3))
    >>> area = spc.Coord(x, y, z)
    >>> spc.random.random_coord(area, seed=123)
    (1.3307395609368218, 3.696682980990738)
    """
    typechecker(field, Coord, "field")
    typechecker(seed, int, "seed")
    seedfixer(seed, 2, 0)
    x = (
        field.x.max() - field.x.min()
    ) * np.random.random() + field.x.min()
    seedfixer(seed, 2, 1)
    y = (
        field.y.max() - field.y.min()
    ) * np.random.random() + field.y.min()

    return (x, y)
