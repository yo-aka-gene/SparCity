"""
function to forget z values
"""
from sparcity.core import Coord, UnknownCoord
from sparcity.dev import typechecker


def forget(field: Coord) -> UnknownCoord:
    """
    function to forget z values (altitude) of Coord class

    Parameters
    ----------
    field: Coord
        original field

    Returns
    -------
    unknown_field: UnknownCoord

    Examples
    --------
    >>> import numpy as np
    >>> from sparcity import UnknownCoord
    >>> from sparcity.dataset import QuadraticGenerator
    >>> from sparcity.debug_utils import forget
    >>> field = QuadraticGenerator()
    >>> area = forget(field)
    >>> isinstance(area, UnknownCoord)
    True
    >>> np.all(area.x == field.x)
    True
    >>> np.all(area.y == field.y)
    True
    >>> area.shape == field.shape
    True
    >>> np.all(np.isnan(area.z))
    True
    """
    typechecker(field, Coord, "field")
    return UnknownCoord(x=field.x, y=field.y)
