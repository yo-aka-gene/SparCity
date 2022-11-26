"""
functions to calculate absolute errors
"""
import numpy as np

from sparcity.core import Coord
from sparcity.debug_utils import arg_check


def absolute_error(
    pred: Coord,
    test: Coord
) -> Coord:
    """
    function to calculate absolute error in each points

    Parameters
    ----------
    pred: Coord
        Coord of predicted coordinates

    test: Coord
        Coord of test data (TestData.field)

    Examples
    --------
    >>> import numpy as np
    >>> from sparcity import Coord
    >>> from sparcity.dataset import QuadraticGenerator
    >>> from sparcity.metrics import absolute_error
    >>> area1 = QuadraticGenerator()
    >>> area2 = QuadraticGenerator(d=1)
    >>> err11 = absolute_error(area1, area1)
    >>> isinstance(err11, Coord)
    True
    """
    arg_check(**locals())
    return Coord(
        x=pred.x,
        y=pred.x,
        z=np.abs(pred.z - test.z)
    )
