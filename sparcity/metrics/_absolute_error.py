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
    >>> err11.shape == area1.shape
    True
    >>> np.all(err11.x == area1.x)
    True
    >>> np.all(err11.y == area1.y)
    True
    >>> np.all(err11.z == np.zeros(area1.shape))
    True
    >>> err12 = absolute_error(area1, area2)
    >>> err12.shape == area1.shape == area2.shape
    True
    >>> np.all(err12.x == area1.x) and np.all(err12.x == area2.x)
    True
    >>> np.all(err12.y == area1.y) and np.all(err12.y == area2.y)
    True
    >>> np.all(err12.z.astype(np.float32) == np.ones(area1.shape))
    True
    >>> err21 = absolute_error(area2, area1)
    >>> np.all(err21.z.astype(np.float32) == np.ones(area1.shape))
    True
    """
    arg_check(**locals())
    return Coord(
        x=pred.x,
        y=pred.y,
        z=np.abs(pred.z - test.z)
    )


def mean_absolute_error_score(
    pred: Coord,
    test: Coord
) -> float:
    """
    function to calculate mean absolute error score

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
    >>> from sparcity.metrics import mean_absolute_error_score as mae
    >>> area1 = QuadraticGenerator()
    >>> area2 = QuadraticGenerator(d=1)
    >>> mae(area1, area1)
    0.0
    >>> mae(area1, area2)
    1.0
    >>> mae(area2, area1)
    1.0
    """
    arg_check(**locals())
    return np.abs(pred.z - test.z).mean()
