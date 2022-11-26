"""
Extension functions for sparcity/metrics
"""
import numpy as np

from sparcity.core import Coord, UnknownCoord
from sparcity.dev import typechecker, valchecker


def arg_check(
    pred: Coord,
    test: Coord
) -> None:
    """
    function to check if the arguments are suitable for
    model evaluation functions

    Parameters
    ----------
    pred: Coord
        predicted coordinates

    test: Coord
        test data's coordinates (TestData.field)

    Examples
    --------
    >>> import numpy as np
    >>> from sparcity import Coord, UnknownCoord
    >>> from sparcity.dataset import QuadraticGenerator
    >>> from sparcity.debug_utils import arg_check, forget
    >>> def my_metric(pred, test):\
            arg_check(**locals())
    >>> coord = QuadraticGenerator()
    >>> u_coord = forget(coord)
    >>> isinstance(coord, Coord)
    True
    >>> isinstance(u_coord, Coord)
    True
    >>> isinstance(coord, UnknownCoord)
    False
    >>> isinstance(u_coord, UnknownCoord)
    True
    >>> my_metric(coord, coord)
    >>> my_metric(u_coord, coord)
    Traceback (most recent call last):
        ...
    AssertionError: Invalid value detected. Check the requirements. Predicted Coord is Unknown.
    >>> my_metric(coord, u_coord)
    Traceback (most recent call last):
        ...
    AssertionError: Invalid value detected. Check the requirements. Test Coord is Unknown.
    >>> inconsistent_xshape = QuadraticGenerator(xrange=np.linspace(0, 1000, 10))
    >>> my_metric(inconsistent_xshape, coord)
    Traceback (most recent call last):
        ...
    AssertionError: Invalid value detected. Check the requirements. Shapes of pred and test should be mutual.
    >>> my_metric(coord, inconsistent_xshape)
    Traceback (most recent call last):
        ...
    AssertionError: Invalid value detected. Check the requirements. Shapes of pred and test should be mutual.
    >>> inconsistent_yshape = QuadraticGenerator(yrange=np.linspace(0, 1000, 10))
    >>> my_metric(inconsistent_yshape, coord)
    Traceback (most recent call last):
        ...
    AssertionError: Invalid value detected. Check the requirements. Shapes of pred and test should be mutual.
    >>> my_metric(coord, inconsistent_yshape)
    Traceback (most recent call last):
        ...
    AssertionError: Invalid value detected. Check the requirements. Shapes of pred and test should be mutual.
    >>> inconsistent_xrange = QuadraticGenerator(xrange=np.linspace(0, 10, 1000))
    >>> my_metric(inconsistent_xrange, coord)
    Traceback (most recent call last):
        ...
    AssertionError: Invalid value detected. Check the requirements. (x, y) coodinates should be mutual with pred and test.
    >>> my_metric(coord, inconsistent_xrange)
    Traceback (most recent call last):
        ...
    AssertionError: Invalid value detected. Check the requirements. (x, y) coodinates should be mutual with pred and test.
    >>> inconsistent_yrange = QuadraticGenerator(yrange=np.linspace(0, 10, 1000))
    >>> my_metric(inconsistent_yrange, coord)
    Traceback (most recent call last):
        ...
    AssertionError: Invalid value detected. Check the requirements. (x, y) coodinates should be mutual with pred and test.
    >>> my_metric(coord, inconsistent_yrange)
    Traceback (most recent call last):
        ...
    AssertionError: Invalid value detected. Check the requirements. (x, y) coodinates should be mutual with pred and test.
    """
    typechecker(pred, Coord, "pred")
    typechecker(test, Coord, "test")
    valchecker(
        not isinstance(pred, UnknownCoord),
        "Predicted Coord is Unknown."
    )
    valchecker(
        not isinstance(test, UnknownCoord),
        "Test Coord is Unknown."
    )
    valchecker(
        pred.shape == test.shape,
        "Shapes of pred and test should be mutual."
    )
    valchecker(
        np.all(pred.x == test.x),
        "(x, y) coodinates should be mutual with pred and test."
    )
    valchecker(
        np.all(pred.y == test.y),
        "(x, y) coodinates should be mutual with pred and test."
    )
