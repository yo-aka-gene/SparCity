"""
Numpy extension functions
"""
import numpy as np

from ._checker import typechecker, valchecker


def is_subarray(arr1: np.ndarray, arr2: np.ndarray) -> bool:
    """
    function to check if the array1 is a subarray of array2

    Parameters
    ----------
    arr1: numpy.ndarray
        1d-array that is tested to be a subarray of array2

    arr2: numpy.ndarray
        1d-array that is tested to be a superarray of array1

    Returns
    -------
    Result: bool

    Examples
    --------
    >>> import numpy as np
    >>> from sparcity.dev import is_subarray
    >>> x1 = np.arange(3)
    >>> x2 = np.linspace(0, 3, 100)
    >>> y = np.arange(10)
    >>> is_subarray(x1, y)
    True
    >>> is_subarray(y, y)
    True
    >>> is_subarray(y, x1)
    False
    >>> is_subarray(x2, y)
    False
    >>> is_subarray(y, x2)
    False
    """
    typechecker(arr1, np.ndarray, "arr1")
    typechecker(arr2, np.ndarray, "arr2")
    valchecker(len(arr1.shape) == 1)
    valchecker(len(arr2.shape) == 1)
    ret = len(arr1) <= len(arr2)
    ret &= arr1.min() in arr2
    ret &= arr1.max() in arr2
    if ret:
        m = np.where(arr2 == arr1.min())[0].item()
        M = np.where(arr2 == arr1.max())[0].item()
        ret &= np.all(arr1 == arr2[m:M + 1])
    return ret
