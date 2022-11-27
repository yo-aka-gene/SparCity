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
        1d-array that is tested to be a subarray of array2.
        arr1 should be monotonically increasing.

    arr2: numpy.ndarray
        1d-array that is tested to be a superarray of array1
        arr2 should be monotonically increasing.

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
    valchecker(arr1.ndim == 1)
    valchecker(arr2.ndim == 1)
    valchecker(is_monotonical_increasing(arr1))
    valchecker(is_monotonical_increasing(arr2))
    ret = arr1.size <= arr2.size
    ret &= arr1.min() in arr2
    ret &= arr1.max() in arr2
    if ret:
        m = np.where(arr2 == arr1.min())[0].item()
        M = np.where(arr2 == arr1.max())[0].item()
        ret &= np.all(arr1 == arr2[m:M + 1])
    return ret


def is_monotonical_increasing(arr: np.ndarray) -> bool:
    """
    function to check if the array is monotonically increasing

    Parameters
    ----------
    arr: numpy.ndarray
        1d-array

    Examples
    --------
    >>> import numpy as np
    >>> from sparcity.dev import is_monotonical_increasing as ism
    >>> x = np.arange(5)
    >>> ism(x)
    True
    >>> y = np.arange(4, -2, -1)
    >>> ism(y)
    False
    >>> z = np.array([0, 1, 1, 2])
    >>> ism(z)
    False
    """
    typechecker(arr, np.ndarray, "arr")
    valchecker(len(arr.shape) == 1)
    return np.all(np.diff(arr) > 0)


def get_subarray_loc(
    subarr: np.ndarray,
    arr: np.ndarray
) -> np.ndarray:
    """
    function to get locations of subarray

    Parameters
    ----------
    subarr: np.ndarray
        subarray of arr

    arr: np.ndarray
        superarray of array

    Returns
    -------
    loc: np.ndarray
        location of subarray in superarray

    Examples
    --------
    >>> import numpy as np
    >>> from sparcity.dev import get_subarray_loc
    >>> arr1 = np.arange(10)
    >>> subarr = np.arange(3, 5)
    >>> get_subarray_loc(subarr, arr1)
    array([3, 4])
    >>> arr2 = np.arange(3, 10)
    >>> get_subarray_loc(subarr, arr2)
    array([0, 1])
    """
    valchecker(is_subarray(subarr, arr))
    m = np.where(arr == subarr.min())[0].item()
    M = np.where(arr == subarr.max())[0].item()
    return np.arange(m, M + 1)
