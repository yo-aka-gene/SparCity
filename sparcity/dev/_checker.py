"""
functions for checking arguments
"""
from typing import Any, Tuple, Union

import numpy as np


def typechecker(
    arg: Any,
    types: Union[Tuple[type], type],
    varname: str
) -> None:
    """
    dtype checker for functions
    raises AssertionError whin the dtype is inappropriate

    Parameters
    ----------
    arg: Any
        argument of the function

    types: Union[Tuple[type], type]
        a tuple of acceptable dtypes or a single dtype

    varname: str
        the name of variable

    Raises
    ------
        AssertionError
            It automatically evaluates dtype of arg and raises AssertionError
            when `isinstance(arg, types)` is False

    Examples
    --------
    >>> import numpy as np
    >>> from sparcity.dev import typechecker
    >>> typechecker("a", str, "alphabet")

    >>> typechecker(0.5, np.float64, "Q")
    Traceback (most recent call last):
        ...
    AssertionError: Invalid dtype for Q; expected <class 'numpy.float64'>, got 0.5[<class 'float'>]
    >>> typechecker(0.5, (float, np.float64), "Q")

    """
    assert isinstance(types, (type, tuple)), \
        f"Invalid definition of argument dtype, {types}. Please assign type or tuple of types"
    if isinstance(types, tuple):
        for v in types:
            assert isinstance(v, type), \
                f"Invalid definition of argument dtype, {v} in {types}."
        maxiter = len(types) - 1
        type_msg = "".join([
            f"{v}, " if i != maxiter else f"or {v}" for i, v in enumerate(types)
        ])
    else:
        type_msg = f"{types}"
    assert isinstance(varname, str), \
        f"Invalid dtype for varname; expected str, got {varname}"
    assert isinstance(arg, types), \
        f"Invalid dtype for {varname}; expected {type_msg}, got {arg}[{type(arg)}]"


def valchecker(
    condition: Union[bool, np.bool_],
    suffix: str = ""
) -> None:
    """
    value checker for functions
    raises AssertionError whin the required condition is not fulfilled

    Parameters
    ----------
    condition: Union[bool, numpy.bool_]
        required condition

    suffix: str, default: ""
        additional comments for the error message if needed

    Raises
    ------
        AssertionError
            It raises AssertionError when condition is False

    Examples
    --------
    >>> from sparcity.dev import valchecker
    >>> valchecker(isinstance("a", str))

    >>> x = 5
    >>> valchecker(x < 10, suffix="I like chocolate!!!")

    >>> valchecker(x > 10)
    Traceback (most recent call last):
        ...
    AssertionError: Invalid value detected. Check the requirements.
    >>> valchecker(x > 10, suffix="I like chocolate!!!")
    Traceback (most recent call last):
        ...
    AssertionError: Invalid value detected. Check the requirements. I like chocolate!!!
    """
    typechecker(condition, (bool, np.bool_), "condition")
    typechecker(suffix, str, "suffix")
    assert condition, \
        f"Invalid value detected. Check the requirements.{' ' + suffix if suffix != '' else suffix}"
