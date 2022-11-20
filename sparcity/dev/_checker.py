from typing import Any, Tuple, Union


def typechecker(
    arg: Any,
    types: Union[Tuple[type], type],
    varname: str
) -> None:
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
        f"Invalid dtype for varname; expected {type_msg}, got {arg}[{type(arg)}]"


def valchecker(
    condition: bool
) -> None:
    typechecker(condition, bool, "condition")
    assert condition, \
        "Invalid value detected. Check the requirements"
