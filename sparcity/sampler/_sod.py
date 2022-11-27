"""
function to get a seubset of the dataset
"""
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch as t

from sparcity.core import Coord, TrainData
from sparcity.dev import typechecker, valchecker


class _SubsetOfData(TrainData):
    """
    Class for subset of training data
    Expected to be called via functions to get SOD

    Methods
    -------
    __init__(

        seed: Union[int, numpy.int64] = 0
    ) -> None:
        initialize attributes listed below.

    as_ndarray() -> Tuple[numpy.ndarray]:
        returns (i, o) where i is a numpy.ndarray for input data and
        o is a numpy.ndarray for the output values

    as_tensor() -> Tuple[torch.Tensor]:
        returns (i, o) where i is a torch.tensor for input data and
        o is a torch.tensor for the output values

    Attributes
    ----------
    input: pandas.core.frame.DataFrame
        DataFrame of input data. Duplicated values are removed.

    output: pandas.core.frame.DataFrame
        DataFrame of output data. Corresponding output for the duplicated inputs
        are removed.

    patch: Coord
        Coords of the input data
    """
    def __init__(
        self,
        input: pd.core.frame.DataFrame,
        output: pd.core.frame.DataFrame,
        patch: Coord
    ) -> None:
        typechecker(input, pd.core.frame.DataFrame, "input")
        typechecker(output, pd.core.frame.DataFrame, "output")
        typechecker(patch, Coord, "patch")
        valchecker(np.any(np.isnan(patch.z)))
        valchecker(np.all(input.index == output.index))
        self.input = input
        self.output = output
        self.patch = patch

    def as_ndarray(self) -> Tuple[np.array]:
        return super().as_ndarray()

    def as_tensor(self) -> Tuple[t.Tensor]:
        return super().as_tensor()


def random_subset_of_data(
        train: TrainData,
        frac: Union[float, np.float64],
        seed: Union[int, np.int64] = 0
) -> _SubsetOfData:
    """
    function to get a random subset of dataset
    used for SoD method

    Parameters
    ----------
    train: TrainData
        TrainData class for the full training dataset

    frac: Union[float, np.int64]
        Fraction of subset items. Expected to be 0 < datasize < 1.

    seed: Union[int, np.int64], default: 0
        random seed

    Returns
    -------
    Subset: _SubsetOfData
        random subset of data

    Examples
    --------
    >>> import numpy as np
    >>> from sparcity import Coord, TrainData
    >>> from sparcity.dataset import QuadraticGenerator
    >>> from sparcity.sampler import RandomSampler
    >>> from sparcity.sampler import random_subset_of_data
    >>> field = QuadraticGenerator()
    >>> train = RandomSampler(n_samples=10, length=5, field=field, seed=0)
    >>> sod = random_subset_of_data(train=train, frac=0.1, seed=123)
    >>> isinstance(sod, TrainData)
    True
    >>> type(sod.input)
    <class 'pandas.core.frame.DataFrame'>
    >>> sod.input.columns == ["longitude", "latitude"]
    array([ True,  True])
    >>> type(sod.output)
    <class 'pandas.core.frame.DataFrame'>
    >>> sod.output.columns == ["altitude"]
    array([ True])
    >>> isinstance(sod.patch, Coord)
    True
    >>> np.all(train.patch.x == sod.patch.x)
    True
    >>> np.all(train.patch.y == sod.patch.y)
    True
    >>> np.any(np.isnan(sod.patch.z))
    True
    >>> np.any(~np.isnan(sod.patch.z))
    True
    >>> np.any(train.patch.z == sod.patch.z)
    True
    >>> train.patch.shape == sod.patch.shape
    True
    """
    typechecker(train, TrainData, "train")
    typechecker(frac, (float, np.float64), "frac")
    typechecker(seed, (int, np.int64), "seed")
    valchecker(0 < frac < 1)
    rows, cols = np.where(~np.isnan(train.patch.z))
    size = int(rows.size * frac)
    valchecker(size > 0, "Consider a larger `frac` value.")

    np.random.seed(seed)
    subrow = np.random.choice(rows, size=size, replace=False)
    np.random.seed(seed)
    subcol = np.random.choice(cols, size=size, replace=False)

    z = np.zeros(train.patch.z.shape)
    z = np.where(z == 0, np.nan, np.nan)
    z[subrow, subcol] = train.patch.z[subrow, subcol]

    longitude = train.patch.x[subcol]
    latitude = train.patch.y[subrow]

    return _SubsetOfData(
        input=pd.DataFrame(
            {
                "longitude": longitude,
                "latitude": latitude
            }
        ),
        output=pd.DataFrame(
            {
                "altitude": z[subrow, subcol]
            }
        ),
        patch=Coord(
            x=train.patch.x,
            y=train.patch.y,
            z=z
        )
    )
