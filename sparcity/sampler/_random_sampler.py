"""
Class for random sampling
"""
from typing import Tuple, Union

import numpy as np
import torch as t

from sparcity.core import Coord, patch, subarea, TrainData
from sparcity.dev import typechecker, valchecker
from sparcity.random import random_coord, seedfixer


class RandomSampler(TrainData):
    """
    Class for random sampling

    Methods
    -------
    __init__(
        n_samples: Union[int, numpy.int64],
        length: Union[int, numpy.int64, float, numpy.float64],
        field: Coord,
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
        n_samples: Union[int, np.int64],
        length: Union[int, np.int64, float, np.float64],
        field: Coord,
        seed: Union[int, np.int64] = 0
    ) -> None:
        """
        Parameters
        ----------
        n_samples: Union[int, numpy.int64]
            number of samples. Expected to be a positive integer

        length: Union[int, numpy.int64, float, numpy.float64]
            length of the side of the sampling squares

        field: Coord
            Coord class of the original field

        seed: Union[int, numpy.int64], default: 0
            random seed

        Examples
        --------
        >>> import sparcity as spc
        >>> from sparcity.sampler import RandomSampler
        >>> field = spc.dataset.QuadraticGenerator()
        >>> train = RandomSampler(n_samples=10, length=5, field=field, seed=123)
        >>> isinstance(train, spc.TrainData)
        True
        >>> isinstance(train.patch, Coord)
        True
        """
        typechecker(n_samples, (int, np.int64), "n_samples")
        typechecker(length, int, "length")
        typechecker(field, Coord, "field")
        typechecker(seed, (int, np.int64), "seed")
        valchecker(n_samples > 0)

        coords = []
        for i in range(n_samples):
            seedfixer(seed, n_samples, i)
            sampling_seed = np.random.randint(0, 2 ** 32 - 1, 1).item()
            coords += [
                    subarea(
                        *random_coord(
                            field,
                            sampling_seed
                        ),
                        length=length,
                        field=field
                    )
                ]

        self.input = TrainData(coords).input
        self.output = TrainData(coords).output
        self.patch = patch(coords, field)

    def as_ndarray(self) -> Tuple[np.array]:
        return super().as_ndarray()

    def as_tensor(self) -> Tuple[t.Tensor]:
        return super().as_tensor()
