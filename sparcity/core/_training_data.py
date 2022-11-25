"""
Class of training dataset
"""
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch as t

from sparcity.dev import typechecker
from ._coord import Coord


class TrainData:
    """
    Class of Training Data

    Methods
    -------
    __init__(coordinates: Union[Coord, List[Coord]]) -> None:
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
    """
    def __init__(
        self,
        coordinates: Union[Coord, List[Coord]]
    ) -> None:
        """
        Parameters
        ----------
        coordinates: Union[Coord, List[Coord]]
            Coord or list of Coord (expected sampled subareas in Gaussian Process Regression)

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from sparcity import Coord, TrainData
        >>> x1 = np.array([0, 1, 2])
        >>> y1 = np.array([3, 4, 5])
        >>> z1 = np.array([[ 6,  7,  8], [ 9, 10, 11], [12, 13, 14]])
        >>> area1 = Coord(x=x1, y=y1, z=z1)
        >>> input1 = pd.DataFrame({\
            "longitude": area1.x_2d.ravel(),\
            "latitude": area1.y_2d.ravel()\
            })
        >>> output1 = pd.DataFrame({\
            "altitude": area1.z.ravel()\
            })
        >>> train1 = TrainData(area1)
        >>> np.all(train1.input == input1)
        True
        >>> np.all(train1.output == output1)
        True
        >>> x2 = np.array([2, 3])
        >>> y2 = np.array([5, 6])
        >>> z2 = np.array([[14, 15], [16, 17]])
        >>> area2 = Coord(x=x2, y=y2, z=z2)
        >>> input2 = pd.DataFrame({\
            "longitude": np.append(*[v.x_2d.ravel() for v in [area1, area2]]),\
            "latitude": np.append(*[v.y_2d.ravel() for v in [area1, area2]])\
            }).drop_duplicates()
        >>> output2 = pd.DataFrame({\
            "altitude": np.append(*[v.z.ravel() for v in [area1, area2]])\
            }).loc[input2.index, :]
        >>> train2 = TrainData([area1, area2])
        >>> np.all(train2.input == input2)
        True
        >>> np.all(train2.output == output2)
        True
        """
        typechecker(coordinates, (Coord, list), "coordinates")
        if isinstance(coordinates, list):
            for i, v in enumerate(coordinates):
                typechecker(v, Coord, f"coordinates[{i}]")
        else:
            coordinates = [coordinates]
        self.input = pd.DataFrame(
            {
                "longitude": np.concatenate(
                    [v.x_2d.ravel() for v in coordinates],
                    axis=0
                ),
                "latitude": np.concatenate(
                    [v.y_2d.ravel() for v in coordinates],
                    axis=0
                )
            }
        ).drop_duplicates()
        self.output = pd.DataFrame(
            {
                "altitude": np.concatenate(
                    [v.z.ravel() for v in coordinates],
                    axis=0
                )
            }
        ).loc[self.input.index, :]

    def as_ndarray(self) -> Tuple[np.array]:
        """
        Returns
        -------
        (i, o): Tuple[numpy.ndarray]
            tuple of numpy.ndarray regarding input and output vectors

        Examples
        --------
        >>> import numpy as np
        >>> from sparcity import Coord, TrainData
        >>> x1 = np.array([0, 1, 2])
        >>> y1 = np.array([3, 4, 5])
        >>> z1 = np.array([[ 6,  7,  8], [ 9, 10, 11], [12, 13, 14]])
        >>> area1 = Coord(x=x1, y=y1, z=z1)
        >>> x2 = np.array([2, 3])
        >>> y2 = np.array([5, 6])
        >>> z2 = np.array([[14, 15], [16, 17]])
        >>> area2 = Coord(x=x2, y=y2, z=z2)
        >>> train = TrainData([area1, area2])
        >>> ret = train.as_ndarray()
        >>> isinstance(ret, tuple)
        True
        >>> len(ret) == 2
        True
        >>> isinstance(ret[0], np.ndarray)
        True
        >>> isinstance(ret[1], np.ndarray)
        True
        >>> np.all(train.input.values == ret[0])
        True
        >>> np.all(train.output.values.ravel() == ret[1])
        True
        """
        return (
            self.input.values,
            self.output.values.ravel()
        )

    def as_tensor(self) -> Tuple[t.Tensor]:
        """
        Returns
        -------
        (i, o): Tuple[torch.Tensor]
            tuple of torch.Tensor regarding input and output vectors

        Examples
        --------
        >>> import numpy as np
        >>> import torch as t
        >>> from sparcity import Coord, TrainData
        >>> x1 = np.array([0, 1, 2])
        >>> y1 = np.array([3, 4, 5])
        >>> z1 = np.array([[ 6,  7,  8], [ 9, 10, 11], [12, 13, 14]])
        >>> area1 = Coord(x=x1, y=y1, z=z1)
        >>> x2 = np.array([2, 3])
        >>> y2 = np.array([5, 6])
        >>> z2 = np.array([[14, 15], [16, 17]])
        >>> area2 = Coord(x=x2, y=y2, z=z2)
        >>> train = TrainData([area1, area2])
        >>> ret = train.as_tensor()
        >>> isinstance(ret, tuple)
        True
        >>> len(ret) == 2
        True
        >>> isinstance(ret[0], t.Tensor)
        True
        >>> isinstance(ret[1], t.Tensor)
        True
        >>> tensorx = t.tensor(train.input.values, dtype=t.float32)
        >>> t.all(tensorx == ret[0])
        tensor(True)
        >>> tensory = t.tensor(train.output.values.ravel(), dtype=t.float32)
        >>> t.all(tensory == ret[1])
        tensor(True)
        """
        return (
            t.tensor(self.input.values, dtype=t.float32),
            t.tensor(self.output.values.ravel(), dtype=t.float32)
        )
