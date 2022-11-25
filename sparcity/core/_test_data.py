"""
Class of test dataset
"""
from typing import Tuple

import numpy as np
import pandas as pd
import torch as t

from sparcity.dev import typechecker
from ._coord import Coord
from ._unknown_coord import UnknownCoord


class TestData:
    """
    Class of Test Data

    Methods
    -------
    __init__(field: Coord) -> None:
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

    is_evaluable: bool
        False iif the dtype of `field` is UnknownCoord, otherwise True.

    shape: Tuple[int]
        shape of `field`
    """
    def __init__(
        self,
        field: Coord,
    ) -> None:
        """
        Parameters
        ----------
        field: Union[Coord, UnknownCoord]]
            Coord or UnknownCoord of the target areas to predict in Gaussian Process Regression

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from sparcity import Coord, TestData, UnknownCoord
        >>> x1 = np.array([0, 1, 2])
        >>> y1 = np.array([3, 4, 5])
        >>> z1 = np.array([[ 6,  7,  8], [ 9, 10, 11], [12, 13, 14]])
        >>> area1 = Coord(x=x1, y=y1, z=z1)
        >>> input1 = pd.DataFrame({\
            "longitude": area1.x_2d.ravel(), \
            "latitude": area1.y_2d.ravel()\
            })
        >>> output1 = pd.DataFrame({\
            "altitude": area1.z.ravel()\
            })
        >>> test1 = TestData(area1)
        >>> np.all(test1.input == input1)
        True
        >>> np.all(test1.output == output1)
        True
        >>> test1.is_evaluable
        True
        >>> test1.shape
        (3, 3)
        >>> x2 = np.array([2, 3])
        >>> y2 = np.array([5, 6, 7])
        >>> area2 = UnknownCoord(x=x2, y=y2)
        >>> input2 = pd.DataFrame({\
            "longitude": area2.x_2d.ravel(), \
            "latitude": area2.y_2d.ravel()\
            })
        >>> output2 = pd.DataFrame({\
            "altitude": area2.z.ravel()\
            })
        >>> np.all(np.isnan(output2))
        True
        >>> test2 = TestData(area2)
        >>> np.all(test2.input == input2)
        True
        >>> np.all(np.isnan(test2.output))
        True
        >>> test2.is_evaluable
        False
        >>> test2.shape
        (3, 2)
        """
        typechecker(field, (Coord, UnknownCoord), "field")
        self.input = pd.DataFrame(
            {
                "longitude": field.x_2d.ravel(),
                "latitude": field.y_2d.ravel()
            }
        ).drop_duplicates()
        self.output = pd.DataFrame(
            {
                "altitude": field.z.ravel()
            }
        ).loc[self.input.index, :]

        self.is_evaluable = not isinstance(field, UnknownCoord)

        self.shape = field.shape

    def as_ndarray(self) -> Tuple[np.array]:
        """
        Returns
        -------
        (i, o): Tuple[numpy.ndarray]
            tuple of numpy.ndarray regarding input and output vectors

        Examples
        --------
        >>> import numpy as np
        >>> from sparcity import Coord, TestData, UnknownCoord
        >>> x1 = np.array([0, 1, 2])
        >>> y1 = np.array([3, 4, 5])
        >>> z1 = np.array([[ 6,  7,  8], [ 9, 10, 11], [12, 13, 14]])
        >>> area1 = Coord(x=x1, y=y1, z=z1)
        >>> test1 = TestData(area1)
        >>> ret1 = test1.as_ndarray()
        >>> isinstance(ret1, tuple)
        True
        >>> len(ret1) == 2
        True
        >>> isinstance(ret1[0], np.ndarray)
        True
        >>> isinstance(ret1[1], np.ndarray)
        True
        >>> np.all(test1.input.values == ret1[0])
        True
        >>> np.all(test1.output.values.ravel() == ret1[1])
        True
        >>> x2 = np.array([2, 3])
        >>> y2 = np.array([5, 6, 7])
        >>> area2 = UnknownCoord(x=x2, y=y2)
        >>> test2 = TestData(area2)
        >>> np.all(np.isnan(test2.output))
        True
        >>> ret2 = test2.as_ndarray()
        >>> isinstance(ret2, tuple)
        True
        >>> len(ret2) == 2
        True
        >>> isinstance(ret2[0], np.ndarray)
        True
        >>> isinstance(ret2[1], np.ndarray)
        True
        >>> np.all(test2.input.values == ret2[0])
        True
        >>> np.all(np.isnan(ret2[1]))
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
        >>> from sparcity import Coord, TestData, UnknownCoord
        >>> x1 = np.array([0, 1, 2])
        >>> y1 = np.array([3, 4, 5])
        >>> z1 = np.array([[ 6,  7,  8], [ 9, 10, 11], [12, 13, 14]])
        >>> area1 = Coord(x=x1, y=y1, z=z1)
        >>> test1 = TestData(area1)
        >>> ret1 = test1.as_tensor()
        >>> isinstance(ret1, tuple)
        True
        >>> len(ret1) == 2
        True
        >>> isinstance(ret1[0], t.Tensor)
        True
        >>> isinstance(ret1[1], t.Tensor)
        True
        >>> tensorx1 = t.tensor(test1.input.values, dtype=t.float32)
        >>> t.all(tensorx1 == ret1[0])
        tensor(True)
        >>> tensory1 = t.tensor(test1.output.values.ravel(), dtype=t.float32)
        >>> t.all(tensory1 == ret1[1])
        tensor(True)
        >>> x2 = np.array([2, 3])
        >>> y2 = np.array([5, 6, 7])
        >>> area2 = UnknownCoord(x=x2, y=y2)
        >>> test2 = TestData(area2)
        >>> np.all(np.isnan(test2.output))
        True
        >>> ret2 = test2.as_tensor()
        >>> isinstance(ret2, tuple)
        True
        >>> len(ret2) == 2
        True
        >>> isinstance(ret2[0], t.Tensor)
        True
        >>> isinstance(ret2[1], t.Tensor)
        True
        >>> tensorx2 = t.tensor(test2.input.values, dtype=t.float32)
        >>> t.all(tensorx2 == ret2[0])
        tensor(True)
        >>> t.all(t.isnan(ret2[1]))
        tensor(True)
        """
        return (
            t.tensor(self.input.values, dtype=t.float32),
            t.tensor(self.output.values.ravel(), dtype=t.float32)
        )
