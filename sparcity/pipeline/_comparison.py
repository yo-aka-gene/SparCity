"""
Class for model comparison in a pipeline
"""
from typing import Callable, List, Union

import numpy as np
import tqdm

from sparcity.core import GPModel, TrainData, TestData
from sparcity.dev import typechecker, valchecker
from sparcity.metrics import mean_absolute_error_score as mae
from ._init_args import init_args


class ComparisonPipeline:
    """
    Class for model comparison in a pipeline

    Methods
    -------
    __init__() -> None:
        initialize attributes listed below

    compare(logging: bool = False) -> None:
        train, predict, and evaluate the models in a row

    Attributes
    ----------
    models: List[GPModel]
        list of GPModels to be compared in the pipeline

    test: TestData
        test data for all models

    predicted: List[Coord], default: None
        list of predicted `Coord`s (to be set in `self.compare`)

    metric: Callable
        evaluation metric

    n_iter: Union[int, numpy.int64]
        number of iterations in optimization (per model)

    scores: List[float], default: None
        list of scores of the evaluation metric (to be set in `self.compare`)
    """
    def __init__(
        self,
        traindata: List[TrainData],
        testdata: TestData,
        gpmodel: GPModel,
        metric: Callable = mae,
        n_iter: Union[int, np.int64] = 100
    ) -> None:
        """
        Parameters
        ----------
        traindata: List[TrainData]
            list of training data

        testdata: TestData
            test data

        gpmodel: type
            type for a subclass of `GPModel`

        metric: Callable, default: sparcity.metrics.mean_absolute_error_score
            evaluation metric that takes `pred` and `test` as the first two arguments

        n_iter: Union[int, numpy.int64], default: 100
            number of iterations in optimization (per model)

        Examples
        --------
        >>> import numpy as np
        >>> from sparcity import GPModel, TestData
        >>> from sparcity.dataset import QuadraticGenerator
        >>> from sparcity.gaussian_process import KISSGPRegression
        >>> from sparcity.metrics import mean_absolute_error_score as mae
        >>> from sparcity.pipeline import ComparisonPipeline
        >>> from sparcity.sampler import RandomInGrid
        >>> area = QuadraticGenerator()
        >>> train = [RandomInGrid(x, y, l, area) for x, y, l in zip([2, 10], [2, 10], [5, 3])]
        >>> test = TestData(area)
        >>> pipe = ComparisonPipeline(train, test, KISSGPRegression)
        >>> isinstance(pipe.models, list)
        True
        >>> np.all([isinstance(v, GPModel) for v in pipe.models])
        True
        >>> np.all([isinstance(v, KISSGPRegression) for v in pipe.models])
        True
        >>> isinstance(pipe.test, TestData)
        True
        >>> # default value for self.predicted is None
        >>> pipe.predicted is None
        True
        >>> pipe.metric == mae
        True
        >>> pipe.n_iter
        100
        >>> # default value for self.scores is None
        >>> pipe.scores is None
        True
        """
        typechecker(traindata, list, "traindata")
        for i, v in enumerate(traindata):
            typechecker(v, TrainData, f"traindata[{i}]")
        typechecker(testdata, TestData, "testdata")
        typechecker(gpmodel, type, "gpmodel")
        typechecker(metric, Callable, "metric")
        args = metric.__code__.co_varnames[:metric.__code__.co_argcount]
        valchecker(
            args[:2] == ("pred", "test"),
            "Arguments of metric should be `pred` and `test` (+ additional args)."
        )
        typechecker(n_iter, (int, np.int64), "n_iter")
        self.models = [gpmodel(traindata=train) for train in traindata]
        typechecker(self.models[0], GPModel, "class of gpmodel")
        self.test = testdata
        self.predicted = None
        self.metric = metric
        self.n_iter = n_iter
        self.scores = None

    def compare(self, logging: bool = False) -> None:
        """
        Parameters
        ----------
        logging: bool, default: False
            if `True` a progress bar is shown

        Examples
        --------
        >>> import numpy as np
        >>> from sparcity import Coord, TestData
        >>> from sparcity.dataset import QuadraticGenerator
        >>> from sparcity.gaussian_process import KISSGPRegression
        >>> from sparcity.pipeline import ComparisonPipeline
        >>> from sparcity.sampler import RandomInGrid
        >>> area = QuadraticGenerator(xrange=np.linspace(0, 100, 50), yrange=np.linspace(0, 100, 50))
        >>> train = [RandomInGrid(x, y, l, area) for x, y, l in zip([2, 10], [2, 10], [5, 3])]
        >>> test = TestData(area)
        >>> pipe = ComparisonPipeline(train, test, KISSGPRegression)
        >>> pipe.compare()
        >>> # self.predicted is calculated and updated
        >>> isinstance(pipe.predicted, list)
        True
        >>> np.all([isinstance(v, Coord) for v in pipe.predicted])
        True
        >>> # self.scores is calculated and updated
        >>> isinstance(pipe.scores, list)
        True
        >>> np.all([isinstance(v, float) for v in pipe.scores])
        True
        """
        self.predicted = []
        self.scores = []
        init_dicts = [init_args(model) for model in self.models]
        iterator = zip(self.models, init_dicts)
        iterator = tqdm.tqdm(
            iterator,
            desc="Training/Evaluating Models",
            total=len(self.models)
        ) if logging else iterator
        for model, kwargs in iterator:
            model.initialize(**kwargs)
            model.fit(n_iter=self.n_iter, logging=logging)
            pred = model.predict(self.test)
            self.predicted += [pred]
            self.scores += [self.metric(pred=pred, test=self.test.field)]

