"""
Class of KISS-GP Regression
"""
import gpytorch as gpt
from gpytorch.likelihoods import GaussianLikelihood
import torch as t
from tqdm import tqdm

from sparcity.core import Coord, GPModel, TrainData, TestData


class KISSGPRegression(GPModel):
    """
    Class of KISS-GP Regression

    Methods
    -------
    __init__(
        traindata: TrainData,
        kernel: gpytorch.kernels.Kernel,
        likelihood: gpytorch.likelihoods.Likelihood,
        mean: gpytorch.means.Mean
    ) -> None:
        initialize attributes listed below

    forward(x: torch.Tensor) -> gpytorch.distributions.MultivariateNornal
        Update parameters in the multivariate normal distribution

    fit(
        optimizer: str,
        lr: float,
        n_iter: int,
        **kwargs
    ) -> None:
        train model & likelihood + optimize hyperparameters
         + set model & likelihood into evaluation mode

    predict() -> Coord:
        Predict coordinates from testdata

    Attributes
    ----------
    train_x: torch.Tensor
        input training data

    train_y: torch.Tensor
        output values for training data

    likelihood: gpytorch.likelihoods.Likelihood
        likelihood function

    mean_module: gpytorch.means.Mean
        mean module for gpytorch.models.ExactGP

    covar_module: gpytorch.kernels.ScaleKernel
        kernel function for gpytorch.models.ExactGP
    """
    def __init__(
        self,
        traindata: TrainData,
        kernel: gpt.kernels.Kernel = gpt.kernels.RBFKernel(),
        likelihood: gpt.likelihoods.Likelihood = GaussianLikelihood(),
        mean: gpt.means.Mean = gpt.means.ConstantMean()
    ) -> None:
        """
        Parameters
        ----------
        traindata: TrainData
            training data

        kernel: gpytorch.kernels.Kernel, default: gpytorch.kernels.RBFKernel()
            kernel function to use inside gpytorch.kernels.GridInterpolationKernel

        likelihood: gpytorch.likelihoods.Likelihood, default: gpytorch.likelihoods.GaussianLikelihood()
            likelihood function

        mean: gpytorch.means.Mean, default: gpytorch.means.ConstantMean()
        """
        super(KISSGPRegression, self).__init__(
            traindata=traindata,
            kernel=kernel,
            likelihood=likelihood,
            mean=mean
        )

        grid_size = gpt.utils.grid.choose_grid_size(traindata.as_tensor()[0])
        self.covar_module = gpt.kernels.ScaleKernel(
            gpt.kernels.GridInterpolationKernel(
                kernel, grid_size=grid_size, num_dims=2
            )
        )

    def forward(
        self,
        x: t.Tensor
    ) -> gpt.distributions.MultivariateNormal:
        """
        function to update parameters in the multivariate normal distribution

        Parameters
        ----------
        x: torch.Tensor
            input tensor

        Returns
        -------
        Norm: gpytorch.distributions.MultivariateNormal
        """
        return super().forward(x=x)

    def fit(
        self,
        optimizer: str = "Adam",
        lr: float = 0.1,
        n_iter: int = 30,
        logging: bool = False,
        **kwargs
    ) -> None:
        """
        function to train model & likelihood, learn hyperparameters,
        and switch model & likelihood into evaluation mode

        Parameters
        ----------
        optimizer: str, default: "Adam"
            name of the optimizer. Expected to be "Adadelta", "Adagrad", "Adam", "AdamW",
            "SparseAdam", "Adamax", "ASGD", "LBFGS", "NAdam", "RAdam", "RMSprop",
            "Rprop", or "SGD".

        lr: float, default: 0.1
            learning rate

        n_iter: int, default: 30
            number of iterations for hyperparameter tuning

        logging: bool, default: False
            if `True` a progress bar is shown
        """
        super().fit(
            optimizer=optimizer,
            lr=lr,
            n_iter=n_iter,
            logging=logging,
            **kwargs
        )

    def predict(
        self,
        testdata: TestData
    ) -> Coord:
        """
        function to predict coordinates from testdata

        Parameters
        ----------
        testdata: TestData
            testdata for prediction

        Returns
        -------
        PredictedCoord: Coord
            Coord class for predicted coordinates
        """
        return super().predict(testdata=testdata)
