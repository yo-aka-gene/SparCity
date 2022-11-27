"""
Class of KISS-GP Regression
"""
import gpytorch as gpt
from gpytorch.likelihoods import GaussianLikelihood
import torch as t
from tqdm import tqdm

from sparcity.core import Coord, TrainData, TestData
from sparcity.dev import typechecker, valchecker


class KISSGPRegression(gpt.models.ExactGP):
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
        typechecker(traindata, TrainData, "traindata")
        typechecker(kernel, gpt.kernels.Kernel, "kernel")
        typechecker(likelihood, gpt.likelihoods.Likelihood, "likelihood")
        typechecker(mean, gpt.means.Mean, "mean")
        super(KISSGPRegression, self).__init__(
            *traindata.as_tensor(),
            likelihood
        )

        grid_size = gpt.utils.grid.choose_grid_size(traindata.as_tensor()[0])

        self.train_x, self.train_y = traindata.as_tensor()
        self.likelihood = likelihood
        self.mean_module = mean
        self.covar_module = gpt.kernels.ScaleKernel(
            gpt.kernels.GridInterpolationKernel(
                kernel, grid_size=grid_size, num_dims=2
            )
        )
        self._progress = (False, False)

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
        typechecker(x, t.Tensor, "x")
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpt.distributions.MultivariateNormal(mean_x, covar_x)

    def _optimize_hyperparams(
        self,
        optimizer: str = "Adam",
        lr: float = 0.1,
        n_iter: int = 30,
        **kwargs
    ) -> None:
        """
        latent funtion to optimize hyperparams.
        Expected to be called within the scope of self.fit()

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
        """
        typechecker(optimizer, str, "optimizer")
        typechecker(lr, float, "lr")
        typechecker(n_iter, int, "n_iter")
        mode = [
            "Adadelta", "Adagrad", "Adam", "AdamW", "SparseAdam",
            "Adamax", "ASGD", "LBFGS", "NAdam", "RAdam",
            "RMSprop", "Rprop", "SGD"
        ]
        valchecker(optimizer in mode, f"Choose from {mode}.")
        valchecker(lr > 0)
        valchecker(n_iter > 0)
        valchecker(self._progress == (True, False), "Train model and likelihood first.")

        optimizer = eval(
            f"t.optim.{optimizer}"
        )(self.parameters(), lr=lr)

        mll = gpt.mlls.ExactMarginalLogLikelihood(
            self.likelihood,
            self
        )

        for i in tqdm(
            range(n_iter),
            desc="Optimization",
            total=n_iter
        ):
            optimizer.zero_grad()
            loss = -mll(self(self.train_x), self.train_y)
            loss.backward()
            optimizer.step()

    def fit(
        self,
        optimizer: str = "Adam",
        lr: float = 0.1,
        n_iter: int = 30,
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
        """
        valchecker(self._progress == (False, False), "Do not `fit` more than once.")
        self.train()
        self.likelihood.train()
        self._progress = (True, False)
        self._optimize_hyperparams(
            optimizer=optimizer,
            lr=lr,
            n_iter=n_iter,
            kwargs=kwargs
        )
        self.eval()
        self.likelihood.eval()
        self._progress = (True, True)

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
        typechecker(testdata, TestData, "testdata")
        valchecker(self._progress == (True, True), "Run `fit` first.")

        with t.no_grad(), gpt.settings.fast_pred_var():
            predicted = self.likelihood(
                self(testdata.as_tensor()[0])
            ).mean.view(*testdata.field.shape)

        return Coord(
            x=testdata.x,
            y=testdata.y,
            z=predicted.numpy()
        )
