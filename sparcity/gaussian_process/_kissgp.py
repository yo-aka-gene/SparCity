"""
Class of KISS-GP Regression
"""
import gpytorch as gpt
from gpytorch.likelihoods import GaussianLikelihood
import torch as t
from tqdm import tqdm

from sparcity.core import TrainData, TestData
from sparcity.dev import typechecker, valchecker

class KISSGPRegression(gpt.models.ExactGP):
    """
    Class of KISS-GP Regression
    """
    def __init__(
        self,
        traindata: TrainData,
        kernel: gpt.kernels.Kernel = gpt.kernels.RBFKernel(),
        likelihood: gpt.likelihoods.Likelihood = GaussianLikelihood(),
        mean: gpt.means.Mean = gpt.means.ConstantMean()
    ) -> None:
        """
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
    
    def forward(
        self,
        x: t.Tensor
    ) -> gpt.distributions.MultivariateNormal:
        """
        """
        typechecker(x, t.Tensor, "x")
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpt.distributions.MultivariateNormal(mean_x, covar_x)

    def optimize_hyperparams(
        self,
        optimizer: str = "Adam",
        lr: float = 0.1,
        n_iter: int = 30,
        **kwargs
    ) -> None:
        """
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

    def train_all(
        self,
        optimizer: str = "Adam",
        lr: float = 0.1,
        n_iter: int = 30,
        **kwargs
    ) -> None:
        """
        """
        self.train()
        self.likelihood.train()
        self.optimize_hyperparams(
            optimizer=optimizer,
            lr=lr,
            n_iter=n_iter,
            kwargs=kwargs
        )
        self.eval()
        self.likelihood.eval()


    def predict(
        self,
        testdata: TestData
    ) -> None:
        typechecker(testdata, TestData, "testdata")

        with t.no_grad(), gpt.settings.fast_pred_var():
            self.predicted = self.likelihood(
                self(testdata.as_tensor()[0])
            ).mean.view(*testdata.shape)

            self.groundtruth = testdata.as_tensor[1].reshape(
                *testdata.shape
            )

            self.error = t.abs(
                self.predicted - self.groundtruth
            ).detach()
