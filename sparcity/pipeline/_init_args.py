"""
Function to fix initial attributes for GPyTroch.models.ExactGP
"""
import copy

import gpytorch as gpt

from sparcity.dev import typechecker


def init_args(model: gpt.models.ExactGP) -> dict:
    """
    Function to fix initial attributes for GPyTroch.models.ExactGP.
    Expected to be called in `sparcity.pipeline.ComparisonPipeline`.

    Parameters
    ----------
    model: gpytorch.models.ExactGP
        gpytorch model that are not trained yet

    Returns
    -------
    InitialAttributes: dict
        arguments dict for model.initialize

    Examples
    --------
    >>> from sparcity.dataset import QuadraticGenerator
    >>> from sparcity.gaussian_process import KISSGPRegression
    >>> from sparcity.pipeline import init_args
    >>> from sparcity.sampler import RandomInGrid
    >>> area = QuadraticGenerator()
    >>> train = RandomInGrid(n_x=2, n_y=2, length=2, field=area, seed=0)
    >>> model = KISSGPRegression(traindata=train)
    >>> args = init_args(model)
    >>> isinstance(args, dict)
    True
    >>> args.keys()
    dict_keys(['likelihood.noise_covar.raw_noise', 'mean_module.raw_constant', 'covar_module.raw_outputscale', 'covar_module.base_kernel.base_kernel.raw_lengthscale'])
    """
    typechecker(model, gpt.models.ExactGP, "model")

    state_dict = dict(model.state_dict())
    named_params = dict(model.named_parameters())
    return copy.deepcopy(
        {v:state_dict[v] for v in named_params}
    )
