"""Top-level package for sparcity"""

from . import core
from . import dataset
from . import debug_utils
from . import dev
from . import gaussian_process
from . import pipeline
from . import random
from . import sampler
from .core import (
    Coord, GPModel, subarea, patch, TestData, TrainData, UnknownCoord
)

__all__ = [
    "core",
    "Coord",
    "dataset",
    "debug_utils",
    "dev",
    "gaussian_process",
    "GPModel",
    "patch",
    "pipeline",
    "random",
    "sampler",
    "subarea",
    "TestData",
    "TrainData",
    "UnknownCoord",
]

__author__ = """Yuji Okano"""
__email__ = "yujiokano@keio.jp"
__version__ = "0.1.0"
