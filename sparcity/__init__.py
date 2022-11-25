"""Top-level package for sparcity"""

from . import core
from . import dataset
from . import dev
from . import gaussian_process
from . import random
from . import sampler
from .core import Coord, Generator, subarea, patch, TestData, TrainData, UnknownCoord

__all__ = [
    "core",
    "Coord",
    "dataset",
    "dev",
    "gaussian_process",
    "Generator",
    "patch",
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
