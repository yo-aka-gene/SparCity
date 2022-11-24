"""Top-level package for sparcity"""

from . import core
from . import dataset
from . import dev
from . import random
from .core import Coord, Generator, subarea, TrainData

__all__ = [
    "core",
    "Coord",
    "dataset",
    "dev",
    "Generator",
    "random",
    "subarea",
    "TrainData",
]

__author__ = """Yuji Okano"""
__email__ = "yujiokano@keio.jp"
__version__ = "0.1.0"
