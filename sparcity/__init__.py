"""Top-level package for sparcity"""

from . import core
from . import dataset
from . import dev
from .core import Coord, Generator

__all__ = [
    "core",
    "Coord",
    "dataset",
    "dev",
    "Generator"
]

__author__ = """Yuji Okano"""
__email__ = "yujiokano@keio.jp"
__version__ = "0.1.0"
