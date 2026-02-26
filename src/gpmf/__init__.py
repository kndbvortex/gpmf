"""Gradual Pattern Mining - Unified Python Package

A unified, professional package for gradual pattern mining with multiple algorithms
and a clean, SPMF-like interface.

Example:
    >>> from gpmf import GradualPatternMiner
    >>> miner = GradualPatternMiner('graank', 'data.csv', min_support=0.5)
    >>> patterns = miner.mine()
    >>> for p in patterns:
    ...     print(f"{p.to_string()} : {p.support}")
"""

__version__ = '0.3.0'
__author__ = 'Gradual Mining Team'

from .miner import GradualPatternMiner
from .core.data_structures import GradualItem, GradualPattern, TimeLag, TemporalGradualPattern
from .core.dataset import GradualDataset
from .core.result import MiningResult

from .algorithms.graank import GRAANK
from .algorithms.grite import GRITE
from .algorithms.sgrite import SGrite
from .algorithms.swarm.ant_graank import AntGRAANK
from .algorithms.closed.paraminer_algorithm import ParaMiner
from .algorithms.closed.paraminer import RUST_AVAILABLE
from .algorithms.temporal.tgrad import TGrad
from .algorithms.seasonal.msgp import MSGP
from .algorithms.closed.pglcm import PGLCM
from .algorithms.closed.glcm import GLCM

from .factory import AlgorithmRegistry

from .config import config

from .exceptions import (
    GradualMiningError,
    InvalidDataError,
    InvalidAlgorithmError,
    InvalidParameterError,
    MiningError,
    NotFittedError,
)

__all__ = [
    'GradualPatternMiner',
    'GradualItem',
    'GradualPattern',
    'TimeLag',
    'TemporalGradualPattern',
    'GradualDataset',
    'MiningResult',
    'GRAANK',
    'GRITE',
    'SGrite',
    'AntGRAANK',
    'ParaMiner',
    'RUST_AVAILABLE',
    'TGrad',
    'MSGP',
    'PGLCM',
    'GLCM',
    'AlgorithmRegistry',
    'config',
    'GradualMiningError',
    'InvalidDataError',
    'InvalidAlgorithmError',
    'InvalidParameterError',
    'MiningError',
    'NotFittedError',
]


def list_algorithms():
    """List all available algorithms."""
    return AlgorithmRegistry.list_algorithms()
