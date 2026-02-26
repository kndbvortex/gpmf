"""Algorithm factory for gradual pattern mining."""
import functools
from typing import Any, Dict, List, Type
import logging

from .algorithms.base_algorithm import BaseAlgorithm
from .algorithms.graank import GRAANK
from .algorithms.grite import GRITE
from .algorithms.sgrite import SGrite
from .algorithms.swarm.ant_graank import AntGRAANK
from .algorithms.closed.paraminer_algorithm import ParaMiner
from .algorithms.temporal.tgrad import TGrad
from .algorithms.seasonal.msgp import MSGP
from .algorithms.closed.glcm import GLCM
from .algorithms.closed.pglcm import PGLCM
from .exceptions import InvalidAlgorithmError

logger = logging.getLogger(__name__)


class AlgorithmRegistry:
    """Registry for gradual pattern mining algorithms."""

    # Values are either a class (Type[BaseAlgorithm]) or a functools.partial
    # that pre-sets default kwargs (e.g., variant aliases for SGrite).
    _algorithms: Dict[str, Any] = {
        'graank':     GRAANK,
        'grite':      GRITE,
        # SGrite: 'sgrite' uses the default variant (sgb1).
        # The four shorthand keys pre-select the corresponding variant.
        'sgrite':     SGrite,
        'sgopt':      functools.partial(SGrite, variant='sgopt'),
        'sg1':        functools.partial(SGrite, variant='sg1'),
        'sgb1':       functools.partial(SGrite, variant='sgb1'),
        'sgb2':       functools.partial(SGrite, variant='sgb2'),
        'ant-graank': AntGRAANK,
        'aco-graank': AntGRAANK,
        'paraminer':  ParaMiner,
        'tgrad':      TGrad,
        't-graank':   TGrad,
        'temporal':   TGrad,
        'msgp':       MSGP,
        'seasonal':   MSGP,
        'glcm':       GLCM,
        'pglcm':      PGLCM,
    }

    @classmethod
    def register(cls, name: str, algorithm_class: Type[BaseAlgorithm]):
        if not issubclass(algorithm_class, BaseAlgorithm):
            raise ValueError(f"{algorithm_class} must inherit from BaseAlgorithm")
        cls._algorithms[name.lower()] = algorithm_class
        logger.info(f"Registered algorithm: {name}")

    @classmethod
    def get(cls, name: str) -> Any:
        name_lower = name.lower()
        if name_lower not in cls._algorithms:
            available = cls.list_algorithms()
            raise InvalidAlgorithmError(
                f"Unknown algorithm '{name}'. Available algorithms: {', '.join(available)}"
            )
        return cls._algorithms[name_lower]

    @classmethod
    def list_algorithms(cls) -> List[str]:
        return sorted(set(cls._algorithms.keys()))

    @classmethod
    def has_algorithm(cls, name: str) -> bool:
        return name.lower() in cls._algorithms
