"""Main interface for gradual pattern mining."""
from typing import Union, List
import pandas as pd
import logging

from .core.data_structures import GradualPattern
from .core.dataset import GradualDataset
from .core.result import MiningResult
from .factory import AlgorithmRegistry
from .config import config

logger = logging.getLogger(__name__)


class GradualPatternMiner:
    """Main interface for gradual pattern mining.

    Provides a simple, unified API for mining gradual patterns using
    different algorithms.

    Args:
        algorithm: Algorithm name (e.g., 'graank', 'grite')
        data: Input data (DataFrame, CSV path, or GradualDataset)
        min_support: Minimum support threshold (0.0 to 1.0)
        **kwargs: Algorithm-specific parameters

    Example:
        >>> # Simple usage
        >>> miner = GradualPatternMiner('graank', 'data.csv', min_support=0.5)
        >>> patterns = miner.mine()

        >>> # With result metadata
        >>> result = miner.mine_and_get_result()
        >>> print(result.summary())
        >>> print(result.to_json())

        >>> # Using different algorithms
        >>> miner = GradualPatternMiner('grite', data, min_support=0.5, n_jobs=-1)
        >>> patterns = miner.mine()
    """

    def __init__(
        self,
        algorithm: str,
        data: Union[pd.DataFrame, str, GradualDataset] = None,
        min_support: float = 0.5,
        **kwargs
    ):
        """Initialize gradual pattern miner.

        Args:
            algorithm: Algorithm name
            data: Input data (optional, can be provided in mine())
            min_support: Minimum support threshold
            **kwargs: Algorithm-specific parameters
        """
        if not config.suppress_prints:
            config.setup_logging()

        self.algorithm_name = algorithm
        algorithm_class = AlgorithmRegistry.get(algorithm)
        self.algorithm = algorithm_class(min_support=min_support, **kwargs)
        self.data = data
        self.is_fitted = False

        logger.info(f"Initialized {algorithm} miner with min_support={min_support}")

    def mine(self, data: Union[pd.DataFrame, str, GradualDataset] = None) -> List[GradualPattern]:
        """Mine gradual patterns from data.

        Args:
            data: Input data (if not provided in constructor)

        Returns:
            List of discovered GradualPattern objects

        Raises:
            ValueError: If no data is provided
        """
        if data is None:
            data = self.data

        if data is None:
            raise ValueError("No data provided. Pass data to constructor or mine() method.")

        patterns = self.algorithm.mine(data)
        self.is_fitted = True

        return patterns

    def mine_and_get_result(self, data: Union[pd.DataFrame, str, GradualDataset] = None) -> MiningResult:
        """Mine patterns and return result with metadata.

        Args:
            data: Input data (if not provided in constructor)

        Returns:
            MiningResult object containing patterns and metadata

        Raises:
            ValueError: If no data is provided
        """
        if data is None:
            data = self.data

        if data is None:
            raise ValueError("No data provided. Pass data to constructor or mine() method.")

        result = self.algorithm.mine_and_get_result(data)
        self.is_fitted = True

        return result

    def get_patterns(self) -> List[GradualPattern]:
        """Get patterns from last mining operation.

        Returns:
            List of patterns

        Raises:
            NotFittedError: If mine() hasn't been called yet
        """
        return self.algorithm.get_patterns()

    def get_result(self) -> MiningResult:
        """Get result from last mining operation.

        Returns:
            MiningResult object

        Raises:
            NotFittedError: If mine() hasn't been called yet
        """
        return self.algorithm.get_result()

    def __repr__(self) -> str:
        """String representation."""
        return f"GradualPatternMiner(algorithm='{self.algorithm_name}', min_support={self.algorithm.min_support})"
