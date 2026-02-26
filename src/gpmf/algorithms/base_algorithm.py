"""Base class for all gradual pattern mining algorithms."""
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
import pandas as pd
import time
import logging

from ..core.data_structures import GradualPattern
from ..core.dataset import GradualDataset
from ..core.result import MiningResult
from ..exceptions import NotFittedError, InvalidParameterError

logger = logging.getLogger(__name__)


class BaseAlgorithm(ABC):
    """Abstract base class for gradual pattern mining algorithms.

    All mining algorithms should inherit from this class and implement
    the _mine() method.

    Attributes:
        min_support: Minimum support threshold (0.0 to 1.0)
        dataset: GradualDataset object
        patterns: Discovered patterns (after fit)
        execution_time: Time taken to mine patterns
        is_fitted: Whether the algorithm has been fitted

    Example:
        >>> class MyAlgorithm(BaseAlgorithm):
        ...     def _mine(self) -> List[GradualPattern]:
        ...         # Implementation here
        ...         return patterns
        >>> algo = MyAlgorithm(min_support=0.5)
        >>> algo.fit('data.csv')
        >>> patterns = algo.get_patterns()
    """

    def __init__(self, min_support: float = 0.5, **kwargs):
        """Initialize base algorithm.

        Args:
            min_support: Minimum support threshold (0.0-1.0 for relative, >1 for absolute)
            **kwargs: Additional algorithm-specific parameters

        Raises:
            InvalidParameterError: If parameters are invalid
        """
        if min_support <= 0:
            raise InvalidParameterError("min_support must be positive")

        self.min_support = min_support
        self.dataset: Optional[GradualDataset] = None
        self.patterns: List[GradualPattern] = []
        self.execution_time: Optional[float] = None
        self.is_fitted = False
        self._params = kwargs

        logger.debug(f"Initialized {self.__class__.__name__} with min_support={min_support}")

    @abstractmethod
    def _mine(self) -> List[GradualPattern]:
        """Mine gradual patterns from the dataset.

        This method must be implemented by subclasses.

        Returns:
            List of discovered GradualPattern objects

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclass must implement _mine() method")

    def fit(self, data: Union[pd.DataFrame, str, GradualDataset], **fit_params) -> 'BaseAlgorithm':
        """Fit the algorithm on the data and mine patterns.

        Args:
            data: Input data (DataFrame, CSV path, or GradualDataset)
            **fit_params: Additional fit parameters

        Returns:
            Self for method chaining

        Raises:
            Exception: If mining fails
        """
        if isinstance(data, GradualDataset):
            self.dataset = data
        else:
            self.dataset = GradualDataset(data, min_sup=self.min_support)

        logger.info(f"Fitting {self.__class__.__name__} on dataset with {self.dataset.row_count} rows")

        start_time = time.time()
        try:
            self.patterns = self._mine()
            self.execution_time = time.time() - start_time
            self.is_fitted = True

            logger.info(f"Mining completed in {self.execution_time:.3f}s, found {len(self.patterns)} patterns")

        except Exception as e:
            logger.error(f"Mining failed: {e}")
            raise

        return self

    def get_patterns(self) -> List[GradualPattern]:
        """Get the discovered patterns.

        Returns:
            List of GradualPattern objects

        Raises:
            NotFittedError: If algorithm hasn't been fitted yet
        """
        if not self.is_fitted:
            raise NotFittedError("Algorithm must be fitted before getting patterns. Call fit() first.")

        return self.patterns

    def get_result(self) -> MiningResult:
        """Get mining results as MiningResult object.

        Returns:
            MiningResult containing patterns and metadata

        Raises:
            NotFittedError: If algorithm hasn't been fitted yet
        """
        if not self.is_fitted:
            raise NotFittedError("Algorithm must be fitted before getting results. Call fit() first.")

        metadata = {
            'num_rows': self.dataset.row_count,
            'num_cols': self.dataset.col_count,
            'num_attr_cols': len(self.dataset.attr_cols),
            **self._params
        }

        return MiningResult(
            patterns=self.patterns,
            algorithm=self.__class__.__name__,
            min_support=self.min_support,
            execution_time=self.execution_time,
            **metadata
        )

    def mine(self, data: Union[pd.DataFrame, str, GradualDataset], **fit_params) -> List[GradualPattern]:
        """Convenience method to fit and get patterns in one call.

        Args:
            data: Input data
            **fit_params: Additional fit parameters

        Returns:
            List of discovered patterns
        """
        self.fit(data, **fit_params)
        return self.get_patterns()

    def mine_and_get_result(self, data: Union[pd.DataFrame, str, GradualDataset], **fit_params) -> MiningResult:
        """Convenience method to fit and get result in one call.

        Args:
            data: Input data
            **fit_params: Additional fit parameters

        Returns:
            MiningResult object
        """
        self.fit(data, **fit_params)
        return self.get_result()

    def get_params(self) -> Dict[str, Any]:
        """Get algorithm parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            'min_support': self.min_support,
            **self._params
        }

    def set_params(self, **params):
        """Set algorithm parameters.

        Args:
            **params: Parameters to set

        Returns:
            Self for method chaining
        """
        if 'min_support' in params:
            self.min_support = params.pop('min_support')

        self._params.update(params)
        self.is_fitted = False  # Reset fit status
        return self

    def __repr__(self) -> str:
        """String representation."""
        params_str = ', '.join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params_str})"
