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
    # Abstract base: subclasses implement _mine()

    def __init__(self, min_support: float = 0.5, **kwargs):
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
        raise NotImplementedError

    def fit(self, data: Union[pd.DataFrame, str, GradualDataset], **fit_params) -> 'BaseAlgorithm':
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
        if not self.is_fitted:
            raise NotFittedError("Call fit() first.")
        return self.patterns

    def get_result(self) -> MiningResult:
        if not self.is_fitted:
            raise NotFittedError("Call fit() first.")
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
        self.fit(data, **fit_params)
        return self.get_patterns()

    def mine_and_get_result(self, data: Union[pd.DataFrame, str, GradualDataset], **fit_params) -> MiningResult:
        self.fit(data, **fit_params)
        return self.get_result()

    def get_params(self) -> Dict[str, Any]:
        return {'min_support': self.min_support, **self._params}

    def set_params(self, **params):
        if 'min_support' in params:
            self.min_support = params.pop('min_support')
        self._params.update(params)
        self.is_fitted = False
        return self

    def __repr__(self) -> str:
        params_str = ', '.join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params_str})"
