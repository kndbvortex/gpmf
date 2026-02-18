"""
ParaMiner algorithm wrapper for gradual_mining package.

This wraps the standalone ParaMiner implementation to work with the
BaseAlgorithm interface while preserving the ability to run with or
without Rust acceleration.
"""

import pandas as pd
from pathlib import Path
from typing import List, Union, Optional
import tempfile

from ..base_algorithm import BaseAlgorithm
from ...core.data_structures import GradualPattern as GP, GradualItem as GI

# Import from the paraminer submodule
from .paraminer.gradual_mining import GradualMiner
from .paraminer import RUST_AVAILABLE
from .paraminer.datatypes import Variation as ParaminerVariation


class ParaMiner(BaseAlgorithm):
    """
    ParaMiner algorithm for mining closed frequent gradual patterns.

    ParaMiner uses a parallel depth-first search with closure operator
    to efficiently mine closed gradual patterns. It can optionally use
    Rust acceleration for significant performance improvements.

    Args:
        min_support: Minimum support threshold (0-1 for relative, >1 for absolute)
        use_rust: Use Rust acceleration if available (default: True)
        num_threads: Number of threads for parallel processing (default: None = auto)
        verbose: Print detailed mining information (default: False)
        **kwargs: Additional arguments (ignored for compatibility)

    Examples:
        >>> # Basic usage with Python implementation
        >>> miner = ParaMiner(min_support=0.5, use_rust=False)
        >>> miner.fit(data)
        >>> patterns = miner.get_patterns()

        >>> # With Rust acceleration (if available)
        >>> miner = ParaMiner(min_support=0.5, use_rust=True)
        >>> miner.fit(data)
        >>> patterns = miner.get_patterns()

        >>> # With multi-threading
        >>> miner = ParaMiner(min_support=0.5, num_threads=4)
        >>> miner.fit(data)
    """

    def __init__(
        self,
        min_support: float = 0.5,
        use_rust: bool = True,
        num_threads: Optional[int] = None,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(min_support=min_support, **kwargs)
        self.use_rust = use_rust
        self.num_threads = num_threads
        self.verbose = verbose
        self._miner = None
        self._temp_file = None

    def _mine(self) -> List[GP]:
        """
        Execute the ParaMiner mining algorithm.

        Returns:
            List of closed frequent gradual patterns
        """
        if self.dataset is None:
            raise ValueError("No data loaded. Call fit() first.")

        data_array = self.dataset.data
        titles = self.dataset.titles

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            self._temp_file = f.name

            f.write(' '.join(map(str, titles)) + '\n')
            for row in data_array:
                f.write(' '.join(map(str, row)) + '\n')

        try:
            self._miner = GradualMiner(
                min_support=self.min_support,
                num_threads=self.num_threads,
                use_rust=self.use_rust,
                verbose=self.verbose
            )

            self._miner.load_data(self._temp_file)
            paraminer_patterns = self._miner.mine()

            patterns = []
            for pp in paraminer_patterns:
                pattern = GP()
                pattern.set_support(pp.support)
                for item in pp.items:
                    symbol = '+' if item.variation == ParaminerVariation.INCREASE else '-'
                    gi = GI(attr_col=item.attribute_index, symbol=symbol)
                    pattern.add_gradual_item(gi)
                patterns.append(pattern)

            return patterns

        finally:
            if self._temp_file and Path(self._temp_file).exists():
                Path(self._temp_file).unlink()

    @property
    def rust_available(self) -> bool:
        """Check if Rust acceleration is available."""
        return RUST_AVAILABLE

    @property
    def is_using_rust(self) -> bool:
        """Check if this instance is configured to use Rust."""
        return self.use_rust and RUST_AVAILABLE

    def get_info(self) -> dict:
        """
        Get algorithm information.

        Returns:
            Dictionary with algorithm metadata
        """
        info = super().get_info()
        info.update({
            'use_rust': self.use_rust,
            'rust_available': RUST_AVAILABLE,
            'is_using_rust': self.is_using_rust,
            'num_threads': self.num_threads,
        })
        return info
