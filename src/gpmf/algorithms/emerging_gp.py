"""Emerging Gradual Pattern mining.

Based on:
    Laurent, A., Lesot, M.-J., & Rifqi, M. (2015). Mining Emerging Gradual Patterns.
    IFSA-EUSFLAT 2015. DOI: 10.2991/ifsa-eusflat-15.2015.234

A pattern P is an emerging gradual pattern (GEP) from D1 to D2 if:
    supp_D2(P) >= s2   (frequent in target)
    supp_D1(P) <  s1   (not frequent in reference)

The algorithm uses a border-based representation: the set of frequent gradual
patterns is convex (anti-monotone property), so it is compactly represented by
its maximal elements. The set difference of the two borders yields the GEPs.
"""
import time
import logging
from typing import List, Union

import pandas as pd

from .base_algorithm import BaseAlgorithm
from ..core.data_structures import GradualItem, GradualPattern
from ..core.dataset import GradualDataset
from ..exceptions import NotFittedError

logger = logging.getLogger(__name__)


class EmergingGP:
    """Emerging Gradual Pattern mining between two datasets.

    Finds gradual patterns that are frequent in a target dataset D2 but absent
    (below threshold) in a reference dataset D1. Such patterns characterise D2
    by contrast to D1 in terms of attribute co-variations.

    Unlike the single-dataset algorithms (GRAANK, GRITE, …) this class takes
    *two* datasets and therefore does not inherit from BaseAlgorithm.

    Args:
        min_support_d1: Support threshold for the reference dataset D1.
            Patterns with supp_D1 >= min_support_d1 are considered present in D1.
        min_support_d2: Support threshold for the target dataset D2.
            Only patterns with supp_D2 >= min_support_d2 are candidates.
        base_algo: Name of the base GP algorithm used for mining each dataset
            individually (any key accepted by AlgorithmRegistry). Default: 'graank'.

    Example:
        >>> import pandas as pd
        >>> from gpmf.algorithms.emerging_gp import EmergingGP
        >>> d1 = pd.DataFrame({'A': [1,2,3], 'B': [3,2,1]})
        >>> d2 = pd.DataFrame({'A': [1,2,3], 'B': [1,2,3]})
        >>> egp = EmergingGP(min_support_d1=0.5, min_support_d2=0.5)
        >>> patterns = egp.fit(d1, d2).get_patterns()
    """

    def __init__(
        self,
        min_support_d1: float = 0.5,
        min_support_d2: float = 0.5,
        base_algo: str = 'graank',
    ):
        if min_support_d1 <= 0 or min_support_d2 <= 0:
            raise ValueError("Support thresholds must be positive")
        self.min_support_d1 = min_support_d1
        self.min_support_d2 = min_support_d2
        self.base_algo = base_algo

        self.patterns: List[GradualPattern] = []
        self._maximals_d1: List[GradualPattern] = []
        self._patterns_d2: List[GradualPattern] = []
        self.execution_time: float = 0.0
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        d1: Union[pd.DataFrame, str, GradualDataset],
        d2: Union[pd.DataFrame, str, GradualDataset],
    ) -> 'EmergingGP':
        """Mine emerging gradual patterns from d2 with respect to d1.

        Args:
            d1: Reference dataset (DataFrame, CSV path, or GradualDataset).
            d2: Target dataset (DataFrame, CSV path, or GradualDataset).

        Returns:
            self
        """
        from ..factory import AlgorithmRegistry

        start = time.time()

        algo_cls = AlgorithmRegistry.get(self.base_algo)

        # Step 1 — mine D1 (reference), keep maximal patterns as border R1
        logger.info("Mining reference dataset D1 (s1=%.3f)", self.min_support_d1)
        algo1 = algo_cls(min_support=self.min_support_d1)
        algo1.fit(d1)
        self._maximals_d1 = algo1.get_patterns()
        logger.info("R1: %d maximal patterns", len(self._maximals_d1))

        # Step 2 — mine D2 (target)
        logger.info("Mining target dataset D2 (s2=%.3f)", self.min_support_d2)
        algo2 = algo_cls(min_support=self.min_support_d2)
        algo2.fit(d2)
        self._patterns_d2 = algo2.get_patterns()
        logger.info("R2: %d maximal patterns", len(self._patterns_d2))

        # Step 3 — border difference: patterns in D2 not subsumed by D1's border
        self.patterns = [
            p for p in self._patterns_d2
            if not self._is_in_d1(p)
        ]

        self.execution_time = time.time() - start
        self.is_fitted = True

        logger.info(
            "EmergingGP: %d emerging patterns found in %.3fs",
            len(self.patterns), self.execution_time,
        )
        return self

    def get_patterns(self) -> List[GradualPattern]:
        """Return emerging gradual patterns.

        Returns:
            List of GradualPattern objects frequent in D2 but not in D1.

        Raises:
            NotFittedError: If fit() has not been called.
        """
        if not self.is_fitted:
            raise NotFittedError("Call fit() first.")
        return self.patterns

    def get_params(self) -> dict:
        return {
            'min_support_d1': self.min_support_d1,
            'min_support_d2': self.min_support_d2,
            'base_algo': self.base_algo,
        }

    def __repr__(self) -> str:
        return (
            f"EmergingGP(min_support_d1={self.min_support_d1}, "
            f"min_support_d2={self.min_support_d2}, "
            f"base_algo='{self.base_algo}')"
        )

    # ------------------------------------------------------------------
    # Border subsumption helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_frozenset(pattern: GradualPattern) -> frozenset:
        """Encode a GP as a frozenset of (attr_col, symbol) pairs."""
        return frozenset(
            (item.attribute_col, item.symbol)
            for item in pattern.gradual_items
        )

    @staticmethod
    def _complement(fset: frozenset) -> frozenset:
        """Return the complement encoding (invert all symbols)."""
        return frozenset(
            (col, '+' if sym == '-' else '-')
            for col, sym in fset
        )

    def _is_in_d1(self, pattern: GradualPattern) -> bool:
        """Return True if *pattern* is subsumed by D1's border (frequent in D1).

        A pattern P is frequent in D1 iff there exists a maximal pattern Q in R1
        such that P ⊆ Q (as item sets). Both canonical and complement encodings
        are checked because {A+,B-} and {A-,B+} represent the same ordering.
        """
        p_set = self._to_frozenset(pattern)
        p_comp = self._complement(p_set)
        for q in self._maximals_d1:
            q_set = self._to_frozenset(q)
            if p_set <= q_set or p_comp <= q_set:
                return True
        return False
