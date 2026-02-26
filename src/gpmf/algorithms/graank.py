"""GRAANK algorithm for gradual pattern mining.

This module implements the GRAANK (GRAdual and Numerical Knowledge)
algorithm proposed by Anne Laurent et al.
"""
import numpy as np
import logging
import gc
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from .base_algorithm import BaseAlgorithm
from ..core.data_structures import GradualItem, GradualPattern
from ..utils.validators import validate_n_jobs

logger = logging.getLogger(__name__)


class GRAANK(BaseAlgorithm):
    """GRAANK algorithm for mining gradual patterns.

    GRAANK uses an Apriori-like approach to generate and validate
    gradual pattern candidates using bitmap representations.

    Args:
        min_support: Minimum support threshold (0.0 to 1.0)
        n_jobs: Number of parallel jobs (default: 1, -1 for all cores)

    Example:
        >>> graank = GRAANK(min_support=0.5)
        >>> patterns = graank.mine('data.csv')
        >>> for p in patterns:
        ...     print(f"{p.to_string()} : {p.support}")
    """

    def __init__(self, min_support: float = 0.5, n_jobs: int = 1, **kwargs):
        """Initialize GRAANK algorithm.

        Args:
            min_support: Minimum support threshold
            n_jobs: Number of parallel jobs
            **kwargs: Additional parameters
        """
        super().__init__(min_support, **kwargs)
        self.n_jobs = validate_n_jobs(n_jobs)
        self._params['n_jobs'] = self.n_jobs

    @staticmethod
    def _inv_arr(g_item: Tuple) -> Tuple:
        """Invert a gradual item tuple.

        Args:
            g_item: Gradual item as tuple (col, symbol)

        Returns:
            Inverted gradual item
        """
        symbol = g_item[1].decode() if hasattr(g_item[1], 'decode') else g_item[1]
        if symbol == '+':
            return tuple([g_item[0], '-'])
        else:
            return tuple([g_item[0], '+'])

    def _gen_apriori_candidates(self, gi_bins: List) -> Tuple[List, int]:
        """Generate Apriori candidates from gradual items.

        Args:
            gi_bins: List of [gradual_item_set, bitmap] pairs

        Returns:
            Tuple of (valid_candidates, invalid_count)
        """
        sup = self.dataset.thd_supp
        n = self.dataset.row_count

        invalid_count = 0
        res = []
        all_candidates = []

        if len(gi_bins) < 2:
            return [], 0

        try:
            set_gi = [{x[0]} for x in gi_bins]
        except TypeError:
            set_gi = [set(x[0]) for x in gi_bins]

        for i in range(len(gi_bins) - 1):
            for j in range(i + 1, len(gi_bins)):
                try:
                    gi_i = {gi_bins[i][0]}
                    gi_j = {gi_bins[j][0]}
                    gi_o = {gi_bins[0][0]}
                except TypeError:
                    gi_i = set(gi_bins[i][0])
                    gi_j = set(gi_bins[j][0])
                    gi_o = set(gi_bins[0][0])

                gp_cand = gi_i | gi_j
                inv_gp_cand = {self._inv_arr(x) for x in gp_cand}

                if (len(gp_cand) == len(gi_o) + 1 and
                    not (all_candidates and gp_cand in all_candidates) and
                    not (all_candidates and inv_gp_cand in all_candidates)):

                    test = 1
                    for k in gp_cand:
                        try:
                            k_set = {k}
                        except TypeError:
                            k_set = set(k)

                        gp_cand_2 = gp_cand - k_set
                        inv_gp_cand_2 = {self._inv_arr(x) for x in gp_cand_2}

                        if gp_cand_2 not in set_gi and inv_gp_cand_2 not in set_gi:
                            test = 0
                            break

                    if test == 1:
                        m = gi_bins[i][1] * gi_bins[j][1]
                        t = float(np.sum(m)) / float(n * (n - 1.0) / 2.0)

                        if t >= sup:
                            res.append([gp_cand, m])
                        else:
                            invalid_count += 1

                    all_candidates.append(gp_cand)
                    gc.collect()

        return res, invalid_count

    def _mine(self) -> List[GradualPattern]:
        """Mine gradual patterns using GRAANK algorithm.

        Returns:
            List of discovered GradualPattern objects
        """
        self.dataset.fit_bitmap()

        if self.dataset.no_bins:
            logger.warning("No valid bitmaps generated. Cannot mine patterns.")
            return []

        gradual_patterns = []
        n = self.dataset.attr_size
        valid_bins = self.dataset.valid_bins.tolist()

        invalid_count = 0

        while len(valid_bins) > 0:
            valid_bins, inv_count = self._gen_apriori_candidates(valid_bins)
            invalid_count += inv_count

            i = 0
            while i < len(valid_bins) and valid_bins:
                gi_tuple = valid_bins[i][0]
                bin_data = valid_bins[i][1]

                sup = float(np.sum(np.array(bin_data))) / float(n * (n - 1.0) / 2.0)

                if sup < self.dataset.thd_supp:
                    del valid_bins[i]
                    invalid_count += 1
                else:
                    z = 0
                    while z < len(gradual_patterns):
                        existing_pattern_set = set(p.tuple for p in gradual_patterns[z].gradual_items)
                        if existing_pattern_set.issubset(set(gi_tuple)):
                            del gradual_patterns[z]
                        else:
                            z += 1

                    gp = GradualPattern()
                    for obj in gi_tuple:
                        col = obj[0]
                        symbol = obj[1].decode() if hasattr(obj[1], 'decode') else obj[1]
                        gi = GradualItem(col, symbol)
                        gp.add_gradual_item(gi)

                    gp.set_support(sup)
                    gradual_patterns.append(gp)
                    i += 1

        logger.info(f"GRAANK found {len(gradual_patterns)} patterns (invalid: {invalid_count})")
        return gradual_patterns
