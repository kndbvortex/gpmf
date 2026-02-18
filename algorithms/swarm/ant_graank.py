"""Ant Colony Optimization for gradual pattern mining."""
import numpy as np
import logging
from typing import List
import gc

from ..base_algorithm import BaseAlgorithm
from ...core.data_structures import GradualItem, GradualPattern

logger = logging.getLogger(__name__)


class AntGRAANK(BaseAlgorithm):
    """ACO-based gradual pattern mining.

    Uses ant colony optimization with pheromone trails to discover
    frequent gradual patterns.

    Args:
        min_support: Minimum support threshold
        max_iter: Maximum number of iterations (default: 10)
        evaporation_factor: Pheromone evaporation rate (default: 0.5)
    """

    def __init__(self, min_support: float = 0.5, max_iter: int = 10,
                 evaporation_factor: float = 0.5, **kwargs):
        super().__init__(min_support, **kwargs)
        self.max_iter = max_iter
        self.evaporation_factor = evaporation_factor
        self._params['max_iter'] = max_iter
        self._params['evaporation_factor'] = evaporation_factor

    def _parse_gi(self, gi_str: str) -> GradualItem:
        attr_col = int(gi_str[:-1])
        symbol = gi_str[-1]
        return GradualItem(attr_col, symbol)

    def _build_distance_matrix(self):
        v_bins = self.dataset.valid_bins
        attr_keys = []

        for bin_obj in v_bins:
            col = int(bin_obj[0][0])
            symbol = bin_obj[0][1].decode() if hasattr(bin_obj[0][1], 'decode') else bin_obj[0][1]
            gi = GradualItem(col, symbol)
            attr_keys.append(gi.as_string())

        n = len(attr_keys)
        d = np.zeros((n, n), dtype=np.int64)

        for i in range(n):
            for j in range(n):
                gi_i = self._parse_gi(attr_keys[i])
                gi_j = self._parse_gi(attr_keys[j])

                if gi_i.attribute_col == gi_j.attribute_col:
                    continue

                bin_1 = v_bins[i][1]
                bin_2 = v_bins[j][1]
                d[i][j] = np.sum(np.multiply(bin_1, bin_2))

        return d, attr_keys

    def _generate_candidate(self, p_matrix, v_matrix, attr_keys):
        pattern = GradualPattern()
        m = p_matrix.shape[0]

        for i in range(m):
            combine_feature = np.multiply(v_matrix[i], p_matrix[i])
            total = np.sum(combine_feature)

            with np.errstate(divide='ignore', invalid='ignore'):
                probability = combine_feature / total

            cum_prob = np.cumsum(probability)
            r = np.random.random_sample()

            try:
                j = np.nonzero(cum_prob > r)[0][0]
                gi = self._parse_gi(attr_keys[j])

                contains = any(
                    item.attribute_col == gi.attribute_col
                    for item in pattern.gradual_items
                )

                if not contains:
                    pattern.add_gradual_item(gi)

            except IndexError:
                continue

        p_matrix = (1 - self.evaporation_factor) * p_matrix
        return pattern, p_matrix

    def _update_pheromones(self, pattern, p_matrix, attr_keys):
        idx = []
        for item in pattern.gradual_items:
            key = item.as_string()
            if key in attr_keys:
                idx.append(attr_keys.index(key))

        for n in range(len(idx)):
            for m in range(n + 1, len(idx)):
                i = idx[n]
                j = idx[m]
                p_matrix[i][j] += 1
                p_matrix[j][i] += 1

        return p_matrix

    def _validate_pattern(self, pattern):
        if len(pattern.gradual_items) < 2:
            return False

        n = self.dataset.attr_size
        valid_bins = self.dataset.valid_bins

        bin_data = None
        for item in pattern.gradual_items:
            for bin_obj in valid_bins:
                col = int(bin_obj[0][0])
                symbol = bin_obj[0][1].decode() if hasattr(bin_obj[0][1], 'decode') else bin_obj[0][1]

                if col == item.attribute_col and symbol == item.symbol:
                    if bin_data is None:
                        bin_data = bin_obj[1].copy()
                    else:
                        bin_data = bin_data * bin_obj[1]
                    break

        if bin_data is None:
            return False

        support = float(np.sum(bin_data)) / float(n * (n - 1.0) / 2.0)
        if support >= self.dataset.thd_supp:
            pattern.set_support(support)
            return True

        return False

    def _mine(self) -> List[GradualPattern]:
        self.dataset.fit_bitmap()

        if self.dataset.no_bins:
            logger.warning("No valid bins, cannot mine patterns")
            return []

        d, attr_keys = self._build_distance_matrix()

        a = self.dataset.attr_size
        fr_count = ((self.dataset.thd_supp * a * (a - 1)) / 2)
        d[d < fr_count] = 0

        p_matrix = np.ones(d.shape, dtype=float)

        winner_gps = []
        loser_gps = []

        for _ in range(self.max_iter):
            candidate, p_matrix = self._generate_candidate(p_matrix, d, attr_keys)

            if self._validate_pattern(candidate):
                is_super = False
                for loser in loser_gps:
                    loser_set = set((item.attribute_col, item.symbol) for item in loser.gradual_items)
                    cand_set = set((item.attribute_col, item.symbol) for item in candidate.gradual_items)
                    if loser_set.issubset(cand_set):
                        is_super = True
                        break

                if not is_super:
                    is_present = False
                    for winner in winner_gps:
                        winner_set = set((item.attribute_col, item.symbol) for item in winner.gradual_items)
                        cand_set = set((item.attribute_col, item.symbol) for item in candidate.gradual_items)
                        if winner_set == cand_set:
                            is_present = True
                            break

                    if not is_present:
                        idx = 0
                        while idx < len(winner_gps):
                            winner = winner_gps[idx]
                            winner_set = set((item.attribute_col, item.symbol) for item in winner.gradual_items)
                            cand_set = set((item.attribute_col, item.symbol) for item in candidate.gradual_items)

                            if cand_set.issubset(winner_set):
                                del winner_gps[idx]
                            else:
                                idx += 1

                        winner_gps.append(candidate)
                        p_matrix = self._update_pheromones(candidate, p_matrix, attr_keys)
            else:
                loser_gps.append(candidate)

            gc.collect()

        logger.info(f"ACO-GRAANK found {len(winner_gps)} patterns after {self.max_iter} iterations")
        return winner_gps
