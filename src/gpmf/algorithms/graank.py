import numpy as np
import logging
from typing import List, Tuple

from .base_algorithm import BaseAlgorithm
from ..core.data_structures import GradualItem, GradualPattern
from ..utils.validators import validate_n_jobs

logger = logging.getLogger(__name__)


class GRAANK(BaseAlgorithm):
    # Apriori-like gradual pattern mining via bitmap AND + support filtering

    def __init__(self, min_support: float = 0.5, n_jobs: int = 1, **kwargs):
        super().__init__(min_support, **kwargs)
        self.n_jobs = validate_n_jobs(n_jobs)
        self._params['n_jobs'] = self.n_jobs

    @staticmethod
    def _inv_arr(g_item: Tuple) -> Tuple:
        symbol = g_item[1].decode() if hasattr(g_item[1], 'decode') else g_item[1]
        return (g_item[0], '-') if symbol == '+' else (g_item[0], '+')

    def _gen_apriori_candidates(self, gi_bins: List) -> Tuple[List, int]:
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
                        t = gi_bins[i][1].sum() / (n * (n - 1.0) / 2.0)
                        if t >= sup:
                            res.append([gp_cand, gi_bins[i][1] * gi_bins[j][1]])
                        else:
                            invalid_count += 1

                    all_candidates.append(gp_cand)

        return res, invalid_count

    def _mine(self) -> List[GradualPattern]:
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
                sup = np.asarray(bin_data).sum() / (n * (n - 1.0) / 2.0)

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
                        gp.add_gradual_item(GradualItem(col, symbol))

                    gp.set_support(sup)
                    gradual_patterns.append(gp)
                    i += 1

        logger.info(f"GRAANK found {len(gradual_patterns)} patterns (invalid: {invalid_count})")
        return gradual_patterns
