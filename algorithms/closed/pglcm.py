"""PGLCM - Parallel mining of closed frequent gradual itemsets.

Reference:
    Do, T.D.T., Termier, A., Laurent, A., Negrevergne, B., Jeudy, B., &
    Gacias, B. (2015). PGLCM: Efficient Parallel Mining of Closed Frequent
    Gradual Itemsets. Knowledge and Information Systems, 43(3), 681-737.
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np

from ..base_algorithm import BaseAlgorithm
from ...core.data_structures import GradualPattern
from .glcm import _build_all_bms, _mine_root, _results_to_patterns

logger = logging.getLogger(__name__)


class PGLCM(BaseAlgorithm):
    """PGLCM algorithm for parallel mining of closed frequent gradual itemsets.

    Parallelises GLCM by distributing each starting attribute across worker
    threads. Each starting item's subtree is fully independent, so no
    synchronisation is needed between workers. Results are merged and
    deduplicated after all workers complete.

    Thread-based parallelism is used (not processes) to avoid serialisation
    overhead for the binary-matrix data. NumPy releases the GIL for most
    array operations, providing real concurrency for the compute-heavy parts.

    Args:
        min_support: Minimum support threshold (0-1 for relative, >1 for absolute)
        n_jobs: Worker threads. -1 uses all available CPUs (default 1).

    Example:
        >>> pglcm = PGLCM(min_support=0.5, n_jobs=4)
        >>> patterns = pglcm.mine('data.csv')
    """

    def __init__(
        self,
        min_support: float = 0.5,
        n_jobs: int = 1,
        max_pattern_size: int = 0,
        use_rc_pruning: bool = False,
        **kwargs,
    ):
        super().__init__(min_support=min_support, **kwargs)
        self.n_jobs = n_jobs
        self.max_pattern_size = max_pattern_size
        self.use_rc_pruning = use_rc_pruning
        self._params['n_jobs'] = n_jobs
        self._params['max_pattern_size'] = max_pattern_size
        self._params['use_rc_pruning'] = use_rc_pruning

    def _mine(self) -> List[GradualPattern]:
        data = self.dataset.data.astype(float)
        n = data.shape[0]
        attr_cols = self.dataset.attr_cols
        min_supp = self.min_support / n if self.min_support >= 1 else self.min_support

        all_bms, enc_to_attr, all_bms_stack = _build_all_bms(data, attr_cols)
        if not all_bms:
            return []

        n_enc = len(all_bms)
        tasks = [
            (e, all_bms[e], all_bms, all_bms_stack, min_supp, self.max_pattern_size, self.use_rc_pruning)
            for e in range(0, n_enc, 2)
        ]

        n_workers = (os.cpu_count() or 1) if self.n_jobs == -1 else max(1, self.n_jobs)

        raw_results: list = []
        if n_workers == 1:
            for args in tasks:
                raw_results.extend(_mine_root(args))
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(_mine_root, args) for args in tasks]
                for future in as_completed(futures):
                    raw_results.extend(future.result())

        patterns = _results_to_patterns(raw_results, enc_to_attr)
        logger.info(
            f"PGLCM found {len(patterns)} closed frequent gradual patterns "
            f"using {n_workers} thread(s)"
        )
        return patterns
