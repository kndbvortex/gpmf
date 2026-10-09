import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List

import numpy as np

from ..base_algorithm import BaseAlgorithm
from ...core.data_structures import GradualPattern
from .glcm import _build_all_bms, _mine_root, _results_to_patterns

logger = logging.getLogger(__name__)

# Shared, read-only state of a worker process: the encoded binary matrices and the mining parameters,
# sent once per worker (initializer) instead of once per task.
_STATE: dict = {}


def _init_worker(all_bms, all_bms_stack, min_supp, max_size, use_rc_pruning):
    _STATE.update(all_bms=all_bms, all_bms_stack=all_bms_stack, min_supp=min_supp, max_size=max_size,
                  use_rc_pruning=use_rc_pruning)


def _mine_root_in_worker(e: int) -> tuple:
    st = _STATE
    return _mine_root((e, st['all_bms'][e], st['all_bms'], st['all_bms_stack'], st['min_supp'],
                       st['max_size'], st['use_rc_pruning']))


class PGLCM(BaseAlgorithm):
    # Parallel GLCM: each starting attribute's subtree is independent, no synchronisation needed.
    # Subtrees run in worker processes: the search is pure Python, so threads would be serialised by the GIL.

    def __init__(self, min_support: float = 0.5, n_jobs: int = 1,
                 max_pattern_size: int = 0, use_rc_pruning: bool = False, **kwargs):
        super().__init__(min_support=min_support, **kwargs)
        self.n_jobs = n_jobs
        self.max_pattern_size = max_pattern_size
        self.use_rc_pruning = use_rc_pruning
        self._params['n_jobs'] = n_jobs
        self._params['max_pattern_size'] = max_pattern_size
        self._params['use_rc_pruning'] = use_rc_pruning

    def _mine(self) -> List[GradualPattern]:
        data = self.dataset.numeric_data()
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
        n_candidates = 0
        if n_workers == 1:
            for args in tasks:
                results, n = _mine_root(args)
                raw_results.extend(results)
                n_candidates += n
        else:
            # default start method of the platform (fork on Linux): workers inherit the module state
            ctx = mp.get_context()
            initargs = (all_bms, all_bms_stack, min_supp, self.max_pattern_size, self.use_rc_pruning)
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=_init_worker,
                                     initargs=initargs) as executor:
                # map keeps the order of the roots, so the output does not depend on scheduling
                for results, n in executor.map(_mine_root_in_worker, [t[0] for t in tasks], chunksize=1):
                    raw_results.extend(results)
                    n_candidates += n

        self.n_candidates = n_candidates
        patterns = _results_to_patterns(raw_results, enc_to_attr)
        logger.info(f"PGLCM found {len(patterns)} closed frequent gradual patterns using {n_workers} process(es)")
        return patterns
