import logging
from typing import List, Tuple

import numpy as np

from ..base_algorithm import BaseAlgorithm
from ...core.data_structures import GradualItem, GradualPattern

logger = logging.getLogger(__name__)


def _build_bm(col: np.ndarray) -> np.ndarray:
    bm = col[:, None] <= col[None, :]
    np.fill_diagonal(bm, False)
    return bm


def _set_contrast_to_zero(bm: np.ndarray) -> np.ndarray:
    result = bm.copy()
    cycles = bm & bm.T
    result &= ~np.tril(cycles, -1)
    return result


def _prune_matrix(bm: np.ndarray, min_chain: int) -> np.ndarray:
    if min_chain <= 1:
        return bm
    threshold = min_chain - 1
    result = bm.copy()
    while True:
        row_s = result.sum(axis=1)
        col_s = result.sum(axis=0)
        keep = (row_s + col_s) >= threshold
        if keep.all():
            break
        result[~keep, :] = False
        result[:, ~keep] = False
    return result


def _compute_support(bm: np.ndarray) -> float:
    # Vectorised topo-sort DP: number of Python iterations = DAG depth
    n = bm.shape[0]
    dag = _set_contrast_to_zero(bm)
    dp = np.ones(n, dtype=np.int32)
    in_deg = dag.sum(axis=0, dtype=np.int32)
    processed = np.zeros(n, dtype=bool)

    while True:
        ready = (in_deg == 0) & ~processed
        if not ready.any():
            break
        ready_dp = dp[ready][:, None]
        succ = dag[ready]
        if succ.any():
            updates = np.where(succ, ready_dp + 1, 0).max(axis=0)
            np.maximum(dp, updates, out=dp)
            in_deg -= succ.sum(axis=0, dtype=np.int32)
        processed |= ready

    return int(dp.max()) / n


def _compute_closure(bm: np.ndarray, all_bms_stack: np.ndarray) -> Tuple[int, ...]:
    violations = bm[None, :, :] > all_bms_stack
    in_closure = ~violations.any(axis=(1, 2))
    return tuple(int(i) for i in np.where(in_closure)[0])


def _ppc_ok(P1: frozenset, Q: Tuple[int, ...], core_i: int) -> bool:
    for enc in Q:
        if enc < core_i and enc not in P1:
            return False
    return True


def _glcm_loop(
    P: frozenset,
    bm: np.ndarray,
    core_i: int,
    min_supp: float,
    all_bms: List[np.ndarray],
    all_bms_stack: np.ndarray,
    results: list,
    max_size: int = 0,
    use_rc_pruning: bool = True,
) -> None:
    n_enc = len(all_bms)
    n = bm.shape[0]
    min_chain = int(np.ceil(min_supp * n))
    min_edges = min_chain * (min_chain - 1) // 2

    for e in range(core_i + 1, n_enc):
        if e in P or (e ^ 1) in P:
            continue

        bm1 = bm & all_bms[e]
        if int(bm1.sum()) < min_edges:
            continue

        bm1_for_supp = _prune_matrix(bm1, min_chain) if use_rc_pruning else bm1
        supp = _compute_support(bm1_for_supp)
        if supp < min_supp:
            continue

        P1 = P | {e}
        Q = _compute_closure(bm1, all_bms_stack)
        if not _ppc_ok(P1, Q, e):
            continue

        results.append((Q, supp))
        if max_size and len(Q) >= max_size:
            continue
        _glcm_loop(frozenset(Q), bm1, e, min_supp, all_bms, all_bms_stack, results, max_size, use_rc_pruning)


def _mine_root(args: tuple) -> list:
    e, bm_e, all_bms, all_bms_stack, min_supp, max_size, use_rc_pruning = args
    supp = _compute_support(bm_e)
    if supp < min_supp:
        return []

    Q = _compute_closure(bm_e, all_bms_stack)
    if any(q < e for q in Q):
        return []

    results = [(Q, supp)]
    if not (max_size and len(Q) >= max_size):
        _glcm_loop(frozenset(Q), bm_e, e, min_supp, all_bms, all_bms_stack, results, max_size, use_rc_pruning)
    return results


def _results_to_patterns(raw_results: list, enc_to_attr: List[Tuple[int, str]]) -> List[GradualPattern]:
    seen: set = set()
    patterns: List[GradualPattern] = []

    for Q, supp in raw_results:
        if len(Q) < 2:
            continue
        key = Q
        if key in seen:
            continue
        encs = set(Q)
        if any((enc ^ 1) in encs for enc in Q):
            continue
        seen.add(key)
        gp = GradualPattern()
        gp.set_support(supp)
        for enc in Q:
            col_idx, symbol = enc_to_attr[enc]
            gp.add_gradual_item(GradualItem(col_idx, symbol))
        patterns.append(gp)

    return patterns


def _build_all_bms(
    data: np.ndarray,
    attr_cols: np.ndarray,
) -> Tuple[List[np.ndarray], List[Tuple[int, str]], np.ndarray]:
    all_bms: List[np.ndarray] = []
    enc_to_attr: List[Tuple[int, str]] = []
    for col_idx in attr_cols:
        col = data[:, col_idx]
        if np.all(col == col[0]):
            continue  # constant column: BM is all-True, degenerates the search
        bm_asc = _build_bm(col)
        all_bms.append(bm_asc)
        all_bms.append(bm_asc.T.copy())
        enc_to_attr.append((int(col_idx), '+'))
        enc_to_attr.append((int(col_idx), '-'))

    if not all_bms:
        return [], [], np.empty((0, 0, 0), dtype=bool)

    all_bms_stack = np.stack(all_bms)
    return all_bms, enc_to_attr, all_bms_stack


class GLCM(BaseAlgorithm):
    # LCM-adapted closed frequent gradual itemset miner; support = longest chain / |T|

    def __init__(self, min_support: float = 0.5, max_pattern_size: int = 0,
                 use_rc_pruning: bool = False, **kwargs):
        super().__init__(min_support=min_support, **kwargs)
        self.max_pattern_size = max_pattern_size
        self.use_rc_pruning = use_rc_pruning
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
        raw_results: list = []

        for e in range(0, n_enc, 2):
            args = (e, all_bms[e], all_bms, all_bms_stack, min_supp, self.max_pattern_size, self.use_rc_pruning)
            raw_results.extend(_mine_root(args))

        patterns = _results_to_patterns(raw_results, enc_to_attr)
        logger.info(f"GLCM found {len(patterns)} closed frequent gradual patterns")
        return patterns
