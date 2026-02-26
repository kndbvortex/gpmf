"""GLCM - LCM-based mining of closed frequent gradual itemsets.

Reference:
    Do, T.D.T., Termier, A., Laurent, A., Negrevergne, B., Jeudy, B., &
    Gacias, B. (2015). PGLCM: Efficient Parallel Mining of Closed Frequent
    Gradual Itemsets. Knowledge and Information Systems, 43(3), 681-737.

Support definition (Di-Jorio et al. 2009):
    support(P) = length of longest ordered transaction sequence / |T|
    where a sequence t1 < t2 < ... < tk satisfies P iff BM_P[ti, ti+1] = 1
    for all consecutive pairs. This is the longest path in the DAG / n.

    Note: this differs from GRAANK's pair-fraction support. A single attribute
    always has support 1.0 (all transactions can be ordered by it), so patterns
    of length ≥ 2 are the meaningful output.
"""
import logging
from typing import List, Tuple

import numpy as np

from ..base_algorithm import BaseAlgorithm
from ...core.data_structures import GradualItem, GradualPattern

logger = logging.getLogger(__name__)


def _build_bm(col: np.ndarray) -> np.ndarray:
    """Build ascending binary matrix for one attribute column.

    bm[i,j] = True iff col[i] <= col[j] and i != j.
    Equal values produce bidirectional edges (cycles), resolved before
    support computation by _set_contrast_to_zero.
    """
    bm = col[:, None] <= col[None, :]
    np.fill_diagonal(bm, False)
    return bm


def _set_contrast_to_zero(bm: np.ndarray) -> np.ndarray:
    """Resolve equal-value cycles to produce a DAG.

    When bm[i,j]=1 and bm[j,i]=1 (equal values), keeps only bm[i,j]
    (upper-triangle edge, i < j in iteration order).
    """
    result = bm.copy()
    cycles = bm & bm.T
    result &= ~np.tril(cycles, -1)
    return result


def _prune_matrix(bm: np.ndarray, min_chain: int) -> np.ndarray:
    """Iteratively zero rows/cols where ML_p + MC_p < min_chain − 1.

    Proposition 1 (Kamga Nguifo, Lonlac, Fleury & Mephu Nguifo, 2025):
    keep transaction tp iff row_sum(tp) + col_sum(tp) ≥ min_chain − 1.
    The matrix shape is preserved so support = longest_path / n_original
    is computed correctly.

    IMPORTANT: only the return value should be passed to _compute_support.
    The original bm must still be used for closure and recursion, otherwise
    the closure operator F(bm_pruned) ⊇ F(bm_orig) which can produce
    spurious closed patterns.
    """
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
    """Compute support as longest path length / n in the DAG.

    Uses vectorised topological-sort DP. Processes all nodes at the same
    depth simultaneously, so the number of Python iterations equals the
    DAG depth rather than the number of edges.
    """
    n = bm.shape[0]
    dag = _set_contrast_to_zero(bm)
    dp = np.ones(n, dtype=np.int32)
    in_deg = dag.sum(axis=0, dtype=np.int32)
    processed = np.zeros(n, dtype=bool)

    while True:
        ready = (in_deg == 0) & ~processed
        if not ready.any():
            break
        ready_dp = dp[ready][:, None]          # (n_ready, 1)
        succ = dag[ready]                       # (n_ready, n) successors mask
        if succ.any():
            updates = np.where(succ, ready_dp + 1, 0).max(axis=0)
            np.maximum(dp, updates, out=dp)
            in_deg -= succ.sum(axis=0, dtype=np.int32)
        processed |= ready

    return int(dp.max()) / n


def _compute_closure(
    bm: np.ndarray,
    all_bms_stack: np.ndarray,
) -> Tuple[int, ...]:
    """Compute F(bm): all encoded items e such that bm ⊆ all_bms_stack[e].

    Uses a single vectorised comparison over all items at once.
    all_bms_stack is a (k, n, n) boolean array.
    """
    # violations[e, i, j] = True where bm[i,j]=1 but all_bms_stack[e,i,j]=0
    violations = bm[None, :, :] > all_bms_stack
    in_closure = ~violations.any(axis=(1, 2))
    return tuple(int(i) for i in np.where(in_closure)[0])


def _ppc_ok(P1: frozenset, Q: Tuple[int, ...], core_i: int) -> bool:
    """Verify the ppc-extension condition.

    Returns True iff Q contains no items with encoding < core_i that are
    not already in P1. Ensures each closed pattern is discovered exactly
    once via its canonical covering-tree path.
    """
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
    """Enumerate all ppc-extensions of closed pattern P.

    For each candidate item e > core_i:
    1. Skip if e or its opposite is already in P.
    2. Compute bm1 = bm & B_e and check support.
    3. Compute closure Q = F(bm1) and verify ppc condition.
    4. Record Q and recurse.
    """
    n_enc = len(all_bms)
    n = bm.shape[0]
    # A chain of length ceil(min_supp * n) needs at least l*(l-1)/2 edges.
    min_chain = int(np.ceil(min_supp * n))
    min_edges = min_chain * (min_chain - 1) // 2

    for e in range(core_i + 1, n_enc):
        if e in P or (e ^ 1) in P:
            continue

        bm1 = bm & all_bms[e]
        if int(bm1.sum()) < min_edges:
            continue  # cannot possibly meet the support threshold

        bm1_for_supp = _prune_matrix(bm1, min_chain) if use_rc_pruning else bm1
        supp = _compute_support(bm1_for_supp)
        if supp < min_supp:
            continue

        P1 = P | {e}
        Q = _compute_closure(bm1, all_bms_stack)  # original bm1, not pruned
        if not _ppc_ok(P1, Q, e):
            continue

        results.append((Q, supp))
        if max_size and len(Q) >= max_size:
            continue
        _glcm_loop(frozenset(Q), bm1, e, min_supp, all_bms, all_bms_stack, results, max_size, use_rc_pruning)


def _mine_root(args: tuple) -> list:
    """Mine all closed patterns rooted at a single starting item.

    Module-level so it is picklable for use with ProcessPoolExecutor.
    args = (e, bm_e, all_bms, all_bms_stack, min_supp, max_size, use_rc_pruning)
    """
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


def _results_to_patterns(
    raw_results: list,
    enc_to_attr: List[Tuple[int, str]],
) -> List[GradualPattern]:
    """Convert raw (Q, supp) pairs to GradualPattern objects.

    Filters out:
    - Patterns with fewer than 2 items.
    - Patterns containing both ascending and descending of the same attribute
      (can arise from degenerate equal-value cases).
    """
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
    """Build binary matrices for all encoded items.

    Columns where all values are equal (constant attributes) produce
    all-True binary matrices, which degenerate the search space.  Such
    columns are excluded from the encoded item list.

    Returns:
        all_bms      : list of (n, n) bool arrays indexed by encoding
        enc_to_attr  : list of (dataset_col_index, symbol) per encoding
        all_bms_stack: (k, n, n) numpy array for vectorised closure checks
    """
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

    all_bms_stack = np.stack(all_bms)  # (k, n, n)
    return all_bms, enc_to_attr, all_bms_stack


class GLCM(BaseAlgorithm):
    """GLCM algorithm for mining closed frequent gradual itemsets.

    Adapts the LCM (Linear time Closed itemset Miner) algorithm to gradual
    itemset mining. Uses binary matrices and a closure operator to enumerate
    all closed frequent gradual patterns without redundancy.

    Support definition: support(P) = longest ordered transaction chain / |T|.
    A single attribute always has support 1.0. Use min_support < 1.0 for
    relative threshold (fraction of transactions), or >= 1 for absolute
    (minimum chain length).

    On datasets with many correlated or binary features the search space can
    grow exponentially. Use max_pattern_size to cap the pattern length and
    bound runtime, or compile the optional Rust extension for best performance.

    Args:
        min_support:      Minimum support threshold (0-1 relative, >1 absolute)
        max_pattern_size: Stop extending patterns longer than this (0 = no limit)

    Example:
        >>> glcm = GLCM(min_support=0.5)
        >>> patterns = glcm.mine('data.csv')
    """

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
