"""SGrite algorithm for gradual pattern mining.

Reference:
    Tayou Djamegni, C., Tabueu Fotso, L.C., & Kenmogne, E.B. (2021).
    A novel algorithm for extracting frequent gradual patterns.
    Machine Learning with Applications, 5, 100068.

Support definition (strict ordering):
    SG(M) = longest path length (in nodes) in the strict precedence DAG / |T|
    where adj[i,j] = True iff col[i] < col[j] for every attribute in M.
    Strict inequality (unlike GRITE which uses ≤). Equal values share no edge.

Key improvements over GRITE (Di-Jorio et al., 2009):
    1. Single bottom-up sweep of the DAG (GRITE does two sweeps).
    2. Search space halved via Lemma 1: SG(M) = SG(complement(M)).

Four variants (Section 3 of the paper):

    ========  =========  =========================================================
    Variant   Traversal  Notes
    ========  =========  =========================================================
    sgopt     DFS/stack  Algorithm 2. May re-push nodes when Memory is updated.
    sg1       BFS/queue  Algorithm 2 with queue. Re-enqueues on update.
    sgb1      DFS/stack  Algorithm 3 (blocking). Each node processed exactly once.
    sgb2      BFS/queue  Algorithm 3 (blocking) with queue. Each node once.
    ========  =========  =========================================================

Empirically (paper, Section 4.2.1): sgb1, sg1, sgopt are the fastest.
"""
import logging
import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from .base_algorithm import BaseAlgorithm
from ..core.data_structures import GradualItem, GradualPattern

logger = logging.getLogger(__name__)

_VARIANTS = frozenset({'sgopt', 'sg1', 'sgb1', 'sgb2'})


# ---------------------------------------------------------------------------
# Support computation — Algorithms 2 and 3 from the paper
# ---------------------------------------------------------------------------

def _leaf_indices(adj: np.ndarray) -> np.ndarray:
    """Return indices of leaf nodes.

    A leaf has no sons (row sum = 0) and at least one parent (col sum > 0).
    Isolated nodes (row=0 and col=0) are excluded.
    """
    return np.where((adj.sum(axis=1) == 0) & (adj.sum(axis=0) > 0))[0]


def _support_alg2(adj: np.ndarray, use_queue: bool) -> float:
    """Algorithm 2: single-sweep bottom-up with re-propagation.

    Starts from leaves, propagates Memory values upward to parents.
    A parent can be (re-)pushed each time its Memory value improves.

    sgopt: use_queue=False (DFS / stack)
    sg1:   use_queue=True  (BFS / queue)
    """
    n = adj.shape[0]
    if n == 0:
        return 0.0

    leaves = _leaf_indices(adj)
    if len(leaves) == 0:
        return 1.0 / n

    memory = np.zeros(n, dtype=np.int32)
    memory[leaves] = 1

    parents: List[np.ndarray] = [np.where(adj[:, e])[0] for e in range(n)]

    container: deque = deque(int(x) for x in leaves)
    while container:
        e = container.popleft() if use_queue else container.pop()
        val = int(memory[e]) + 1
        for p in parents[e]:
            if memory[p] < val:
                memory[p] = val
                container.append(int(p))

    return int(memory.max()) / n


def _support_alg3(adj: np.ndarray, use_stack: bool) -> float:
    """Algorithm 3: blocking single-sweep — each node processed exactly once.

    A parent is enqueued only when ALL its sons have been processed,
    eliminating redundant updates. Memory is updated incrementally as each
    son is processed; the final value is correct when the node is dequeued.

    sgb2: use_stack=False (BFS / queue)
    sgb1: use_stack=True  (DFS / stack)
    """
    n = adj.shape[0]
    if n == 0:
        return 0.0

    leaves = _leaf_indices(adj)
    if len(leaves) == 0:
        return 1.0 / n

    memory = np.zeros(n, dtype=np.int32)
    memory[leaves] = 1

    # pending[p] = number of sons of p not yet processed.
    pending = adj.sum(axis=1, dtype=np.int32).copy()  # row sum = #sons of node

    parents: List[np.ndarray] = [np.where(adj[:, e])[0] for e in range(n)]

    container: deque = deque(int(x) for x in leaves)
    while container:
        e = container.pop() if use_stack else container.popleft()
        val = int(memory[e]) + 1
        for p in parents[e]:
            if memory[p] < val:
                memory[p] = val
            pending[p] -= 1
            if pending[p] == 0:
                container.append(int(p))

    return int(memory.max()) / n


class _Pat:
    """Lightweight internal pattern: sorted item tuple + combined adj matrix."""

    __slots__ = ('items', 'adj', 'support')

    def __init__(
        self,
        items: Tuple[Tuple[int, str], ...],
        adj: np.ndarray,
        support: float = 0.0,
    ) -> None:
        self.items = items    # sorted tuple of (col_idx, symbol) pairs
        self.adj = adj        # (n, n) bool numpy array
        self.support = support

    def key(self) -> frozenset:
        return frozenset(f"{i}{s}" for i, s in self.items)

    def complement(self) -> '_Pat':
        """Return complement pattern: flip all symbols, transpose adj matrix."""
        comp_items = tuple((i, '-' if s == '+' else '+') for i, s in self.items)
        return _Pat(items=comp_items, adj=self.adj.T.copy(), support=self.support)


def _prune_matrix(adj: np.ndarray, min_chain: int) -> np.ndarray:
    """Iteratively zero rows/cols where ML_p + MC_p < min_chain − 1.

    Proposition 1 (Kamga Nguifo, Lonlac, Fleury & Mephu Nguifo, 2025):
    keep transaction tp iff row_sum(tp) + col_sum(tp) ≥ min_chain − 1,
    where row_sum = number of successors and col_sum = number of predecessors
    in the precedence DAG.  Rows/cols of pruned transactions are zeroed so
    the matrix shape stays fixed and support = longest_path / n_original is
    still computed correctly.
    """
    if min_chain <= 1:
        return adj
    threshold = min_chain - 1
    result = adj.copy()
    while True:
        row_s = result.sum(axis=1)
        col_s = result.sum(axis=0)
        keep = (row_s + col_s) >= threshold
        if keep.all():
            break
        result[~keep, :] = False
        result[:, ~keep] = False
    return result


def _join(p1: _Pat, p2: _Pat) -> Optional[_Pat]:
    """Apriori join of two k-patterns into a (k+1)-candidate.

    Valid iff the symmetric difference contains exactly 2 items belonging
    to different attributes. The combined adj matrix is the bitwise AND of
    the two parent matrices (correct since adj(P1 ∪ P2) = adj(P1) & adj(P2)
    when items in the intersection appear in both operands).

    Returns None if the join is invalid.
    """
    s1, s2 = set(p1.items), set(p2.items)
    diff = s1.symmetric_difference(s2)
    if len(diff) != 2:
        return None
    (c1, _), (c2, _) = tuple(diff)
    if c1 == c2:
        # Same attribute with opposite directions — conflict.
        return None
    return _Pat(items=tuple(sorted(s1 | s2)), adj=p1.adj & p2.adj)


class SGrite(BaseAlgorithm):
    """SGrite: improved single-sweep gradual pattern miner.

    Improves on GRITE by computing gradual support in a single bottom-up
    sweep of the strict-ordering precedence DAG instead of GRITE's two-sweep
    recursive RecursiveCorvering. The search space is halved by Lemma 1
    (a pattern and its complement have identical support).

    Uses strict ordering (< / >) in the precedence graph. Unlike GRITE
    (which uses ≤ / ≥), equal values produce no edge in the DAG. On datasets
    with many ties this can yield lower support values than GRITE for the
    same patterns.

    Four variants select the traversal strategy for the support computation:

    ========  =========  =========================================================
    Variant   Traversal  Notes
    ========  =========  =========================================================
    sgopt     DFS/stack  Algorithm 2. May re-push nodes when Memory is updated.
    sg1       BFS/queue  Algorithm 2 with queue. Re-enqueues on update.
    sgb1      DFS/stack  Algorithm 3 (blocking). Each node processed exactly once.
    sgb2      BFS/queue  Algorithm 3 (blocking) with queue. Each node once.
    ========  =========  =========================================================

    All four variants produce identical output; sgb1 and sg1 tend to be fastest
    (Tayou Djamegni et al., 2021, Section 4.2.1).

    Args:
        min_support: Minimum support threshold (0-1 relative, >1 absolute).
        variant:     Support-computation variant (default ``'sgb1'``).

    Example:
        >>> sgrite = SGrite(min_support=0.5, variant='sgb1')
        >>> patterns = sgrite.mine('data.csv')
    """

    def __init__(
        self,
        min_support: float = 0.5,
        variant: str = 'sgb1',
        use_rc_pruning: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(min_support=min_support, **kwargs)
        if variant not in _VARIANTS:
            raise ValueError(
                f"variant must be one of {sorted(_VARIANTS)}, got '{variant}'"
            )
        self.variant = variant
        self.use_rc_pruning = use_rc_pruning
        self._params['variant'] = variant
        self._params['use_rc_pruning'] = use_rc_pruning

    # ------------------------------------------------------------------
    # Support dispatch
    # ------------------------------------------------------------------

    def _compute_support(self, adj: np.ndarray) -> float:
        v = self.variant
        if v == 'sgopt':
            return _support_alg2(adj, use_queue=False)
        if v == 'sg1':
            return _support_alg2(adj, use_queue=True)
        if v == 'sgb2':
            return _support_alg3(adj, use_stack=False)
        return _support_alg3(adj, use_stack=True)  # sgb1


    def _mine(self) -> List[GradualPattern]:
        data = self.dataset.data.astype(float)
        n = data.shape[0]
        attr_cols = self.dataset.attr_cols

        min_supp = (
            self.min_support / n if self.min_support >= 1 else self.min_support
        )
        min_chain = math.ceil(min_supp * n)

        level: List[_Pat] = []
        for col_idx in attr_cols:
            col = data[:, col_idx]
            adj_asc = col[:, None] < col[None, :]   # strict ascending DAG
            supp = self._compute_support(adj_asc)
            if supp < min_supp:
                continue
            pat = _Pat(items=((int(col_idx), '+'),), adj=adj_asc, support=supp)
            level.append(pat)
            level.append(pat.complement())

        logger.debug(
            f"SGrite ({self.variant}): {len(level) // 2} frequent 1-item attributes"
        )

        # ---- Levels 2, 3, … --------------------------------------------
        all_frequent: List[_Pat] = []
        k = 2
        while level:
            next_level: List[_Pat] = []
            seen: set = set()

            for i in range(0, len(level), 2):
                p1 = level[i]

                for j in range(i + 2, len(level)):
                    p2 = level[j]
                    cand = _join(p1, p2)
                    if cand is None:
                        continue

                    key = cand.key()
                    if key in seen:
                        continue
                    seen.add(key)


                    adj = _prune_matrix(cand.adj, min_chain) if self.use_rc_pruning else cand.adj
                    cand.adj = adj
                    non_iso = int(
                        ((adj.sum(axis=1) > 0) | (adj.sum(axis=0) > 0)).sum()
                    )
                    if non_iso < min_chain:
                        continue

                    supp = self._compute_support(adj)
                    if supp < min_supp:
                        continue

                    cand.support = supp
                    comp = cand.complement()
                    seen.add(comp.key())

                    next_level.append(cand)
                    next_level.append(comp)
                    all_frequent.append(cand)
                    all_frequent.append(comp)

            logger.debug(
                f"SGrite ({self.variant}): {len(next_level) // 2} patterns at level {k}"
            )
            level = next_level
            k += 1

        patterns: List[GradualPattern] = []
        for pat in all_frequent:
            gp = GradualPattern()
            gp.set_support(pat.support)
            for col_idx, symbol in pat.items:
                gp.add_gradual_item(GradualItem(col_idx, symbol))
            patterns.append(gp)

        logger.info(
            f"SGrite ({self.variant}) found {len(patterns)} frequent gradual patterns"
        )
        return patterns
