# SGrite: improved GRITE using single bottom-up DAG sweep + Lemma 1 symmetry
# Variants: sgopt (DFS/alg2), sg1 (BFS/alg2), sgb1 (DFS/alg3), sgb2 (BFS/alg3)
import logging
import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from .base_algorithm import BaseAlgorithm
from ..core.data_structures import GradualItem, GradualPattern

logger = logging.getLogger(__name__)

_VARIANTS = frozenset({'sgopt', 'sg1', 'sgb1', 'sgb2'})


def _leaf_indices(adj: np.ndarray) -> np.ndarray:
    return np.where((adj.sum(axis=1) == 0) & (adj.sum(axis=0) > 0))[0]


def _support_alg2(adj: np.ndarray, use_queue: bool) -> float:
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
    n = adj.shape[0]
    if n == 0:
        return 0.0

    leaves = _leaf_indices(adj)
    if len(leaves) == 0:
        return 1.0 / n

    memory = np.zeros(n, dtype=np.int32)
    memory[leaves] = 1

    pending = adj.sum(axis=1, dtype=np.int32).copy()
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
    __slots__ = ('items', 'adj', 'support')

    def __init__(self, items: Tuple[Tuple[int, str], ...], adj: np.ndarray, support: float = 0.0) -> None:
        self.items = items
        self.adj = adj
        self.support = support

    def key(self) -> frozenset:
        return frozenset(f"{i}{s}" for i, s in self.items)

    def complement(self) -> '_Pat':
        comp_items = tuple((i, '-' if s == '+' else '+') for i, s in self.items)
        return _Pat(items=comp_items, adj=self.adj.T.copy(), support=self.support)


def _prune_matrix(adj: np.ndarray, min_chain: int) -> np.ndarray:
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
    s1, s2 = set(p1.items), set(p2.items)
    diff = s1.symmetric_difference(s2)
    if len(diff) != 2:
        return None
    (c1, _), (c2, _) = tuple(diff)
    if c1 == c2:
        return None
    return _Pat(items=tuple(sorted(s1 | s2)), adj=p1.adj & p2.adj)


class SGrite(BaseAlgorithm):
    # Strict-ordering precedence DAG miner; SG(M) = SG(complement(M)) halves search space

    def __init__(self, min_support: float = 0.5, variant: str = 'sgb1',
                 use_rc_pruning: bool = False, **kwargs) -> None:
        super().__init__(min_support=min_support, **kwargs)
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {sorted(_VARIANTS)}, got '{variant}'")
        self.variant = variant
        self.use_rc_pruning = use_rc_pruning
        self._params['variant'] = variant
        self._params['use_rc_pruning'] = use_rc_pruning

    def _compute_support(self, adj: np.ndarray) -> float:
        v = self.variant
        if v == 'sgopt':
            return _support_alg2(adj, use_queue=False)
        if v == 'sg1':
            return _support_alg2(adj, use_queue=True)
        if v == 'sgb2':
            return _support_alg3(adj, use_stack=False)
        return _support_alg3(adj, use_stack=True)

    def _mine(self) -> List[GradualPattern]:
        data = self.dataset.data.astype(float)
        n = data.shape[0]
        attr_cols = self.dataset.attr_cols

        min_supp = self.min_support / n if self.min_support >= 1 else self.min_support
        min_chain = math.ceil(min_supp * n)

        level: List[_Pat] = []
        for col_idx in attr_cols:
            col = data[:, col_idx]
            adj_asc = col[:, None] < col[None, :]
            supp = self._compute_support(adj_asc)
            if supp < min_supp:
                continue
            pat = _Pat(items=((int(col_idx), '+'),), adj=adj_asc, support=supp)
            level.append(pat)
            level.append(pat.complement())

        logger.debug(f"SGrite ({self.variant}): {len(level) // 2} frequent 1-item attributes")

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
                    non_iso = int(((adj.sum(axis=1) > 0) | (adj.sum(axis=0) > 0)).sum())
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

            logger.debug(f"SGrite ({self.variant}): {len(next_level) // 2} patterns at level {k}")
            level = next_level
            k += 1

        patterns: List[GradualPattern] = []
        for pat in all_frequent:
            gp = GradualPattern()
            gp.set_support(pat.support)
            for col_idx, symbol in pat.items:
                gp.add_gradual_item(GradualItem(col_idx, symbol))
            patterns.append(gp)

        logger.info(f"SGrite ({self.variant}) found {len(patterns)} frequent gradual patterns")
        return patterns
