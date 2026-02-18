"""GRITE algorithm for gradual pattern mining.

GRITE (GRadual Itemsets ExTraction) uses precedence graphs and
dynamic covering to mine gradual patterns efficiently.

Reference:
    Di-Jorio, L., Laurent, A., & Teisseire, M. (2009). Mining Frequent
    Gradual Itemsets from Large Databases. IDA 2009.
"""
import math
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Set
from dataclasses import dataclass

from .base_algorithm import BaseAlgorithm
from ..core.data_structures import GradualItem, GradualPattern
from ..utils.validators import validate_n_jobs

logger = logging.getLogger(__name__)


@dataclass
class GritePattern:
    col_names: List[str]
    variations: List[str]
    bin_mat: pd.DataFrame
    freq: float = 0.0
    longest_path: List[int] = None

    def __post_init__(self):
        if self.longest_path is None:
            self.longest_path = []

    def names(self) -> List[str]:
        return [f"{name}{var}" for name, var in zip(self.col_names, self.variations)]

    def complement(self) -> 'GritePattern':
        comp_map = {'+': '-', '-': '+'}
        c = ~self.bin_mat

        for i in range(len(c)):
            c.iloc[i, i] = False

        for i in range(len(c)):
            for j in range(i):
                if self.bin_mat.iloc[i, j] and self.bin_mat.iloc[j, i]:
                    c.iloc[i, j] = True
                    c.iloc[j, i] = True

        comp = GritePattern(
            col_names=self.col_names.copy(),
            variations=[comp_map[v] for v in self.variations],
            bin_mat=c,
            freq=self.freq
        )
        comp.longest_path = self.longest_path.copy() if self.longest_path else []
        return comp

    def _compute_sons(self):
        """Compute sons and roots using original GRITE logic.

        This handles bidirectional edges that occur when values are equal.
        """
        columns = self.bin_mat.columns
        self._sons = {col: [] for col in columns}
        s = set()  # Set of all nodes that are children of someone

        for i, col in enumerate(columns):
            for j in columns[i + 1:]:
                o = False
                if self.bin_mat.loc[col, j]:
                    self._sons[col].append(j)
                    s.add(j)
                    o = True
                if self.bin_mat.loc[j, col]:
                    if not o:
                        self._sons[j].append(col)
                        s.add(col)

        # Roots are nodes that are never children of anyone
        self._roots = list(set(columns).difference(s))

    @property
    def sons(self):
        if not hasattr(self, '_sons') or self._sons is None:
            self._compute_sons()
        return self._sons

    @property
    def roots(self):
        if not hasattr(self, '_roots'):
            self._compute_sons()
        return self._roots

    def init_memory(self):
        """Initialize memory for dynamic covering."""
        self.memory = {col: -1 for col in self.bin_mat.columns}
        self.paths = {col: [] for col in self.bin_mat.columns}

    def dynamic_covering(self, node, memory: dict, paths: dict):
        """Compute longest path using iterative DFS with visited tracking."""
        stack = [node]
        visited = set()

        while stack:
            current_node = stack[-1]

            if current_node not in visited:
                visited.add(current_node)
                # Push unvisited children onto stack
                for son in self.sons[current_node]:
                    if memory[son] == -1:
                        stack.append(son)
            else:
                # All children processed, update depth and path for current node
                max_child_depth = 0
                best_child = None
                for son in self.sons[current_node]:
                    if memory[son] > max_child_depth:
                        max_child_depth = memory[son]
                        best_child = son

                memory[current_node] = max_child_depth + 1

                # Build path: current node + best child's path
                if best_child is not None:
                    paths[current_node] = [current_node] + paths[best_child]
                else:
                    paths[current_node] = [current_node]

                stack.pop()

        return memory[node]

    def compute_frequency(self, item_number: int):
        """Compute frequency as longest path / number of items."""
        _ = self.sons
        for root in self.roots:
            self.init_memory()
            self.dynamic_covering(root, self.memory, self.paths)
            max_node = max(self.memory, key=self.memory.get)
            freq = self.memory[max_node] / item_number
            if freq > self.freq:
                self.freq = freq
                self.longest_path = self.paths[max_node]


class GRITE(BaseAlgorithm):
    """GRITE algorithm for mining gradual patterns.

    Uses precedence graphs and dynamic covering for efficient pattern extraction.

    Args:
        min_support: Minimum support threshold (0-1 relative, >1 absolute).
            The minimum chain length is derived as ceil(min_support × n).
        n_jobs: Number of parallel jobs (not used in current implementation)
        use_rc_pruning: Apply Row–Column pruning (Kamga Nguifo et al., 2025)
            to speed up mining. Off by default; enable to benchmark.

    Example:
        >>> grite = GRITE(min_support=0.5)
        >>> patterns = grite.mine('data.csv')
    """

    def __init__(self, min_support: float = 0.5, n_jobs: int = 1,
                 use_rc_pruning: bool = False, **kwargs):
        super().__init__(min_support, **kwargs)
        self.n_jobs = validate_n_jobs(n_jobs)
        self.use_rc_pruning = use_rc_pruning
        self._params['n_jobs'] = self.n_jobs
        self._params['use_rc_pruning'] = use_rc_pruning
        self._relative_support = None  # Will be set in _mine()

    def _from_order_to_binary_matrix(self, col: pd.Series) -> pd.DataFrame:
        n = len(col)
        mat = pd.DataFrame(
            data=np.zeros((n, n), dtype=bool),
            columns=list(range(n)),
            index=list(range(n))
        )

        for i_, i in enumerate(col.index):
            for j in col.index[i_ + 1:]:
                mat.loc[i, j] = True
                if col[i] == col[j]:
                    mat.loc[j, i] = True

        return mat

    def _sublist(self, sub_list: List, liste: List) -> bool:
        return set(sub_list).issubset(set(liste))

    def _join(
        self,
        item1: GritePattern,
        item2: GritePattern,
        already_generated: Set[Tuple],
        gradual_patterns: List[GritePattern],
        k: int,
        min_chain: int = 2,
    ) -> Tuple[bool, GritePattern]:
        name1, name2 = item1.names(), item2.names()
        diff = list(set(name1).difference(name2).union(set(name2).difference(name1)))

        if len(diff) != 2:
            return False, None

        if diff[0][:-1] == diff[1][:-1]:
            return False, None

        name_types = list(
            set(zip(item1.col_names, item1.variations)).union(
                zip(item2.col_names, item2.variations)
            )
        )
        name_types = sorted(name_types, key=lambda x: x[0])
        names = [x[0] for x in name_types]
        types = [x[1] for x in name_types]

        candidate_name = tuple(f"{n}{t}" for n, t in zip(names, types))
        if candidate_name in already_generated:
            return False, None

        result = pd.DataFrame(
            data=np.ones_like(item1.bin_mat, dtype=bool),
            columns=item1.bin_mat.columns,
            index=item1.bin_mat.index
        )

        considered = 0
        for grad_p in gradual_patterns:
            if self._sublist(grad_p.names(), list(candidate_name)):
                result = result & grad_p.bin_mat
                considered += 1

        if considered != k:
            return False, None

        result = result.fillna(False)

        # Row–Column pruning (Kamga Nguifo et al., 2025, Proposition 1):
        # iteratively remove transactions tp where ML_p + MC_p < min_chain − 1.
        threshold = min_chain - 1
        if self.use_rc_pruning and threshold > 0:
            while True:
                vals = result.values
                row_s = vals.sum(axis=1)
                col_s = vals.sum(axis=0)
                to_keep = (row_s + col_s) >= threshold
                if to_keep.all():
                    break
                drop_labels = result.index[~to_keep].tolist()
                if not drop_labels:
                    break
                result = result.drop(index=drop_labels, columns=drop_labels)
                if result.shape[0] == 0:
                    break

        if result.shape[0] < min_chain:
            return False, None

        return True, GritePattern(col_names=names, variations=types, bin_mat=result)

    def _generate_k_patterns(
        self,
        gradual_items: List[GritePattern],
        item_number: int,
        k: int
    ) -> List[GritePattern]:
        if len(gradual_items) < k:
            return []

        gradual_k = []
        already_generated = set()
        min_chain = max(2, math.ceil(self._relative_support * item_number))

        for i in range(0, len(gradual_items), 2):
            for j in range(i + 2, len(gradual_items)):
                is_possible, candidate = self._join(
                    gradual_items[i],
                    gradual_items[j],
                    already_generated,
                    gradual_items,
                    k,
                    min_chain=min_chain,
                )

                if not is_possible:
                    continue

                candidate.compute_frequency(item_number)

                if candidate.freq >= self._relative_support:
                    gradual_k.append(candidate)
                    already_generated.add(tuple(candidate.names()))

                    comp = candidate.complement()
                    comp.freq = candidate.freq
                    gradual_k.append(comp)
                    already_generated.add(tuple(comp.names()))

        logger.debug(f"Generated {len(gradual_k)} patterns of size {k}")
        return gradual_k

    def _mine(self) -> List[GradualPattern]:
        df = pd.DataFrame(self.dataset.data, columns=[
            self.dataset.titles[i][1].decode() if hasattr(self.dataset.titles[i][1], 'decode')
            else str(self.dataset.titles[i][1])
            for i in range(self.dataset.col_count)
        ])

        num_rows = df.shape[0]
        if self.min_support < 1:
            self._relative_support = self.min_support
            logger.info(f"Using relative support: {self._relative_support}")
        else:
            self._relative_support = self.min_support / num_rows
            logger.info(f"Converting absolute support {self.min_support} to relative: {self._relative_support} "
                       f"(threshold: {int(self.min_support)}/{num_rows} rows)")

        gradual_items = {1: []}

        for col in df.columns:
            if col in [self.dataset.titles[i][1].decode() if hasattr(self.dataset.titles[i][1], 'decode') else str(self.dataset.titles[i][1]) for i in self.dataset.attr_cols]:
                s = self._from_order_to_binary_matrix(df[col].sort_values())
                grad = GritePattern(col_names=[col], variations=['+'], bin_mat=s)
                grad.compute_frequency(df.shape[0])

                gradual_items[1].append(grad)

                comp = grad.complement()
                comp.freq = grad.freq
                gradual_items[1].append(comp)

        logger.info(f"Generated {len(gradual_items[1])} 1-patterns")

        k = 2
        while gradual_items[k - 1]:
            gradual_items[k] = self._generate_k_patterns(
                gradual_items[k - 1],
                df.shape[0],
                k
            )
            k += 1

        all_patterns = []
        for size in range(2, k):
            if size in gradual_items:
                for grite_pattern in gradual_items[size]:
                    if grite_pattern.freq >= self._relative_support:
                        gp = GradualPattern()

                        col_name_to_idx = {}
                        for i, title in enumerate(self.dataset.titles):
                            name = title[1].decode() if hasattr(title[1], 'decode') else str(title[1])
                            col_name_to_idx[name] = i

                        for col_name, var in zip(grite_pattern.col_names, grite_pattern.variations):
                            if col_name in col_name_to_idx:
                                gi = GradualItem(col_name_to_idx[col_name], var)
                                gp.add_gradual_item(gi)

                        gp.set_support(grite_pattern.freq)
                        gp.path = grite_pattern.longest_path if grite_pattern.longest_path else []
                        all_patterns.append(gp)

        logger.info(f"GRITE found {len(all_patterns)} patterns")
        return all_patterns
