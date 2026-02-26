import math
import sys
import numpy as np
from typing import List, Set, Tuple, Optional

from .datatypes import TransactionTable, GradualItem, GradualPattern, Variation
from . import RUST_AVAILABLE, RustGradualMiner
from .bool_matrix import BoolMatrix
from .support import compute_gradual_support


class GradualMiner:

    def __init__(
        self,
        min_support: float,
        num_threads: Optional[int] = None,
        use_rust: bool = True,
        verbose: bool = False,
    ):
        self.min_support = min_support
        self.num_threads = num_threads
        self.verbose = verbose
        self.use_rust = use_rust and RUST_AVAILABLE

        if use_rust and not RUST_AVAILABLE:
            print("Warning: Rust acceleration requested but not available. Using Python implementation.")
            print("To enable Rust: maturin develop --release")
            self.use_rust = False

        if self.use_rust and RustGradualMiner is not None:
            self._rust_miner = RustGradualMiner(min_support, num_threads, verbose)
            self._use_full_rust = True
        else:
            self._use_full_rust = False
            self.transaction_table: TransactionTable = None
            self.num_attributes = 0
            self.num_transactions = 0
            self.num_vtrans = 0
            self.threshold = 0
            self.vtid_to_pair: np.ndarray = None
            self.item_to_vtids: List[np.ndarray] = []
            self.item_matrices: List[BoolMatrix] = []
            self.patterns: List[GradualPattern] = []
            self.found_patterns: Set[Tuple[int, ...]] = set()

    def load_data(self, filename: str):
        table = TransactionTable.from_file(filename)
        if self._use_full_rust:
            transactions = [list(t.values) for t in table.transactions]
            self._rust_miner.load_transactions(transactions, table.num_attributes)
        else:
            self.transaction_table = table
            self.num_transactions = len(self.transaction_table)
            self.num_attributes = self.transaction_table.num_attributes
        if not self._use_full_rust or self.verbose:
            print(f"Loaded {len(table.transactions)} transactions with {table.num_attributes} attributes")

    def _build_from_data_numpy(self):
        N = self.num_transactions
        num_attrs = self.num_attributes
        num_items = num_attrs * 2

        all_i = np.repeat(np.arange(N, dtype=np.int32), N)
        all_j = np.tile(np.arange(N, dtype=np.int32), N)
        diag_mask = all_i != all_j
        i_pairs = all_i[diag_mask]
        j_pairs = all_j[diag_mask]
        self.num_vtrans = len(i_pairs)
        self.vtid_to_pair = np.stack([i_pairs, j_pairs], axis=1)

        data = np.array([self.transaction_table[r].values for r in range(N)], dtype=np.float64)

        self.item_to_vtids = [None] * num_items
        self.item_matrices = [BoolMatrix(N) for _ in range(num_items)]

        for k in range(num_attrs):
            val_i = data[i_pairs, k]
            val_j = data[j_pairs, k]

            item_inc = (val_i < val_j) | ((val_i == val_j) & (i_pairs < j_pairs))
            item_dec = (val_i > val_j) | ((val_i == val_j) & (i_pairs > j_pairs))

            self.item_to_vtids[2 * k]     = item_inc
            self.item_to_vtids[2 * k + 1] = item_dec

            if item_inc.any():
                self.item_matrices[2 * k].data[j_pairs[item_inc], i_pairs[item_inc]] = True
            if item_dec.any():
                self.item_matrices[2 * k + 1].data[j_pairs[item_dec], i_pairs[item_dec]] = True

    def _build_matrix_from_vtids(self, vtids: np.ndarray):
        vtid_indices = np.where(vtids)[0]
        if len(vtid_indices) == 0:
            return None, 0, None

        pairs = self.vtid_to_pair[vtid_indices]
        unique_tids = np.unique(pairs)
        num_tids = len(unique_tids)

        tid_to_idx = np.full(self.num_transactions, -1, dtype=np.int32)
        tid_to_idx[unique_tids] = np.arange(num_tids, dtype=np.int32)

        matrix = BoolMatrix(num_tids)
        matrix.data[tid_to_idx[pairs[:, 1]], tid_to_idx[pairs[:, 0]]] = True

        return matrix, num_tids, tid_to_idx

    def _membership_oracle(
        self,
        base_set: List[int],
        extension: int,
        base_vtids: np.ndarray,
        ext_vtids: np.ndarray,
    ) -> int:
        pattern = sorted(base_set + [extension])
        if pattern[0] % 2 == 1:
            return 0

        supporting_vtids = base_vtids & ext_vtids
        if not supporting_vtids.any():
            return 0

        matrix, _, _ = self._build_matrix_from_vtids(supporting_vtids)
        if matrix is None:
            return 0

        if matrix.detect_short_cycles():
            raise ValueError("Short cycle detected in boolean matrix!")

        support = compute_gradual_support(matrix)
        return support if support >= self.threshold else 0

    def _closure(self, pattern: List[int], supporting_vtids: np.ndarray) -> List[int]:
        matrix, num_tids, tid_to_idx = self._build_matrix_from_vtids(supporting_vtids)
        if matrix is None:
            return list(pattern)

        closed = list(pattern)
        first_positive_flag = False

        for item_code in range(self.num_attributes * 2):
            if item_code in pattern:
                if item_code % 2 == 0:
                    first_positive_flag = True
                continue

            if item_code % 2 == 1 and not first_positive_flag:
                continue

            common_vtids = supporting_vtids & self.item_to_vtids[item_code]
            if not common_vtids.any():
                continue

            vtid_indices = np.where(common_vtids)[0]
            common_pairs = self.vtid_to_pair[vtid_indices]

            valid = (tid_to_idx[common_pairs[:, 0]] >= 0) & (tid_to_idx[common_pairs[:, 1]] >= 0)
            valid_pairs = common_pairs[valid]

            item_matrix = BoolMatrix(num_tids)
            if len(valid_pairs) > 0:
                item_matrix.data[
                    tid_to_idx[valid_pairs[:, 1]],
                    tid_to_idx[valid_pairs[:, 0]],
                ] = True

            if item_matrix == matrix:
                closed.append(item_code)
                if item_code % 2 == 0:
                    first_positive_flag = True

        return sorted(closed)

    def _mine_recursive(
        self,
        current_pattern: List[int],
        current_vtids: np.ndarray,
        exclusion_list: Set[int],
        depth: int = 0,
    ):
        max_item = self.num_attributes * 2

        for item_code in range(max_item):
            if item_code in exclusion_list or item_code in current_pattern:
                continue

            item_vtids = self.item_to_vtids[item_code]
            support = self._membership_oracle(current_pattern, item_code, current_vtids, item_vtids)

            if support >= self.threshold:
                extended_pattern = sorted(current_pattern + [item_code])
                extended_vtids = current_vtids & item_vtids

                closed_pattern = self._closure(extended_pattern, extended_vtids)

                new_elements = set(closed_pattern) - set(extended_pattern)
                if any(e < item_code for e in new_elements):
                    continue
                if any(e in exclusion_list for e in new_elements):
                    continue

                pattern_key = tuple(closed_pattern)
                if pattern_key in self.found_patterns:
                    continue

                self.found_patterns.add(pattern_key)
                pattern = GradualPattern.from_codes(closed_pattern)
                pattern.support = support
                self.patterns.append(pattern)

                if self.verbose:
                    print(f"{'  ' * depth}Found: {pattern}")

                new_exclusion = exclusion_list | set(closed_pattern)
                self._mine_recursive(closed_pattern, extended_vtids, new_exclusion, depth + 1)

    def mine(self) -> List[GradualPattern]:
        if self._use_full_rust:
            rust_results = self._rust_miner.mine()
            patterns = []
            for r in rust_results:
                p = GradualPattern()
                p.items = [
                    GradualItem(
                        attribute_index=item.attribute_index,
                        variation=Variation.INCREASE if str(item.variation) == "+" else Variation.DECREASE,
                    )
                    for item in r.items
                ]
                p.support = r.support
                patterns.append(p)
            if not self.verbose:
                print(f"\nMining complete! Found {len(patterns)} patterns")
            return patterns
        else:
            return self._mine_python()

    def _mine_python(self) -> List[GradualPattern]:
        if self.transaction_table is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if self.min_support < 1:
            self.threshold = math.ceil(self.min_support * self.num_transactions)
        else:
            self.threshold = int(self.min_support)

        print(f"Mining with min_support={self.min_support} (threshold={self.threshold})")
        sys.setrecursionlimit(max(sys.getrecursionlimit(), self.num_transactions * 2 + 100))

        self._build_from_data_numpy()

        self.patterns = []
        self.found_patterns = set()
        all_vtids = np.ones(self.num_vtrans, dtype=bool)

        self._mine_recursive([], all_vtids, set())

        print(f"\nMining complete! Found {len(self.patterns)} patterns")
        return self.patterns
