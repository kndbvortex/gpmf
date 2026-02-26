"""


This implements the core mining logic based on ParaMiner framework.
"""

import math
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from .datatypes import (
    Transaction, TransactionTable, GradualItem, GradualPattern,
    TransactionPair, VirtualTransaction, Variation
)

# Import will use Rust or Python based on availability
# These are set by __init__.py to use either Rust or Python implementations
from . import BoolMatrix, compute_gradual_support, RUST_AVAILABLE, RustGradualMiner


class GradualMiner:


    def __init__(self, min_support: float, num_threads: Optional[int] = None, use_rust: bool = True, verbose: bool = False):
        """
        Initialize the miner.

        Args:
            min_support: Minimum support threshold (0-1 for relative, >1 for absolute)
            num_threads: Number of threads for parallel processing (None = auto)
            use_rust: Use Rust acceleration if available (default: True)
            verbose: Print debug information during mining
        """
        self.min_support = min_support
        self.num_threads = num_threads
        self.verbose = verbose
        self.use_rust = use_rust and RUST_AVAILABLE

        if use_rust and not RUST_AVAILABLE:
            print("Warning: Rust acceleration requested but not available. Using Python implementation.")
            print("To enable Rust: maturin develop --release")
            self.use_rust = False

        if self.use_rust and RustGradualMiner is not None:
            self._rust_miner = RustGradualMiner(
                min_support, num_threads, verbose
            )
            self._use_full_rust = True
        else:
            self._use_full_rust = False
            self.transaction_table: TransactionTable = None
            self.num_attributes = 0
            self.num_transactions = 0
            self.num_vtrans = 0
            self.threshold = 0

            self.item_matrices: List[BoolMatrix] = []

            self.vtransactions: List[VirtualTransaction] = []

            self.vtid_to_pair: List[TransactionPair] = []

            # Transposed representation: item -> list of vtids containing it
            self.item_to_vtids: List[List[int]] = []

            # Found patterns
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

    def _convert_to_gradual_items(self) -> List[VirtualTransaction]:
        """
        Convert original transactions to virtual transactions with gradual items.

        For each pair of transactions (i, j) where i != j:
        - If value[k] in trans_i < value[k] in trans_j, add item 2*k (increase)
        - If value[k] in trans_i > value[k] in trans_j, add item 2*k+1 (decrease)
        - If equal, add the item based on i>j (handles null variations per GLCM definition)

        Returns:
            List of virtual transactions
        """
        vtransactions = []
        vtid = 0

        for i in range(self.num_transactions):
            for j in range(self.num_transactions):
                if i == j:
                    continue

                trans_i = self.transaction_table[i]
                trans_j = self.transaction_table[j]

                items = []
                for k in range(self.num_attributes):
                    val_i = trans_i[k]
                    val_j = trans_j[k]

                    if val_i < val_j:

                        items.append(2 * k)
                    elif val_i > val_j:

                        items.append(2 * k + 1)
                    else:

                        items.append(2 * k + (1 if i > j else 0))

                pair = TransactionPair(tid1=i, tid2=j)
                vtrans = VirtualTransaction(items=items, pair=pair, vtid=vtid)

                vtransactions.append(vtrans)
                self.vtid_to_pair.append(pair)
                vtid += 1

        self.num_vtrans = len(vtransactions)
        return vtransactions

    def _build_item_matrices(self):
        """
        Build boolean matrices for each gradual item.

        For each item, the matrix BM[i,j] = 1 if the virtual transaction
        formed by original transactions (i,j) contains that item.
        """
        num_items = self.num_attributes * 2
        self.item_matrices = [BoolMatrix(self.num_transactions) for _ in range(num_items)]
        self.item_to_vtids = [[] for _ in range(num_items)]

        for vtid, vtrans in enumerate(self.vtransactions):
            for item in vtrans.items:
                self.item_to_vtids[item].append(vtid)

        for item_code in range(num_items):
            matrix = self.item_matrices[item_code]
            for vtid in self.item_to_vtids[item_code]:
                pair = self.vtid_to_pair[vtid]
                # Set matrix[tid2, tid1] = 1 (note the order)
                matrix.set(pair.tid2, pair.tid1, True)

    def _get_transaction_pairs_from_vtids(self, vtids: Set[int]) -> Tuple[List[TransactionPair], int, int]:

        present = set()
        pairs = []

        for vtid in vtids:
            pair = self.vtid_to_pair[vtid]
            pairs.append(pair)
            present.add(pair.tid1)
            present.add(pair.tid2)

        num_tids = len(present)
        max_tid = max(present) if present else 0

        return pairs, num_tids, max_tid

    def _membership_oracle(self, base_set: List[int], extension: int,
                          base_vtids: Set[int], ext_vtids: Set[int]) -> int:

        pattern = sorted(base_set + [extension])

        if pattern[0] % 2 == 1:
            return 0

        supporting_vtids = base_vtids & ext_vtids

        if not supporting_vtids:
            return 0

        pairs, num_tids, max_tid = self._get_transaction_pairs_from_vtids(supporting_vtids)


        tid_to_idx = {}
        idx = 0
        for pair in pairs:
            if pair.tid1 not in tid_to_idx:
                tid_to_idx[pair.tid1] = idx
                idx += 1
            if pair.tid2 not in tid_to_idx:
                tid_to_idx[pair.tid2] = idx
                idx += 1

        matrix = BoolMatrix(num_tids)
        for pair in pairs:
            idx1 = tid_to_idx[pair.tid1]
            idx2 = tid_to_idx[pair.tid2]
            matrix.set(idx2, idx1, True)

        if matrix.detect_short_cycles():
            raise ValueError("Short cycle detected in boolean matrix!")

        # Compute support
        support = compute_gradual_support(matrix)

        return support if support >= self.threshold else 0

    def _closure(self, pattern: List[int], supporting_vtids: Set[int]) -> List[int]:
        """
        Compute the closure of a pattern.

        The closure includes all items that can be added to the pattern
        without changing its support.

        Args:
            pattern: Current pattern (list of item codes)
            supporting_vtids: Virtual transaction IDs supporting the pattern

        Returns:
            Closed pattern (list of item codes)
        """
        closed = list(pattern)

        pairs, num_tids, max_tid = self._get_transaction_pairs_from_vtids(supporting_vtids)

        tid_to_idx = {}
        idx = 0
        for pair in pairs:
            if pair.tid1 not in tid_to_idx:
                tid_to_idx[pair.tid1] = idx
                idx += 1
            if pair.tid2 not in tid_to_idx:
                tid_to_idx[pair.tid2] = idx
                idx += 1

        pattern_matrix = BoolMatrix(num_tids)
        for pair in pairs:
            idx1 = tid_to_idx[pair.tid1]
            idx2 = tid_to_idx[pair.tid2]
            pattern_matrix.set(idx2, idx1, True)

        first_positive_flag = False

        for item_code in range(self.num_attributes * 2):
            if item_code in pattern:
                if item_code % 2 == 0:  # Positive item
                    first_positive_flag = True
                continue

            if item_code % 2 == 1 and not first_positive_flag:
                continue

            item_vtids = set(self.item_to_vtids[item_code])

            common_vtids = item_vtids & supporting_vtids

            item_matrix = BoolMatrix(num_tids)
            for vtid in common_vtids:
                pair = self.vtid_to_pair[vtid]
                if pair.tid1 in tid_to_idx and pair.tid2 in tid_to_idx:
                    idx1 = tid_to_idx[pair.tid1]
                    idx2 = tid_to_idx[pair.tid2]
                    item_matrix.set(idx2, idx1, True)

            if item_matrix == pattern_matrix:
                closed.append(item_code)
                if item_code % 2 == 0:
                    first_positive_flag = True

        return sorted(closed)

    def _mine_recursive(self, current_pattern: List[int], current_vtids: Set[int],
                       exclusion_list: Set[int], depth: int = 0):
        """
        Recursive pattern mining using DFS.

        Args:
            current_pattern: Current pattern being extended
            current_vtids: Virtual transactions supporting current pattern
            exclusion_list: Items that should not be considered for extension
            depth: Current recursion depth
        """
        max_item = self.num_attributes * 2

        for item_code in range(max_item):
            if item_code in exclusion_list or item_code in current_pattern:
                continue

            item_vtids = set(self.item_to_vtids[item_code])

            support = self._membership_oracle(
                current_pattern, item_code,
                current_vtids, item_vtids
            )


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
                        variation=Variation.INCREASE if str(item.variation) == "+"
                        else Variation.DECREASE
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


        self.vtransactions = self._convert_to_gradual_items()

        self._build_item_matrices()

        self.patterns = []
        self.found_patterns = set()
        empty_pattern = []
        all_vtids = set(range(self.num_vtrans))
        empty_exclusion = set()

        self._mine_recursive(empty_pattern, all_vtids, empty_exclusion)

        print(f"\nMining complete! Found {len(self.patterns)} patterns")
        return self.patterns
