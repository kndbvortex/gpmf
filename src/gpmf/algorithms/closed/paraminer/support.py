"""
Support computation for gradual patterns using longest path algorithm.

The support of a gradual pattern is the length of the longest path in the
boolean matrix representing transaction relationships.

# RUST-CANDIDATE: The path computation algorithms are computationally intensive
# and would benefit significantly from Rust implementation.
"""

from typing import List
from .bool_matrix import BoolMatrix


def compute_path_lengths(matrix: BoolMatrix) -> List[int]:
    """
    Compute the longest path length from each node in the matrix.

    Uses recursive DFS with memoization to find the longest path starting
    from each transaction node.

    Args:
        matrix: Boolean matrix representing transaction relationships

    Returns:
        List where path_lengths[i] is the longest path starting from transaction i

    # RUST-CANDIDATE: This recursive algorithm is performance-critical
    """
    size = matrix.size
    path_lengths = [0] * size

    def rec_compute_path_length(trans: int):
        """Recursively compute path length for transaction trans."""
        # Already computed
        if path_lengths[trans] != 0:
            return

        # If this row is null (no outgoing edges), path length is 1
        if matrix.is_null_row(trans):
            path_lengths[trans] = 1
            return

        # Find longest path through any successor
        max_path = 0
        for j in range(size):
            if matrix.get(trans, j):
                # Recursively compute path length for successor
                if path_lengths[j] == 0:
                    rec_compute_path_length(j)

                current_path = path_lengths[j] + 1
                if current_path > max_path:
                    max_path = current_path

        path_lengths[trans] = max_path if max_path > 0 else 1

    # Compute path lengths for all transactions
    for i in range(size):
        rec_compute_path_length(i)

    return path_lengths


def compute_gradual_support(matrix: BoolMatrix) -> int:
    """
    Compute the support of a gradual pattern.

    The support is defined as the length of the longest path in the
    transaction relationship matrix.

    Args:
        matrix: Boolean matrix representing transaction relationships

    Returns:
        The support value (longest path length)

    # RUST-CANDIDATE: Critical for mining performance
    """
    path_lengths = compute_path_lengths(matrix)
    return max(path_lengths) if path_lengths else 0


def support_from_path_lengths(path_lengths: List[int]) -> int:
    """
    Get support value from computed path lengths.

    Args:
        path_lengths: List of path lengths for each transaction

    Returns:
        Maximum path length (the support)
    """
    return max(path_lengths) if path_lengths else 0


def find_longest_path_nodes(matrix: BoolMatrix, path_lengths: List[int]) -> List[int]:
    """
    Find all transaction nodes that are part of any longest path.

    Args:
        matrix: Boolean matrix
        path_lengths: Pre-computed path lengths

    Returns:
        List of transaction IDs that are part of longest paths
    """
    max_length = max(path_lengths) if path_lengths else 0
    if max_length == 0:
        return []

    # Find all nodes that are part of longest paths
    nodes = set()

    def trace_path(trans: int):
        """Recursively trace nodes in longest paths."""
        if trans in nodes:
            return

        if path_lengths[trans] == max_length or path_lengths[trans] == 0:
            nodes.add(trans)

            # Add successors that continue the longest path
            for j in range(matrix.size):
                if matrix.get(trans, j):
                    if path_lengths[j] == path_lengths[trans] - 1:
                        trace_path(j)

    # Start from all nodes with maximum path length
    for i in range(len(path_lengths)):
        if path_lengths[i] == max_length:
            trace_path(i)

    return sorted(list(nodes))
