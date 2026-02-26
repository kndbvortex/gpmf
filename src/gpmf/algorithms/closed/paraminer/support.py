import sys
import numpy as np
from typing import List
from .bool_matrix import BoolMatrix


def compute_path_lengths(matrix: BoolMatrix) -> List[int]:
    size = matrix.size
    path_lengths = [0] * size

    required = size + 50
    if sys.getrecursionlimit() < required:
        sys.setrecursionlimit(required)

    def rec_compute_path_length(trans: int):
        if path_lengths[trans] != 0:
            return
        successors = np.where(matrix.data[trans])[0]
        if len(successors) == 0:
            path_lengths[trans] = 1
            return
        for j in successors:
            if path_lengths[j] == 0:
                rec_compute_path_length(int(j))
        path_lengths[trans] = max(int(path_lengths[j]) + 1 for j in successors)

    for i in range(size):
        rec_compute_path_length(i)

    return path_lengths


def compute_gradual_support(matrix: BoolMatrix) -> int:
    path_lengths = compute_path_lengths(matrix)
    return max(path_lengths) if path_lengths else 0


def support_from_path_lengths(path_lengths: List[int]) -> int:
    return max(path_lengths) if path_lengths else 0


def find_longest_path_nodes(matrix: BoolMatrix, path_lengths: List[int]) -> List[int]:
    max_length = max(path_lengths) if path_lengths else 0
    if max_length == 0:
        return []

    nodes: set = set()

    def trace_path(trans: int):
        if trans in nodes:
            return
        if path_lengths[trans] == max_length or path_lengths[trans] == 0:
            nodes.add(trans)
            for j in np.where(matrix.data[trans])[0]:
                if path_lengths[j] == path_lengths[trans] - 1:
                    trace_path(int(j))

    for i in range(len(path_lengths)):
        if path_lengths[i] == max_length:
            trace_path(i)

    return sorted(nodes)
