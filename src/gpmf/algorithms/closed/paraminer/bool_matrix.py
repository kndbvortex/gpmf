import numpy as np
from typing import List


class BoolMatrix:

    def __init__(self, size: int, fill: bool = False):
        self.size = size
        self.data = np.full((size, size), fill, dtype=bool)

    def get(self, row: int, col: int) -> bool:
        assert 0 <= row < self.size and 0 <= col < self.size
        return bool(self.data[row, col])

    def set(self, row: int, col: int, value: bool):
        assert 0 <= row < self.size and 0 <= col < self.size
        self.data[row, col] = value

    def is_null_row(self, row: int) -> bool:
        assert 0 <= row < self.size
        return not self.data[row].any()

    def bitwise_and(self, other: "BoolMatrix") -> "BoolMatrix":
        assert self.size == other.size, "Matrices must be same size"
        result = BoolMatrix(self.size)
        result.data = self.data & other.data
        return result

    def bitwise_and_inplace(self, other: "BoolMatrix"):
        assert self.size == other.size, "Matrices must be same size"
        self.data &= other.data

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoolMatrix):
            return False
        return self.size == other.size and np.array_equal(self.data, other.data)

    def copy(self) -> "BoolMatrix":
        result = BoolMatrix(self.size)
        result.data = self.data.copy()
        return result

    def __repr__(self) -> str:
        return f"BoolMatrix(size={self.size})"

    def to_string(self) -> str:
        lines = []
        for row in range(self.size):
            lines.append(" ".join("1" if v else "0" for v in self.data[row]))
        return "\n".join(lines)

    @classmethod
    def from_transaction_pairs(cls, pairs: List[tuple], size: int) -> "BoolMatrix":
        matrix = cls(size)
        for tid1, tid2 in pairs:
            matrix.data[tid2, tid1] = True
        return matrix

    def detect_short_cycles(self) -> bool:
        return bool(np.any(np.triu(self.data, 1) & np.triu(self.data.T, 1)))
