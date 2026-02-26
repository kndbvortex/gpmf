"""
Boolean matrix implementation for gradual pattern mining.

# RUST-CANDIDATE: This module is a prime candidate for Rust implementation
# for better performance. The matrix operations (especially bitwise AND)
# are heavily used in the mining algorithm.
"""

from typing import List


class BoolMatrix:
    """
    Square boolean matrix for representing transaction pair relationships.

    In gradual mining, BoolMatrix[i, j] = 1 means transaction i < transaction j
    according to the pattern (all attributes vary in the expected direction).
    """

    def __init__(self, size: int, fill: bool = False):
        """
        Create a boolean matrix of size x size.

        Args:
            size: Number of rows (and columns) in the square matrix
            fill: Initial value for all cells
        """
        self.size = size
        self.data = [fill] * (size * size)

    def get(self, row: int, col: int) -> bool:
        """Get value at [row, col]."""
        assert 0 <= row < self.size and 0 <= col < self.size
        return self.data[row * self.size + col]

    def set(self, row: int, col: int, value: bool):
        """Set value at [row, col]."""
        assert 0 <= row < self.size and 0 <= col < self.size
        self.data[row * self.size + col] = value

    def is_null_row(self, row: int) -> bool:
        """Check if row has no True values."""
        assert 0 <= row < self.size
        start = row * self.size
        end = (row + 1) * self.size
        return not any(self.data[start:end])

    def bitwise_and(self, other: "BoolMatrix") -> "BoolMatrix":
        """
        Compute bitwise AND with another matrix.

        Returns a new matrix where result[i,j] = self[i,j] AND other[i,j]

        # RUST-CANDIDATE: This operation is performance-critical
        """
        assert self.size == other.size, "Matrices must be same size"
        result = BoolMatrix(self.size)
        for i in range(len(self.data)):
            result.data[i] = self.data[i] and other.data[i]
        return result

    def bitwise_and_inplace(self, other: "BoolMatrix"):
        """
        Compute bitwise AND in-place.

        # RUST-CANDIDATE: This operation is performance-critical
        """
        assert self.size == other.size, "Matrices must be same size"
        for i in range(len(self.data)):
            self.data[i] = self.data[i] and other.data[i]

    def __eq__(self, other: object) -> bool:
        """Check if two matrices are equal."""
        if not isinstance(other, BoolMatrix):
            return False
        return self.size == other.size and self.data == other.data

    def copy(self) -> "BoolMatrix":
        """Create a deep copy of the matrix."""
        result = BoolMatrix(self.size)
        result.data = self.data.copy()
        return result

    def __repr__(self) -> str:
        return f"BoolMatrix(size={self.size})"

    def to_string(self) -> str:
        """Convert matrix to readable string format."""
        lines = []
        for row in range(self.size):
            row_vals = []
            for col in range(self.size):
                row_vals.append('1' if self.get(row, col) else '0')
            lines.append(' '.join(row_vals))
        return '\n'.join(lines)

    @classmethod
    def from_transaction_pairs(cls, pairs: List[tuple], size: int) -> "BoolMatrix":
        """
        Create a boolean matrix from a list of transaction pairs.

        Args:
            pairs: List of (tid1, tid2) tuples where tid1 < tid2 according to pattern
            size: Size of the matrix (number of transactions)

        Returns:
            BoolMatrix where matrix[tid2, tid1] = 1 for each pair
        """
        matrix = cls(size)
        for tid1, tid2 in pairs:
            # Note: we set [tid2, tid1] not [tid1, tid2]
            # This represents that tid2 comes after tid1 in the ordering
            matrix.set(tid2, tid1, True)
        return matrix

    def detect_short_cycles(self) -> bool:
        """
        Detect if there are any short cycles (i.e., matrix[i,j] and matrix[j,i] both true).

        Returns True if a cycle is detected.

        Short cycles should not exist in valid gradual patterns since they
        would imply both i < j and j < i.
        """
        for i in range(self.size):
            for j in range(i + 1, self.size):
                if self.get(i, j) and self.get(j, i):
                    return True
        return False
