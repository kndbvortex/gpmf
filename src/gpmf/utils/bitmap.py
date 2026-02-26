"""Bitmap operations for gradual pattern mining."""
import numpy as np
from typing import Tuple


class BitmapHelper:
    """Helper class for bitmap operations.

    Provides common operations on binary matrices used in
    gradual pattern mining algorithms.
    """

    @staticmethod
    def compute_support(bitmap: np.ndarray) -> float:
        """Compute support of a bitmap.

        Args:
            bitmap: Binary matrix

        Returns:
            Support value (0.0 to 1.0)
        """
        n = bitmap.shape[0]
        if n <= 1:
            return 0.0

        num_ones = np.sum(bitmap)
        max_possible = n * (n - 1.0) / 2.0

        if max_possible == 0:
            return 0.0

        return float(num_ones) / max_possible

    @staticmethod
    def intersect_bitmaps(bitmap1: np.ndarray, bitmap2: np.ndarray) -> np.ndarray:
        """Compute intersection of two bitmaps.

        Args:
            bitmap1: First binary matrix
            bitmap2: Second binary matrix

        Returns:
            Intersection bitmap
        """
        return np.logical_and(bitmap1, bitmap2).astype(int)

    @staticmethod
    def union_bitmaps(bitmap1: np.ndarray, bitmap2: np.ndarray) -> np.ndarray:
        """Compute union of two bitmaps.

        Args:
            bitmap1: First binary matrix
            bitmap2: Second binary matrix

        Returns:
            Union bitmap
        """
        return np.logical_or(bitmap1, bitmap2).astype(int)

    @staticmethod
    def is_valid_bitmap(bitmap: np.ndarray, min_support: float) -> bool:
        """Check if bitmap meets minimum support threshold.

        Args:
            bitmap: Binary matrix
            min_support: Minimum support threshold

        Returns:
            True if valid, False otherwise
        """
        support = BitmapHelper.compute_support(bitmap)
        return support >= min_support

    @staticmethod
    def extract_gradual_item_from_bin(bin_obj: np.ndarray) -> Tuple[int, str]:
        """Extract gradual item from bitmap object.

        Args:
            bin_obj: Bitmap object (format: [gi_info, bitmap])

        Returns:
            Tuple of (attribute_col, symbol)
        """
        gi_info = bin_obj[0]
        attr_col = int(gi_info[0])
        symbol = gi_info[1].decode() if hasattr(gi_info[1], 'decode') else gi_info[1]
        return attr_col, symbol

    @staticmethod
    def get_bitmap_from_bin(bin_obj: np.ndarray) -> np.ndarray:
        """Extract bitmap matrix from bitmap object.

        Args:
            bin_obj: Bitmap object (format: [gi_info, bitmap])

        Returns:
            Bitmap matrix
        """
        return bin_obj[1]
