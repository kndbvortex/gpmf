"""Core data structures for gradual pattern mining.

This module provides the fundamental data structures used across all
gradual pattern mining algorithms.
"""
import numpy as np
from typing import List, Union, Tuple, Optional
from dataclasses import dataclass, field


class GradualItem:
    """Represents a gradual item (attribute, variation).

    A gradual item is a pair (i, v) where i is a column index and v is
    a variation symbol (+ for increasing, - for decreasing).

    Example:
        >>> gi = GradualItem(0, '+')
        >>> print(gi.to_string())
        0+

    Attributes:
        attribute_col: Column index in the dataset
        symbol: Variation symbol ('+' or '-')
    """

    def __init__(self, attr_col: int, symbol: str):
        """Initialize a gradual item.

        Args:
            attr_col: Column index
            symbol: Variation symbol ('+' or '-')

        Raises:
            ValueError: If symbol is not '+' or '-'
        """
        if symbol not in ['+', '-']:
            raise ValueError(f"Symbol must be '+' or '-', got '{symbol}'")

        self.attribute_col = attr_col
        self.symbol = symbol
        self.rank_sum = 0  # For compatibility with existing algorithms

    @property
    def gradual_item(self) -> np.ndarray:
        """Return gradual item as numpy array."""
        return np.array((self.attribute_col, self.symbol), dtype='i, S1')

    @property
    def tuple(self) -> Tuple[int, str]:
        """Return gradual item as tuple."""
        return (self.attribute_col, self.symbol)

    def inv(self) -> np.ndarray:
        """Return inverted gradual item as numpy array."""
        inv_symbol = '-' if self.symbol == '+' else '+'
        return np.array((self.attribute_col, inv_symbol), dtype='i, S1')

    def inv_gi(self) -> 'GradualItem':
        """Return inverted gradual item object."""
        inv_symbol = '-' if self.symbol == '+' else '+'
        return GradualItem(self.attribute_col, inv_symbol)

    def as_integer(self) -> int:
        """Convert symbol to integer (+ to 1, - to -1)."""
        return 1 if self.symbol == '+' else -1

    def as_string(self) -> str:
        """Return string representation (e.g., '0+')."""
        return f"{self.attribute_col}{self.symbol}"

    def to_string(self) -> str:
        """Return string representation (alias for as_string)."""
        return self.as_string()

    def __str__(self) -> str:
        return self.as_string()

    def __repr__(self) -> str:
        return f"GradualItem({self.attribute_col}, '{self.symbol}')"

    def __eq__(self, other) -> bool:
        if not isinstance(other, GradualItem):
            return False
        return self.attribute_col == other.attribute_col and self.symbol == other.symbol

    def __hash__(self) -> int:
        return hash((self.attribute_col, self.symbol))


class GradualPattern:
    """Represents a gradual pattern (set of gradual items).

    A gradual pattern is a set of gradual items with a support value
    measuring its quality.

    Example:
        >>> gp = GradualPattern()
        >>> gp.add_gradual_item(GradualItem(0, '+'))
        >>> gp.add_gradual_item(GradualItem(1, '-'))
        >>> gp.set_support(0.75)
        >>> print(f"{gp.to_string()} : {gp.support}")
        ['0+', '1-'] : 0.75

    Attributes:
        gradual_items: List of GradualItem objects
        support: Support value (0.0 to 1.0)
    """

    def __init__(self):
        """Initialize an empty gradual pattern."""
        self.gradual_items: List[GradualItem] = []
        self.support: float = 0.0
        self.path: List[int] = []  # For GRITE: longest path in precedence graph

    def set_support(self, support: float):
        """Set the support value.

        Args:
            support: Support value (0.0 to 1.0)
        """
        self.support = round(support, 3)

    def add_gradual_item(self, item: GradualItem):
        """Add a gradual item to the pattern.

        Args:
            item: GradualItem to add
        """
        if item.symbol in ['+', '-']:
            self.gradual_items.append(item)

    def add_items_from_list(self, lst_items: List[str]):
        """Add gradual items from a list of strings.

        Args:
            lst_items: List of strings like ['0+', '1-', '2+']
        """
        for str_gi in lst_items:
            if len(str_gi) >= 2:
                attr_col = int(str_gi[:-1])
                symbol = str_gi[-1]
                self.add_gradual_item(GradualItem(attr_col, symbol))

    def get_pattern(self) -> List[np.ndarray]:
        """Get pattern as list of numpy arrays."""
        return [item.gradual_item for item in self.gradual_items]

    def get_tuples(self) -> List[Tuple[int, str]]:
        """Get pattern as list of tuples."""
        return [item.tuple for item in self.gradual_items]

    def get_np_array(self) -> np.ndarray:
        """Get pattern as numpy array."""
        return np.array(self.get_pattern())

    def to_string(self) -> List[str]:
        """Get pattern as list of strings."""
        return [item.as_string() for item in self.gradual_items]

    def to_string_list(self) -> List[str]:
        """Alias for to_string()."""
        return self.to_string()

    def to_dict(self) -> dict:
        """Convert pattern to dictionary."""
        return {
            'pattern': self.to_string(),
            'support': self.support
        }

    def __len__(self) -> int:
        return len(self.gradual_items)

    def __str__(self) -> str:
        return f"{self.to_string()} : {self.support}"

    def __repr__(self) -> str:
        return f"GradualPattern(items={self.to_string()}, support={self.support})"


@dataclass
class TimeLag:
    """Represents a time lag with support.

    Used for temporal gradual patterns.

    Attributes:
        timestamp: Time lag value
        support: Support of the time lag
        sign: Sign of the time lag (+/-)
        time_format: Format for displaying the timestamp
    """
    timestamp: float
    support: float = 0.0
    sign: str = "~"
    time_format: str = "%Y-%m-%d"

    def to_string(self) -> str:
        """Return string representation of time lag."""
        if self.sign == "+":
            return f"+{self.timestamp}"
        elif self.sign == "-":
            return f"-{self.timestamp}"
        else:
            return f"~{self.timestamp}"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'support': self.support,
            'sign': self.sign
        }

    def __str__(self) -> str:
        return self.to_string()


class TemporalGradualPattern(GradualPattern):
    """Represents a temporal gradual pattern with time lags.

    Extends GradualPattern with temporal information.

    Attributes:
        time_lags: List of TimeLag objects
    """

    def __init__(self):
        """Initialize an empty temporal gradual pattern."""
        super().__init__()
        self.time_lags: List[TimeLag] = []

    def add_time_lag(self, time_lag: TimeLag):
        """Add a time lag to the pattern.

        Args:
            time_lag: TimeLag object to add
        """
        self.time_lags.append(time_lag)

    def to_dict(self) -> dict:
        """Convert to dictionary including time lags."""
        result = super().to_dict()
        result['time_lags'] = [tl.to_dict() for tl in self.time_lags]
        return result

    def __str__(self) -> str:
        base_str = super().__str__()
        if self.time_lags:
            time_str = ", ".join(tl.to_string() for tl in self.time_lags)
            return f"{base_str} | Time lags: [{time_str}]"
        return base_str
