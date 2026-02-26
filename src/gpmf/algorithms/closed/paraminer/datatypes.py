"""
Data structures for gradual pattern mining.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set, Tuple


class Variation(Enum):
    """Direction of variation for a gradual item."""
    INCREASE = "+"  # Positive variation
    DECREASE = "-"  # Negative variation


@dataclass
class GradualItem:
    """
    A gradual item represents an attribute with a direction of variation.

    For example, "Age+" means increasing age, "Salary-" means decreasing salary.
    Items are encoded as: attribute_index * 2 + (0 for +, 1 for -)
    """
    attribute_index: int
    variation: Variation

    def to_code(self) -> int:
        """Convert to integer encoding: 2*k for +, 2*k+1 for -"""
        return self.attribute_index * 2 + (0 if self.variation == Variation.INCREASE else 1)

    @classmethod
    def from_code(cls, code: int) -> "GradualItem":
        """Create from integer encoding."""
        return cls(
            attribute_index=code // 2,
            variation=Variation.INCREASE if code % 2 == 0 else Variation.DECREASE
        )

    def __str__(self) -> str:
        return f"{self.attribute_index + 1}{self.variation.value}"

    def __repr__(self) -> str:
        return f"GradualItem({self.attribute_index + 1}{self.variation.value})"

    def __hash__(self) -> int:
        return hash((self.attribute_index, self.variation))

    def __eq__(self, other) -> bool:
        if not isinstance(other, GradualItem):
            return False
        return self.attribute_index == other.attribute_index and self.variation == other.variation


@dataclass
class Transaction:
    """
    A transaction represents numerical values for multiple attributes.
    """
    values: List[float]
    tid: int  # Transaction ID
    weight: int = 1

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> float:
        return self.values[index]


@dataclass
class TransactionPair:
    """
    Represents a pair of transactions (tid1, tid2).
    Used for representing gradual relationships between records.
    """
    tid1: int
    tid2: int

    def __hash__(self) -> int:
        # Combine the two IDs into a single hash
        # Using bit shifting to pack two 16-bit values into 32 bits
        return (self.tid1 << 16) | self.tid2

    def __eq__(self, other) -> bool:
        if not isinstance(other, TransactionPair):
            return False
        return self.tid1 == other.tid1 and self.tid2 == other.tid2


@dataclass
class VirtualTransaction:
    """
    A virtual transaction created from a pair of original transactions.
    Contains the gradual items that are satisfied by the pair.
    """
    items: List[int]  # List of gradual item codes
    pair: TransactionPair
    vtid: int  # Virtual transaction ID
    weight: int = 1

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class TransactionTable:
    """
    A table of transactions (dataset).
    """
    transactions: List[Transaction]
    num_attributes: int = 0

    def __post_init__(self):
        if self.transactions and self.num_attributes == 0:
            self.num_attributes = len(self.transactions[0].values)

    def __len__(self) -> int:
        return len(self.transactions)

    def __getitem__(self, index: int) -> Transaction:
        return self.transactions[index]

    @classmethod
    def from_file(cls, filename: str) -> "TransactionTable":
        """
        Read a transaction table from a file.

        File format:
        - First line is skipped (header)
        - Each subsequent line contains space-separated numerical values
        """
        transactions = []
        num_attributes = 0

        with open(filename, 'r') as f:
            # Skip header
            next(f)

            tid = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue

                values = [float(x) for x in line.split()]
                if values:
                    if num_attributes == 0:
                        num_attributes = len(values)

                    # Multiply by 1000 to handle float inputs (similar to C++ version)
                    values = [v * 1000 for v in values]
                    transactions.append(Transaction(values=values, tid=tid))
                    tid += 1

        return cls(transactions=transactions, num_attributes=num_attributes)


@dataclass
class GradualPattern:
    """
    A gradual pattern is a set of gradual items.

    For example: {Age+, Salary+, Experience-}
    """
    items: List[GradualItem] = field(default_factory=list)
    support: int = 0
    supporting_pairs: List[TransactionPair] = field(default_factory=list)

    def add_item(self, item: GradualItem):
        """Add a gradual item to the pattern."""
        self.items.append(item)

    def to_codes(self) -> List[int]:
        """Convert items to integer codes."""
        return sorted([item.to_code() for item in self.items])

    @classmethod
    def from_codes(cls, codes: List[int]) -> "GradualPattern":
        """Create pattern from integer codes."""
        items = [GradualItem.from_code(code) for code in codes]
        return cls(items=items)

    def is_valid(self) -> bool:
        """
        Check if pattern is valid.

        Rules:
        - First item must be an increase (X+)
        - Cannot have both X+ and X- for same attribute
        """
        if not self.items:
            return True

        # First item must be an increase
        if self.items[0].variation == Variation.DECREASE:
            return False

        # Check for conflicting items
        seen_attrs = {}
        for item in self.items:
            if item.attribute_index in seen_attrs:
                # Same attribute appears twice - not allowed
                return False
            seen_attrs[item.attribute_index] = item.variation

        return True

    def __str__(self) -> str:
        item_strs = [str(item) for item in sorted(self.items, key=lambda x: x.attribute_index)]
        return f"{{{', '.join(item_strs)}}} (sup={self.support})"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return len(self.items)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.to_codes())))

    def __eq__(self, other) -> bool:
        if not isinstance(other, GradualPattern):
            return False
        return self.to_codes() == other.to_codes()
