"""Result container for gradual pattern mining."""
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

from .data_structures import GradualPattern, TemporalGradualPattern


class MiningResult:
    """Container for gradual pattern mining results.

    Stores patterns and metadata from mining operations.

    Attributes:
        patterns: List of discovered patterns
        algorithm: Name of algorithm used
        min_support: Minimum support threshold
        execution_time: Time taken to mine patterns (seconds)
        metadata: Additional algorithm-specific metadata

    Example:
        >>> result = MiningResult(patterns, 'graank', 0.5, execution_time=2.5)
        >>> print(result.to_json())
        >>> df = result.to_dataframe()
    """

    def __init__(
        self,
        patterns: List[GradualPattern],
        algorithm: str,
        min_support: float,
        execution_time: Optional[float] = None,
        **metadata
    ):
        """Initialize mining result.

        Args:
            patterns: List of mined patterns
            algorithm: Algorithm name
            min_support: Minimum support threshold used
            execution_time: Execution time in seconds
            **metadata: Additional metadata (dataset info, parameters, etc.)
        """
        self.patterns = patterns
        self.algorithm = algorithm
        self.min_support = min_support
        self.execution_time = execution_time
        self.metadata = metadata
        self.timestamp = datetime.now()

    def __len__(self) -> int:
        """Return number of patterns."""
        return len(self.patterns)

    def __iter__(self):
        """Iterate over patterns."""
        return iter(self.patterns)

    def __getitem__(self, index: int) -> GradualPattern:
        """Get pattern by index."""
        return self.patterns[index]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary.

        Returns:
            Dictionary with all result information
        """
        return {
            'algorithm': self.algorithm,
            'min_support': self.min_support,
            'num_patterns': len(self.patterns),
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat(),
            'patterns': [p.to_dict() for p in self.patterns],
            'metadata': self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert result to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, filepath: str):
        """Save result to JSON file.

        Args:
            filepath: Path to output file
        """
        with open(filepath, 'w') as f:
            f.write(self.to_json())

    def to_dataframe(self) -> pd.DataFrame:
        """Convert result to pandas DataFrame.

        Returns:
            DataFrame with pattern and support columns
        """
        data = []
        for pattern in self.patterns:
            row = {
                'pattern': str(pattern.to_string()),
                'support': pattern.support,
                'length': len(pattern)
            }

            if isinstance(pattern, TemporalGradualPattern) and pattern.time_lags:
                row['time_lags'] = str([tl.to_string() for tl in pattern.time_lags])

            data.append(row)

        return pd.DataFrame(data)

    def save_csv(self, filepath: str):
        """Save result to CSV file.

        Args:
            filepath: Path to output file
        """
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

    def filter_by_support(self, min_support: float) -> 'MiningResult':
        """Filter patterns by support threshold.

        Args:
            min_support: Minimum support threshold

        Returns:
            New MiningResult with filtered patterns
        """
        filtered = [p for p in self.patterns if p.support >= min_support]
        return MiningResult(
            patterns=filtered,
            algorithm=self.algorithm,
            min_support=min_support,
            execution_time=self.execution_time,
            **self.metadata
        )

    def filter_by_length(self, min_length: int = None, max_length: int = None) -> 'MiningResult':
        """Filter patterns by length.

        Args:
            min_length: Minimum pattern length
            max_length: Maximum pattern length

        Returns:
            New MiningResult with filtered patterns
        """
        filtered = self.patterns
        if min_length is not None:
            filtered = [p for p in filtered if len(p) >= min_length]
        if max_length is not None:
            filtered = [p for p in filtered if len(p) <= max_length]

        return MiningResult(
            patterns=filtered,
            algorithm=self.algorithm,
            min_support=self.min_support,
            execution_time=self.execution_time,
            **self.metadata
        )

    def sort_by_support(self, reverse: bool = True) -> 'MiningResult':
        """Sort patterns by support.

        Args:
            reverse: Sort in descending order (default True)

        Returns:
            New MiningResult with sorted patterns
        """
        sorted_patterns = sorted(self.patterns, key=lambda p: p.support, reverse=reverse)
        return MiningResult(
            patterns=sorted_patterns,
            algorithm=self.algorithm,
            min_support=self.min_support,
            execution_time=self.execution_time,
            **self.metadata
        )

    def summary(self) -> str:
        """Get a summary of the mining results.

        Returns:
            Summary string
        """
        lines = [
            f"Algorithm: {self.algorithm}",
            f"Minimum Support: {self.min_support}",
            f"Patterns Found: {len(self.patterns)}",
        ]

        if self.execution_time is not None:
            lines.append(f"Execution Time: {self.execution_time:.3f}s")

        if self.patterns:
            supports = [p.support for p in self.patterns]
            lines.append(f"Support Range: [{min(supports):.3f}, {max(supports):.3f}]")

            lengths = [len(p) for p in self.patterns]
            lines.append(f"Pattern Length Range: [{min(lengths)}, {max(lengths)}]")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation of result."""
        return self.summary()

    def __repr__(self) -> str:
        """Detailed representation of result."""
        return f"MiningResult(algorithm='{self.algorithm}', patterns={len(self.patterns)}, support={self.min_support})"
