"""MSGP - Mining Seasonal Gradual Patterns.

Reference: Mining Frequent Seasonal Gradual Patterns
"""
import numpy as np
import logging
import os
import tempfile
from typing import List, Optional
from pathlib import Path

from ..base_algorithm import BaseAlgorithm
from ...core.data_structures import GradualItem, GradualPattern
from ...config import config

logger = logging.getLogger(__name__)


class MSGP(BaseAlgorithm):
    """MSGP algorithm for mining seasonal gradual patterns.

    Uses SPMF's MPFPS-BFS algorithm for periodic pattern mining combined
    with gradual pattern extraction.

    Args:
        min_support: Minimum support threshold
        cycle_size: Size of one cycle/season (default: 8)
        min_ra: Minimum ratio (default: auto-calculated)
        max_periodicity: Maximum periodicity (default: dataset size)
        max_std: Maximum standard deviation (default: dataset size)
        spmf_path: Path to SPMF jar file

    Example:
        >>> msgp = MSGP(min_support=0.5, cycle_size=8)
        >>> patterns = msgp.mine('data.csv')
    """

    def __init__(self, min_support: float = 0.5, cycle_size: int = 8,
                 min_ra: Optional[float] = None, max_periodicity: Optional[int] = None,
                 max_std: Optional[int] = None, spmf_path: Optional[str] = None, **kwargs):
        super().__init__(min_support, **kwargs)
        self.cycle_size = cycle_size
        self.min_ra = min_ra
        self.max_periodicity = max_periodicity
        self.max_std = max_std
        self.spmf_path = spmf_path or config.spmf_path
        self._params.update({
            'cycle_size': cycle_size,
            'min_ra': min_ra,
            'max_periodicity': max_periodicity,
            'max_std': max_std
        })

    def _numerical_to_sequential(self, output_file: str):
        """Convert numerical database to sequential database."""
        data = self.dataset.data
        n_rows, n_cols = data.shape

        sequences = []
        for cycle_id in range(self.cycle_size):
            sequence = []
            start_idx = cycle_id
            while start_idx < n_rows:
                itemset = []
                for col_idx in self.dataset.attr_cols:
                    if start_idx > 0:
                        if data[start_idx][col_idx] > data[start_idx - 1][col_idx]:
                            itemset.append(f"{col_idx * 2 + 2}")
                        elif data[start_idx][col_idx] < data[start_idx - 1][col_idx]:
                            itemset.append(f"{col_idx * 2 + 1}")
                if itemset:
                    sequence.append(itemset)
                start_idx += self.cycle_size

            if sequence:
                sequences.append(sequence)

        with open(output_file, 'w') as f:
            for seq in sequences:
                for itemset in seq:
                    f.write(' '.join(itemset) + ' -1 ')
                f.write('-2\n')

        logger.debug(f"Created {len(sequences)} sequences")

    def _extract_periodic_patterns(self, input_file: str, output_file: str):
        """Extract periodic patterns using SPMF."""
        min_ra = self.min_ra if self.min_ra is not None else (1 / self.cycle_size)
        max_per = self.max_periodicity if self.max_periodicity is not None else self.dataset.row_count
        max_std = self.max_std if self.max_std is not None else self.dataset.row_count
        min_sup_count = int(self.min_support * self.cycle_size) if self.min_support < 1 else 2

        try:
            from spmf import Spmf

            spmf = Spmf(
                "MPFPS-BFS",
                input_filename=input_file,
                output_filename=output_file,
                arguments=[max_std, min_ra, max_per, min_sup_count]
            )
            spmf.run()
            logger.debug("SPMF periodic pattern mining completed")
            return True

        except ImportError:
            logger.error("SPMF library not available. Install with: pip install spmf-py")
            return False
        except Exception as e:
            logger.error(f"SPMF execution failed: {e}")
            return False

    def _extract_seasonal_patterns(self, input_file: str, output_file: str):
        """Extract seasonal gradual patterns from periodic patterns."""
        patterns_data = []

        try:
            with open(input_file, 'r') as f:
                for line in f:
                    parts = line.split('#')
                    if len(parts) < 3:
                        continue

                    pattern_items = parts[0].strip()
                    if len(pattern_items) < 6:
                        continue

                    gradual_items = []
                    support_part = parts[2]

                    i = 0
                    while i < len(support_part):
                        if support_part[i] == '[':
                            num_str = ''
                            j = i - 1
                            while j >= 0 and support_part[j].strip() and support_part[j] not in ['[', ']', ' ']:
                                num_str = support_part[j] + num_str
                                j -= 1

                            if num_str.isdigit():
                                item_num = int(num_str)
                                if item_num % 2 == 0:
                                    attr_idx = (item_num // 2) - 1
                                    gradual_items.append(f"{attr_idx}+")
                                else:
                                    attr_idx = (item_num // 2)
                                    gradual_items.append(f"{attr_idx}-")
                        i += 1

                    if gradual_items:
                        patterns_data.append({
                            'items': gradual_items,
                            'support': pattern_items
                        })

        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file}")
            return []

        with open(output_file, 'w') as f:
            for p in patterns_data:
                f.write(' '.join(p['items']) + ' : ' + p['support'] + '\n')

        return patterns_data

    def _remove_redundancy(self, patterns_data: List[dict]) -> List[GradualPattern]:
        """Remove redundant patterns and convert to GradualPattern objects."""
        unique_patterns = {}

        for p_data in patterns_data:
            items_key = tuple(sorted(p_data['items']))
            if items_key not in unique_patterns:
                unique_patterns[items_key] = p_data

        result_patterns = []
        for items_key, p_data in unique_patterns.items():
            gp = GradualPattern()

            for item_str in p_data['items']:
                attr_idx = int(item_str[:-1])
                symbol = item_str[-1]
                gi = GradualItem(attr_idx, symbol)
                gp.add_gradual_item(gi)

            try:
                support_val = float(p_data['support']) / self.cycle_size
            except ValueError:
                support_val = self.min_support

            gp.set_support(support_val)
            result_patterns.append(gp)

        return result_patterns

    def _mine(self) -> List[GradualPattern]:
        """Mine seasonal gradual patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_file = os.path.join(tmpdir, 'sequences.txt')
            periodic_file = os.path.join(tmpdir, 'periodic.txt')
            seasonal_file = os.path.join(tmpdir, 'seasonal.txt')

            self._numerical_to_sequential(seq_file)

            if not self._extract_periodic_patterns(seq_file, periodic_file):
                logger.warning("Periodic pattern extraction failed")
                return []

            if not os.path.exists(periodic_file) or os.path.getsize(periodic_file) == 0:
                logger.warning("No periodic patterns found")
                return []

            patterns_data = self._extract_seasonal_patterns(periodic_file, seasonal_file)

            if not patterns_data:
                logger.warning("No seasonal patterns extracted")
                return []

            result_patterns = self._remove_redundancy(patterns_data)

        logger.info(f"MSGP found {len(result_patterns)} seasonal patterns")
        return result_patterns
