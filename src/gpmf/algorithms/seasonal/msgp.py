"""MSGP - Mining Seasonal Gradual Patterns.

Reference: Mining Frequent Seasonal Gradual Patterns
"""
import re

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

    def _n_cycles(self) -> int:
        return self.dataset.row_count // self.cycle_size

    def _item_code(self, col_idx: int, symbol: str) -> int:
        return col_idx * 2 + (2 if symbol == '+' else 1)

    @staticmethod
    def _decode(code: int) -> str:
        return f"{code // 2 - 1}+" if code % 2 == 0 else f"{code // 2}-"

    def _numerical_to_sequential(self, output_file: str):
        """Sequence database of Lonlac et al.: the series is cut into consecutive cycles of
        `cycle_size` observations (an incomplete last cycle is dropped); each cycle is one
        sequence, whose k-th itemset holds the gradual items respected between observations k and
        k+1 of the cycle. Transitions across two cycles are not used."""
        data = self.dataset.data
        attrs = list(self.dataset.attr_cols)
        vals = np.asarray(data[:, attrs], dtype=float)
        L = self.cycle_size
        lines = []
        for c in range(self._n_cycles()):
            block = vals[c * L:(c + 1) * L]
            diff = np.diff(block, axis=0)
            seq = []
            for row in diff:
                items = [self._item_code(attrs[j], '+') for j in np.flatnonzero(row > 0)]
                items += [self._item_code(attrs[j], '-') for j in np.flatnonzero(row < 0)]
                # SPMF needs a non-empty itemset at every position to keep positions aligned;
                # a transition without any variation gets a placeholder item that is ignored.
                seq.append(' '.join(str(i) for i in sorted(items)) if items else '0')
            lines.append(' -1 '.join(seq) + ' -1 -2')
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        logger.debug(f"Created {len(lines)} sequences of {L - 1} transitions")

    def _extract_periodic_patterns(self, input_file: str, output_file: str):
        """Periodic patterns common to several sequences, with SPMF's MPFPS-BFS."""
        n_seq = self._n_cycles()
        min_ra = self.min_ra if self.min_ra is not None else (1 / self.cycle_size)
        max_per = self.max_periodicity if self.max_periodicity is not None else self.cycle_size
        max_std = self.max_std if self.max_std is not None else self.cycle_size
        min_seq = self.min_support if self.min_support >= 1 else max(1, int(np.ceil(self.min_support * n_seq)))
        try:
            from spmf import Spmf
            kw = {}
            if self.spmf_path:
                kw['spmf_bin_location_dir'] = os.path.dirname(os.path.abspath(self.spmf_path))
            spmf = Spmf("MPFPS-BFS", input_filename=input_file, output_filename=output_file,
                        arguments=[max_std, min_ra, max_per, int(min_seq)], **kw)
            spmf.run()
            return True
        except ImportError:
            logger.error("SPMF library not available. Install with: pip install spmf")
            return False
        except Exception as e:
            logger.error(f"SPMF execution failed: {e}")
            return False

    def _extract_seasonal_patterns(self, input_file: str, output_file: str):
        """Parse MPFPS output lines "<items> #RA:<ra> #SIDOCC: <sid>[<pos>] ... <sid>[...]".
        The support is the fraction of cycles (sequence ids) in which the pattern is periodic."""
        n_seq = self._n_cycles()
        patterns_data = []
        with open(input_file) as f:
            for line in f:
                if line.startswith('@') or '#SIDOCC' not in line:
                    continue
                head, _, occ = line.partition('#SIDOCC:')
                codes = [int(t) for t in head.split('#')[0].split()]
                if 0 in codes:
                    continue
                sids = set(re.findall(r'(\d+)\[', occ))
                patterns_data.append({'items': [self._decode(c) for c in codes],
                                      'support': len(sids) / n_seq})
        with open(output_file, 'w') as f:
            for p in patterns_data:
                f.write(' '.join(p['items']) + ' : ' + str(p['support']) + '\n')
        return patterns_data

    def _remove_redundancy(self, patterns_data: List[dict]) -> List[GradualPattern]:
        """One pattern per item set; patterns holding an attribute in both directions and, unless
        include_size_one, single items are dropped."""
        min_size = 1 if self.include_size_one else 2
        unique = {}
        for p in patterns_data:
            key = tuple(sorted(p['items']))
            attrs = [i[:-1] for i in key]
            if len(key) < min_size or len(set(attrs)) < len(attrs):
                continue
            if key not in unique or p['support'] > unique[key]['support']:
                unique[key] = p
        out = []
        for key, p in unique.items():
            gp = GradualPattern()
            for item in key:
                gp.add_gradual_item(GradualItem(int(item[:-1]), item[-1]))
            gp.set_support(p['support'])
            out.append(gp)
        return out

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
