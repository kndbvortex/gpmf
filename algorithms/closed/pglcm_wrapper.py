"""PGLCM - Parallel Generic L-Close Miner for closed gradual patterns.

Reference: Mining closed gradual patterns with parallel computing.
"""
import subprocess
import tempfile
import os
import logging
from typing import List
from pathlib import Path

from ..base_algorithm import BaseAlgorithm
from ...core.data_structures import GradualItem, GradualPattern

logger = logging.getLogger(__name__)


class PGLCM(BaseAlgorithm):
    """PGLCM algorithm for mining closed gradual patterns.

    Wrapper for the C++ PGLCM implementation with parallel support.
    Falls back to Python implementation if binary not available.

    Args:
        min_support: Minimum support threshold
        n_jobs: Number of parallel threads
        pglcm_binary: Path to PGLCM binary

    Example:
        >>> pglcm = PGLCM(min_support=0.5, n_jobs=4)
        >>> patterns = pglcm.mine('data.csv')
    """

    def __init__(self, min_support: float = 0.5, n_jobs: int = 1,
                 pglcm_binary: str = None, **kwargs):
        super().__init__(min_support, **kwargs)
        self.n_jobs = n_jobs
        self.pglcm_binary = pglcm_binary or self._find_pglcm_binary()
        self._params['n_jobs'] = n_jobs

    def _find_pglcm_binary(self) -> str:
        """Try to find PGLCM binary in common locations."""
        possible_paths = [
            './pglcm',
            './bin/pglcm',
            '../codeGLCM_PGLCM/PGLCM/pglcm',
            '/usr/local/bin/pglcm'
        ]

        for path in possible_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                logger.info(f"Found PGLCM binary at: {path}")
                return path

        logger.warning("PGLCM binary not found, will use Python fallback")
        return None

    def _prepare_input_file(self, filepath: str):
        """Prepare input file in PGLCM format."""
        data = self.dataset.data
        n_rows = data.shape[0]

        with open(filepath, 'w') as f:
            f.write(f"{n_rows}\n")

            for col_idx in self.dataset.attr_cols:
                col_name = self.dataset.titles[col_idx][1]
                if hasattr(col_name, 'decode'):
                    col_name = col_name.decode()
                f.write(f"{col_name}\n")

                for row_idx in range(n_rows):
                    f.write(f"{data[row_idx][col_idx]} ")
                f.write("\n")

    def _parse_pglcm_output(self, output_file: str) -> List[GradualPattern]:
        """Parse PGLCM output file."""
        patterns = []

        if not os.path.exists(output_file):
            return patterns

        try:
            with open(output_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split(':')
                    if len(parts) < 2:
                        continue

                    items_str = parts[0].strip()
                    support_str = parts[1].strip()

                    gp = GradualPattern()

                    for item in items_str.split():
                        if len(item) >= 2:
                            try:
                                attr_idx = int(item[:-1])
                                symbol = item[-1]
                                if symbol in ['+', '-']:
                                    gi = GradualItem(attr_idx, symbol)
                                    gp.add_gradual_item(gi)
                            except ValueError:
                                continue

                    if len(gp.gradual_items) >= 2:
                        try:
                            support = float(support_str)
                            gp.set_support(support)
                            if support >= self.min_support:
                                patterns.append(gp)
                        except ValueError:
                            continue

        except Exception as e:
            logger.error(f"Error parsing PGLCM output: {e}")

        return patterns

    def _run_pglcm_binary(self, input_file: str, output_file: str) -> bool:
        """Run PGLCM C++ binary."""
        if not self.pglcm_binary:
            return False

        try:
            min_supp_param = int(self.min_support * 100) if self.min_support < 1 else int(self.min_support)

            cmd = [
                self.pglcm_binary,
                input_file,
                str(min_supp_param),
                str(self.n_jobs),
                output_file
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                logger.debug("PGLCM binary executed successfully")
                return True
            else:
                logger.warning(f"PGLCM binary failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("PGLCM execution timed out")
            return False
        except Exception as e:
            logger.error(f"Error running PGLCM binary: {e}")
            return False

    def _python_fallback(self) -> List[GradualPattern]:
        """Python fallback implementation for closed pattern mining."""
        logger.info("Using Python fallback for closed pattern mining")

        from ..parallel.paraminer import ParaMiner

        paraminer = ParaMiner(min_support=self.min_support, epsilon=2)
        paraminer.dataset = self.dataset

        try:
            return paraminer._mine()
        except Exception as e:
            logger.error(f"Python fallback failed: {e}")
            return []

    def _mine(self) -> List[GradualPattern]:
        """Mine closed gradual patterns using PGLCM."""

        if self.pglcm_binary:
            with tempfile.TemporaryDirectory() as tmpdir:
                input_file = os.path.join(tmpdir, 'input.txt')
                output_file = os.path.join(tmpdir, 'output.txt')

                self._prepare_input_file(input_file)

                if self._run_pglcm_binary(input_file, output_file):
                    patterns = self._parse_pglcm_output(output_file)
                    if patterns:
                        logger.info(f"PGLCM found {len(patterns)} patterns")
                        return patterns

        patterns = self._python_fallback()
        logger.info(f"PGLCM (fallback) found {len(patterns)} patterns")
        return patterns
