import numpy as np
import logging
from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor

from ..base_algorithm import BaseAlgorithm
from ...core.data_structures import GradualItem, GradualPattern, TimeLag, TemporalGradualPattern
from ..graank import GRAANK

logger = logging.getLogger(__name__)


class TGrad(BaseAlgorithm):
    # Temporal gradual pattern mining via fuzzy membership + data transformation

    def __init__(self, min_support: float = 0.5, target_col: int = 0,
                 min_rep: float = 0.5, n_jobs: int = 1, **kwargs):
        super().__init__(min_support, **kwargs)
        self.target_col = target_col
        self.min_rep = min_rep
        self.n_jobs = n_jobs
        self._params.update({
            'target_col': target_col,
            'min_rep': min_rep,
            'n_jobs': n_jobs
        })

    def _get_time_diffs(self, step: int) -> Tuple[bool, List]:
        if len(self.dataset.time_cols) == 0:
            return False, []

        time_col = self.dataset.time_cols[0]
        time_data = self.dataset.data[:, time_col]

        time_diffs = []
        n = self.dataset.row_count

        for i in range(n - step):
            try:
                t1 = float(time_data[i])
                t2 = float(time_data[i + step])
                time_diffs.append(abs(t2 - t1))
            except (ValueError, TypeError):
                return False, [i, i + step]

        return True, time_diffs

    def _transform_data(self, step: int) -> Optional[np.ndarray]:
        n = self.dataset.row_count
        transformed = []

        for col_idx in range(self.dataset.col_count):
            if col_idx == self.target_col or col_idx in self.dataset.time_cols:
                temp_row = self.dataset.data[:(n - step), col_idx]
            else:
                temp_row = self.dataset.data[step:n, col_idx]

            transformed.append(temp_row)

        return np.array(transformed).T

    def _compute_time_lag(self, time_diffs: List, support: float) -> TimeLag:
        if not time_diffs:
            return TimeLag(0, 0)

        try:
            import skfuzzy as fuzzy

            time_array = np.array(time_diffs)
            universe = np.linspace(time_array.min(), time_array.max(), 100)

            mf_low = fuzzy.trimf(universe, [universe.min(), universe.min(), np.median(universe)])
            mf_med = fuzzy.trimf(universe, [universe.min(), np.median(universe), universe.max()])
            mf_high = fuzzy.trimf(universe, [np.median(universe), universe.max(), universe.max()])

            memberships = []
            for val in time_array:
                idx = np.argmin(np.abs(universe - val))
                memberships.append(max(mf_low[idx], mf_med[idx], mf_high[idx]))

            avg_membership = np.mean(memberships)
            median_time = float(np.median(time_array))

            return TimeLag(median_time, support * avg_membership, "~")

        except ImportError:
            median_time = float(np.median(time_diffs))
            return TimeLag(median_time, support, "~")

    def _transform_and_mine(self, step: int) -> List[TemporalGradualPattern]:
        ok, time_diffs = self._get_time_diffs(step)
        if not ok:
            return []

        transformed_data = self._transform_data(step)
        if transformed_data is None or len(transformed_data) < 2:
            return []

        from ...core.dataset import GradualDataset
        temp_dataset = GradualDataset.__new__(GradualDataset)
        temp_dataset.data = transformed_data
        temp_dataset.thd_supp = self.dataset.thd_supp
        temp_dataset.equal = self.dataset.equal
        temp_dataset.row_count = len(transformed_data)
        temp_dataset.col_count = self.dataset.col_count
        temp_dataset.attr_cols = self.dataset.attr_cols
        temp_dataset.time_cols = self.dataset.time_cols
        temp_dataset.titles = self.dataset.titles
        temp_dataset.no_bins = False
        temp_dataset.attr_size = temp_dataset.row_count

        graank = GRAANK(min_support=self.min_support)
        graank.dataset = temp_dataset

        try:
            base_patterns = graank._mine()
        except Exception as e:
            logger.debug(f"Mining failed for step {step}: {e}")
            return []

        temporal_patterns = []
        for gp in base_patterns:
            tgp = TemporalGradualPattern()
            for gi in gp.gradual_items:
                tgp.add_gradual_item(gi)
            tgp.set_support(gp.support)

            time_lag = self._compute_time_lag(time_diffs, gp.support)
            tgp.add_time_lag(time_lag)

            temporal_patterns.append(tgp)

        return temporal_patterns

    def _mine(self) -> List[TemporalGradualPattern]:
        if len(self.dataset.time_cols) == 0:
            logger.error("No datetime columns found in dataset")
            return []

        max_step = self.dataset.row_count - int(self.min_rep * self.dataset.row_count)

        all_patterns = []

        if self.n_jobs > 1:
            try:
                with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    results = executor.map(self._transform_and_mine, range(1, max_step + 1))
                    for patterns in results:
                        if patterns:
                            all_patterns.extend(patterns)
            except Exception as e:
                logger.warning(f"Parallel execution failed: {e}, falling back to sequential")
                self.n_jobs = 1

        if self.n_jobs == 1:
            for step in range(1, max_step + 1):
                patterns = self._transform_and_mine(step)
                if patterns:
                    all_patterns.extend(patterns)

        unique_patterns = []
        seen = set()
        for tgp in all_patterns:
            key = tuple(sorted((gi.attribute_col, gi.symbol) for gi in tgp.gradual_items))
            if key not in seen:
                seen.add(key)
                unique_patterns.append(tgp)

        logger.info(f"TGrad found {len(unique_patterns)} temporal patterns")
        return unique_patterns
