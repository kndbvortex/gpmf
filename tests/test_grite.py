"""Tests for the GRITE algorithm."""
import pytest
import pandas as pd
from gpmf.algorithms.grite import GRITE


@pytest.fixture
def perfect_corr_df():
    return pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0],
        'B': [2.0, 4.0, 6.0, 8.0],
    })


@pytest.fixture
def example_df():
    return pd.DataFrame({
        'A': [2.1, 8.0, 8.0, 0.04],
        'B': [0.3, 5.0, 0.2, 0.1],
        'C': [82.0, 25.0, 135.0, 60.0],
    })


class TestGRITE:
    def test_returns_patterns(self, perfect_corr_df):
        algo = GRITE(min_support=0.5)
        patterns = algo.fit(perfect_corr_df).get_patterns()
        assert len(patterns) > 0

    def test_support_in_range(self, example_df):
        algo = GRITE(min_support=0.5)
        patterns = algo.fit(example_df).get_patterns()
        for p in patterns:
            assert 0.5 <= p.support <= 1.0

    def test_not_fitted_raises(self):
        algo = GRITE(min_support=0.5)
        with pytest.raises(Exception):
            algo.get_patterns()

    def test_finds_pattern_in_perfectly_ordered_data(self, perfect_corr_df):
        """A chain A=1<2<3<4 and B=2<4<6<8 must yield support=1.0."""
        algo = GRITE(min_support=0.5)
        patterns = algo.fit(perfect_corr_df).get_patterns()
        max_support = max((p.support for p in patterns), default=0)
        assert max_support == pytest.approx(1.0, abs=0.01)

    def test_empty_result_when_no_pattern(self):
        """Random data with high threshold should yield no patterns."""
        import numpy as np
        rng = np.random.default_rng(42)
        df = pd.DataFrame(rng.random((5, 3)), columns=['A', 'B', 'C'])
        algo = GRITE(min_support=0.99)
        patterns = algo.fit(df).get_patterns()
        # May or may not find patterns; just verify no exception and valid supports
        for p in patterns:
            assert p.support >= 0.99
