"""Tests for the GRAANK algorithm."""
import pytest
import pandas as pd
from gpmf.algorithms.graank import GRAANK


@pytest.fixture
def perfect_corr_df():
    """A and B perfectly correlated (A+, B+ should be the only maximal pattern)."""
    return pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0],
        'B': [2.0, 4.0, 6.0, 8.0],
    })


@pytest.fixture
def perfect_anticorr_df():
    """A and B perfectly anti-correlated."""
    return pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0],
        'B': [8.0, 6.0, 4.0, 2.0],
    })


@pytest.fixture
def example_df():
    """Standard example dataset used in documentation."""
    return pd.DataFrame({
        'A': [2.1, 8.0, 8.0, 0.04],
        'B': [0.3, 5.0, 0.2, 0.1],
        'C': [82.0, 25.0, 135.0, 60.0],
    })


class TestGRAANK:
    def test_returns_patterns(self, perfect_corr_df):
        algo = GRAANK(min_support=0.5)
        patterns = algo.fit(perfect_corr_df).get_patterns()
        assert len(patterns) > 0

    def test_perfect_positive_correlation(self, perfect_corr_df):
        algo = GRAANK(min_support=0.5)
        patterns = algo.fit(perfect_corr_df).get_patterns()
        # {A+, B+} should be found
        items = {(p.attribute_col, p.symbol) for pat in patterns for p in pat.gradual_items}
        # At least one pattern containing both A+(col0) and B+(col1)
        found = any(
            {(0, '+'), (1, '+')} == {(p.attribute_col, p.symbol) for p in pat.gradual_items}
            for pat in patterns
        )
        assert found

    def test_perfect_anti_correlation(self, perfect_anticorr_df):
        algo = GRAANK(min_support=0.5)
        patterns = algo.fit(perfect_anticorr_df).get_patterns()
        found = any(
            {(0, '+'), (1, '-')} == {(p.attribute_col, p.symbol) for p in pat.gradual_items}
            for pat in patterns
        )
        assert found

    def test_support_in_range(self, example_df):
        algo = GRAANK(min_support=0.5)
        patterns = algo.fit(example_df).get_patterns()
        for p in patterns:
            assert 0.5 <= p.support <= 1.0

    def test_no_patterns_below_threshold(self, perfect_corr_df):
        """With a threshold of 1.0 only perfect-support patterns survive."""
        algo = GRAANK(min_support=1.0)
        patterns = algo.fit(perfect_corr_df).get_patterns()
        for p in patterns:
            assert p.support == 1.0

    def test_patterns_are_maximal(self, example_df):
        """No returned pattern should be a sub-pattern of another."""
        algo = GRAANK(min_support=0.5)
        patterns = algo.fit(example_df).get_patterns()
        item_sets = [
            frozenset((gi.attribute_col, gi.symbol) for gi in p.gradual_items)
            for p in patterns
        ]
        for i, s_i in enumerate(item_sets):
            for j, s_j in enumerate(item_sets):
                if i != j:
                    assert not s_i < s_j, (
                        f"Pattern {list(s_i)} is a strict sub-pattern of {list(s_j)}"
                    )

    def test_get_result_metadata(self, example_df):
        algo = GRAANK(min_support=0.5)
        result = algo.mine_and_get_result(example_df)
        assert result.algorithm == 'GRAANK'
        assert result.min_support == 0.5
        assert result.execution_time >= 0

    def test_not_fitted_raises(self):
        algo = GRAANK(min_support=0.5)
        with pytest.raises(Exception):
            algo.get_patterns()

    def test_two_column_dataset(self):
        df = pd.DataFrame({'X': [1.0, 2.0, 3.0], 'Y': [3.0, 1.0, 2.0]})
        algo = GRAANK(min_support=0.5)
        patterns = algo.fit(df).get_patterns()
        assert isinstance(patterns, list)
