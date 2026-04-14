"""Tests for EmergingGP (Mining Emerging Gradual Patterns).

Reference:
    Laurent, Lesot, Rifqi (2015). Mining Emerging Gradual Patterns.
    IFSA-EUSFLAT. DOI: 10.2991/ifsa-eusflat-15.2015.234
"""
import pytest
import pandas as pd
from gpmf.algorithms.emerging_gp import EmergingGP


@pytest.fixture
def d1_no_positive():
    """D1: A and B are anti-correlated (A+, B- holds, A+B+ does not)."""
    return pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0],
        'B': [8.0, 6.0, 4.0, 2.0],
    })


@pytest.fixture
def d2_positive():
    """D2: A and B are perfectly correlated (A+, B+ holds)."""
    return pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0],
        'B': [2.0, 4.0, 6.0, 8.0],
    })


@pytest.fixture
def d_identical():
    """Same data used as both D1 and D2 — no emerging patterns expected."""
    return pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0],
        'B': [2.0, 4.0, 6.0, 8.0],
    })


class TestEmergingGP:
    def test_returns_patterns(self, d1_no_positive, d2_positive):
        egp = EmergingGP(min_support_d1=0.5, min_support_d2=0.5)
        patterns = egp.fit(d1_no_positive, d2_positive).get_patterns()
        assert isinstance(patterns, list)

    def test_finds_emerging_pattern(self, d1_no_positive, d2_positive):
        """A+B+ holds in D2 but not D1 → should be an emerging pattern."""
        egp = EmergingGP(min_support_d1=0.5, min_support_d2=0.5)
        patterns = egp.fit(d1_no_positive, d2_positive).get_patterns()
        found = any(
            {(gi.attribute_col, gi.symbol) for gi in p.gradual_items}
            == {(0, '+'), (1, '+')}
            for p in patterns
        )
        assert found, "Expected {A+, B+} to be an emerging pattern"

    def test_no_emerging_when_identical(self, d_identical):
        """If D1 == D2 with same threshold, no pattern can emerge."""
        egp = EmergingGP(min_support_d1=0.5, min_support_d2=0.5)
        patterns = egp.fit(d_identical, d_identical).get_patterns()
        assert patterns == []

    def test_not_fitted_raises(self):
        egp = EmergingGP()
        with pytest.raises(Exception):
            egp.get_patterns()

    def test_invalid_support_raises(self):
        with pytest.raises(ValueError):
            EmergingGP(min_support_d1=0)

    def test_execution_time_recorded(self, d1_no_positive, d2_positive):
        egp = EmergingGP(min_support_d1=0.5, min_support_d2=0.5)
        egp.fit(d1_no_positive, d2_positive)
        assert egp.execution_time >= 0

    def test_fitted_flag(self, d1_no_positive, d2_positive):
        egp = EmergingGP(min_support_d1=0.5, min_support_d2=0.5)
        assert not egp.is_fitted
        egp.fit(d1_no_positive, d2_positive)
        assert egp.is_fitted

    def test_symmetric_threshold(self, d1_no_positive, d2_positive):
        """Lowering s2 should find at least as many patterns as higher s2."""
        egp_high = EmergingGP(min_support_d1=0.5, min_support_d2=0.9)
        egp_low = EmergingGP(min_support_d1=0.5, min_support_d2=0.5)
        p_high = egp_high.fit(d1_no_positive, d2_positive).get_patterns()
        p_low = egp_low.fit(d1_no_positive, d2_positive).get_patterns()
        assert len(p_low) >= len(p_high)

    def test_repr(self):
        egp = EmergingGP(min_support_d1=0.6, min_support_d2=0.7, base_algo='graank')
        r = repr(egp)
        assert 'EmergingGP' in r
        assert '0.6' in r
        assert '0.7' in r
