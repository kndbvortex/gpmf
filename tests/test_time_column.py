"""Miners that work on numeric attributes must ignore a date column (attr_cols), not fail on it."""
import numpy as np
import pandas as pd
import pytest

from gpmf.factory import AlgorithmRegistry
from gpmf.core.dataset import GradualDataset

MINERS = [
    ("sgrite", {}), ("sgopt", {}), ("sg1", {}), ("sgb2", {}),
    ("glcm", {}), ("pglcm", {"n_jobs": 1}),
    ("paraminer", {"use_rust": False}), ("paraminer", {"use_rust": True}),
]


def _data():
    rng = np.random.default_rng(0)
    a = np.arange(12, dtype=float)
    return pd.DataFrame({"A": a, "B": 2 * a + rng.normal(0, 3, 12), "C": rng.normal(0, 1, 12)})


def _patterns(name, kw, df):
    algo = AlgorithmRegistry.get(name)(min_support=0.3, **kw)
    algo.fit(GradualDataset(df, min_sup=0.3))
    return {(tuple(sorted((g.attribute_col, g.symbol) for g in p.gradual_items)), round(float(p.support), 9))
            for p in algo.get_patterns()}


@pytest.mark.parametrize("name,kw", MINERS)
def test_date_column_is_ignored(name, kw):
    df = _data()
    dated = df.copy()
    dated.insert(0, "date", pd.date_range("2020-01-01", periods=len(df), freq="D").strftime("%Y-%m-%d"))
    try:
        plain = _patterns(name, kw, df)
    except ImportError as e:  # Rust extension not built
        pytest.skip(str(e))
    shifted = {(tuple((c + 1, s) for c, s in items), sup) for items, sup in plain}
    assert plain, "the data should contain at least one pattern"
    assert _patterns(name, kw, dated) == shifted
