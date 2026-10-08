"""rc-pruning in GLCM and PGLCM changes the cost of the search, never its output."""
import numpy as np
import pandas as pd
import pytest

from gpmf.factory import AlgorithmRegistry
from gpmf.core.dataset import GradualDataset


def _patterns(name, df, s, rc):
    kw = dict(n_jobs=1) if name == "pglcm" else {}
    a = AlgorithmRegistry.get(name)(min_support=s, use_rc_pruning=rc, **kw)
    a.fit(GradualDataset(df, min_sup=s))
    return {(tuple(sorted((g.attribute_col, g.symbol) for g in p.gradual_items)), round(float(p.support), 9))
            for p in a.get_patterns()}


@pytest.mark.parametrize("name", ["glcm", "pglcm"])
@pytest.mark.parametrize("seed,s", [(0, 0.2), (1, 0.3), (2, 0.25)])
def test_rc_pruning_keeps_patterns(name, seed, s):
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(size=40))
    df = pd.DataFrame({f"A{i}": base * rng.choice([-1, 1]) + rng.normal(0, 2, 40) for i in range(6)})
    assert _patterns(name, df, s, False) == _patterns(name, df, s, True)
