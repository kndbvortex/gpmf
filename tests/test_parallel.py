"""Parallel miners return exactly what their sequential versions return.

ParaMiner (Rust) runs the candidate tests of each node of its search in parallel; PGLCM mines the
subtree of each root item in a separate process. Both must give the same patterns and supports as one
thread, including on data with ties, where the support of ParaMiner depends on the search path.
"""
import numpy as np
import pandas as pd
import pytest

from gpmf import GradualPatternMiner


def _data(seed: int, n: int = 40, m: int = 6, levels: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 1))
    x = base @ rng.normal(size=(1, m)) + 0.5 * rng.normal(size=(n, m))
    if levels:  # discretise to create many ties
        x = np.round(x * levels) / levels
    return pd.DataFrame(x, columns=[f"a{i}" for i in range(m)])


def _canon(patterns):
    return sorted((tuple(sorted(p.to_string())) if isinstance(p.to_string(), list) else (p.to_string(),),
                   round(float(p.support), 12)) for p in patterns)


def _paraminer_available() -> bool:
    try:
        from gpmf.algorithms.closed.paraminer import paraminer_gradual_rust  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _paraminer_available(), reason="Rust extension not built")
@pytest.mark.parametrize("seed,levels,support", [(0, 0, 0.2), (1, 2, 0.2), (2, 1, 0.3)])
def test_paraminer_threads_match_sequential(seed, levels, support):
    df = _data(seed, levels=levels)
    ref = _canon(GradualPatternMiner("paraminer", df, min_support=support, use_rust=True,
                                     num_threads=1).mine())
    assert ref, "the test needs at least one pattern"
    for t in (2, 4):
        got = _canon(GradualPatternMiner("paraminer", df, min_support=support, use_rust=True,
                                         num_threads=t).mine())
        assert got == ref


@pytest.mark.parametrize("seed,levels,support", [(0, 0, 0.2), (1, 2, 0.2)])
def test_pglcm_processes_match_sequential(seed, levels, support):
    df = _data(seed, levels=levels)
    ref = _canon(GradualPatternMiner("pglcm", df, min_support=support, n_jobs=1).mine())
    assert ref
    got = _canon(GradualPatternMiner("pglcm", df, min_support=support, n_jobs=3).mine())
    assert got == ref
