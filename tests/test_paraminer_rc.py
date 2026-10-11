"""Row-column pruning in ParaMiner changes the cost of the search, never its output.

The Rust and Python back-ends both apply the criterion to the graph on which the support is computed;
with and without it, and in both back-ends, the patterns and supports must be identical, including on
data with ties.
"""
import pytest

from gpmf import GradualPatternMiner

from .test_parallel import _canon, _data, _paraminer_available


@pytest.mark.skipif(not _paraminer_available(), reason="Rust extension not built")
@pytest.mark.parametrize("seed,levels,support", [(0, 0, 0.2), (1, 2, 0.2), (2, 1, 0.3), (3, 0, 0.4)])
def test_paraminer_rc_pruning_keeps_patterns(seed, levels, support):
    df = _data(seed, levels=levels)

    def run(rust, rc):
        return _canon(GradualPatternMiner("paraminer", df, min_support=support, use_rust=rust,
                                          num_threads=1, use_rc_pruning=rc).mine())

    ref = run(True, False)
    assert ref, "the test needs at least one pattern"
    assert run(True, True) == ref
    assert run(False, True) == ref
