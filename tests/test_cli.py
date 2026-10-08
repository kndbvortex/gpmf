"""CLI options: --use-rc-pruning and --maximal reach the algorithm, or are refused when unsupported."""
import sys

import numpy as np
import pandas as pd
import pytest

from gpmf import cli
from gpmf.miner import GradualPatternMiner


@pytest.fixture
def csv_path(tmp_path):
    rng = np.random.default_rng(0)
    a = np.arange(15, dtype=float)
    p = tmp_path / "data.csv"
    pd.DataFrame({"A": a, "B": a + rng.normal(0, 2, 15), "C": rng.normal(0, 1, 15)}).to_csv(p, index=False)
    return str(p)


def run(monkeypatch, *argv):
    seen = {}
    real_init = GradualPatternMiner.__init__

    def spy(self, *a, **kw):
        real_init(self, *a, **kw)
        seen["algorithm"] = self.algorithm

    monkeypatch.setattr(GradualPatternMiner, "__init__", spy)
    monkeypatch.setattr(sys, "argv", ["gradual-mine", *argv, "--no-print"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    return e.value.code, seen.get("algorithm")


def test_rc_pruning_reaches_the_algorithm(monkeypatch, csv_path):
    code, algo = run(monkeypatch, "sgrite", csv_path, "-s", "0.3", "--use-rc-pruning")
    assert code == 0 and algo.use_rc_pruning is True


def test_rc_pruning_off_by_default(monkeypatch, csv_path):
    code, algo = run(monkeypatch, "sgrite", csv_path, "-s", "0.3")
    assert code == 0 and algo.use_rc_pruning is False


def test_maximal_reaches_graank(monkeypatch, csv_path):
    code, algo = run(monkeypatch, "graank", csv_path, "--maximal")
    assert code == 0 and algo.maximal is True


def test_unsupported_option_is_refused(monkeypatch, csv_path, capsys):
    code, _ = run(monkeypatch, "graank", csv_path, "--use-rc-pruning")
    assert code == 2
    assert "does not support --use-rc-pruning" in capsys.readouterr().err
