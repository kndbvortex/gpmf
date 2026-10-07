"""MSGP on the weekly rehabilitation example of the thesis (4 observations x 4 weeks).

Within every week, heart rate, blood pressure and dose rise and oxygen falls on Mon->Wed and
Wed->Fri and the reverse holds on Fri->Sun, so with a one-week cycle the pattern
{HR+, BP+, O2-, Dose+} is periodic in all four weeks (support 1.0).
"""
import os
import shutil

import pandas as pd
import pytest

from gpmf.algorithms.seasonal.msgp import MSGP

ROWS = [(70, 115, 98, 5), (75, 122, 97, 8), (82, 130, 95, 12), (79, 126, 96, 10),
        (68, 113, 99, 4), (74, 121, 97, 7), (85, 133, 94, 13), (80, 127, 96, 11),
        (71, 116, 98, 5), (76, 124, 96, 9), (83, 131, 95, 11), (78, 125, 97, 9),
        (69, 114, 99, 4), (73, 120, 97, 7), (88, 136, 93, 14), (81, 128, 95, 10)]

needs_spmf = pytest.mark.skipif(not os.environ.get("SPMF_PATH") or not shutil.which("java"),
                                reason="SPMF_PATH and java are required")


def data():
    return pd.DataFrame(ROWS, columns=["hr", "bp", "o2", "dose"])


def test_sequences_are_cycles(tmp_path):
    m = MSGP(min_support=0.5, cycle_size=4)
    from gpmf.core.dataset import GradualDataset
    m.dataset = GradualDataset(data(), min_sup=0.5)
    out = tmp_path / "seq.txt"
    m._numerical_to_sequential(str(out))
    lines = out.read_text().splitlines()
    assert len(lines) == 4                                   # one sequence per week
    up = " ".join(str(c) for c in sorted([2, 4, 5, 8]))      # hr+, bp+, o2-, dose+
    down = " ".join(str(c) for c in sorted([1, 3, 6, 7]))
    assert lines[0] == f"{up} -1 {up} -1 {down} -1 -2"


@needs_spmf
def test_weekly_pattern_has_full_support():
    pats = MSGP(min_support=0.5, cycle_size=4).mine(data())
    found = {tuple(sorted(p.to_string())): p.support for p in pats}
    assert found[("0+", "1+", "2-", "3+")] == 1.0
