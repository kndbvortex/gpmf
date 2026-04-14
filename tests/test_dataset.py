"""Tests for GradualDataset loading and preprocessing."""
import pytest
import pandas as pd
import numpy as np
from gpmf.core.dataset import GradualDataset


@pytest.fixture
def simple_df():
    """4-row, 3-column numeric dataframe."""
    return pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0],
        'B': [4.0, 3.0, 2.0, 1.0],
        'C': [1.0, 2.0, 3.0, 4.0],
    })


@pytest.fixture
def perfect_df():
    """Perfect positive correlation between A and B."""
    return pd.DataFrame({
        'A': [1.0, 2.0, 3.0],
        'B': [2.0, 4.0, 6.0],
    })


class TestGradualDataset:
    def test_load_from_dataframe(self, simple_df):
        ds = GradualDataset(simple_df, min_sup=0.5)
        assert ds.row_count == 4
        assert ds.col_count == 3

    def test_load_from_csv(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("A,B\n1,2\n2,4\n3,6\n")
        ds = GradualDataset(str(csv), min_sup=0.5)
        assert ds.row_count == 3

    def test_attr_cols_excludes_datetime(self):
        df = pd.DataFrame({
            'date': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03']),
            'A': [1.0, 2.0, 3.0],
            'B': [3.0, 2.0, 1.0],
        })
        ds = GradualDataset(df, min_sup=0.5)
        assert len(ds.attr_cols) == 2

    def test_invalid_support(self, simple_df):
        with pytest.raises(Exception):
            GradualDataset(simple_df, min_sup=0)

    def test_fit_bitmap(self, perfect_df):
        ds = GradualDataset(perfect_df, min_sup=0.5)
        ds.fit_bitmap()
        assert not ds.no_bins
        assert len(ds.valid_bins) > 0

    def test_no_numeric_columns(self):
        from gpmf.exceptions import InvalidDataError
        df = pd.DataFrame({'label': ['a', 'b', 'c']})
        with pytest.raises(InvalidDataError):
            GradualDataset(df, min_sup=0.5)
