"""Tests for bundled datasets."""
import pytest
import pandas as pd
from gpmf.datasets import load_example, load_air_quality


class TestLoadExample:
    def test_shape(self):
        df = load_example()
        assert df.shape == (4, 3)

    def test_columns(self):
        df = load_example()
        assert list(df.columns) == ['A', 'B', 'C']

    def test_numeric(self):
        df = load_example()
        assert df.dtypes.apply(lambda d: d.kind in ('f', 'i', 'u')).all()

    def test_no_nulls(self):
        df = load_example()
        assert not df.isnull().any().any()


class TestLoadAirQuality:
    def test_loads_dataframe(self):
        try:
            df = load_air_quality()
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert df.shape[1] > 0
        except FileNotFoundError:
            pytest.skip("Air quality dataset not available in this environment")

    def test_no_missing_values(self):
        try:
            df = load_air_quality()
            assert not df.isnull().any().any()
        except FileNotFoundError:
            pytest.skip("Air quality dataset not available in this environment")
