"""Bundled reference datasets for gradual pattern mining.

These datasets are used for testing, benchmarking, and tutorial examples.
They are loaded as pandas DataFrames ready to pass to any GPMF algorithm.

Available datasets
------------------
load_example()
    Tiny 4-row toy dataset (3 numeric attributes). Useful for unit tests and
    quick sanity checks.

load_vehicle(class_label=None)
    UCI Statlog Vehicle dataset — 18 shape descriptors measured on images of
    vans and buses. Used in the reference paper on Emerging Gradual Patterns
    (Laurent, Lesot, Rifqi, IFSA-EUSFLAT 2015).
    When class_label is 'van' or 'bus', returns only that subset.

load_air_quality()
    UCI Air Quality dataset — 13 sensor readings (hourly averages) from an
    air quality monitoring station in an Italian city.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd

_DATA_DIR = Path(__file__).parent / "_data"


def load_example() -> pd.DataFrame:
    """Return the 4-row toy dataset (3 numeric attributes).

    Returns:
        DataFrame with columns ['A', 'B', 'C'].

    Example:
        >>> from gpmf.datasets import load_example
        >>> df = load_example()
        >>> df.shape
        (4, 3)
    """
    data = """A,B,C
2.1,0.3,82
8.0,5.0,25
8.0,0.2,135
0.04,0.1,60"""
    return pd.read_csv(io.StringIO(data))


def load_vehicle(class_label: Optional[str] = None) -> pd.DataFrame:
    """Return the UCI Statlog Vehicle dataset.

    18 shape descriptors (compactness, circularity, elongatedness, …) measured
    on silhouettes of vans and buses. Used as the illustrative example in:
        Laurent, Lesot, Rifqi (2015). Mining Emerging Gradual Patterns.
        IFSA-EUSFLAT 2015. DOI: 10.2991/ifsa-eusflat-15.2015.234

    Args:
        class_label: If 'van' or 'bus', return only that subset.
            If None, return the full dataset with a 'class' column.

    Returns:
        DataFrame with 18 numeric feature columns and (optionally) a 'class' column.

    Example:
        >>> from gpmf.datasets import load_vehicle
        >>> vans = load_vehicle('van')
        >>> buses = load_vehicle('bus')
        >>> vans.shape[1] == buses.shape[1] == 18
        True
    """
    path = _DATA_DIR / "vehicle.csv"
    if not path.exists():
        raise FileNotFoundError(
            "UCI Vehicle dataset not found. "
            f"Place vehicle.csv in {_DATA_DIR} or download it from the UCI repository."
        )
    df = pd.read_csv(path)
    if class_label is not None:
        label = class_label.lower()
        if label not in ('van', 'bus'):
            raise ValueError("class_label must be 'van', 'bus', or None")
        df = df[df['class'].str.lower() == label].drop(columns=['class']).reset_index(drop=True)
    return df


def load_air_quality() -> pd.DataFrame:
    """Return a cleaned subset of the UCI Air Quality dataset.

    Sensor readings (CO, NOx, NO2, O3, Temperature, RH, AH) — hourly averages
    from an Italian monitoring station. Missing values (-200) are removed.

    Returns:
        DataFrame with numeric sensor columns.

    Example:
        >>> from gpmf.datasets import load_air_quality
        >>> df = load_air_quality()
        >>> df.shape[1] > 0
        True
    """
    import os

    # Walk up from package dir to find the project-level data directory
    candidates = [
        Path(__file__).parents[4] / "data" / "air+quality" / "AirQualityUCI.csv",
        _DATA_DIR / "air_quality.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Air Quality dataset not found. "
            "Expected at data/air+quality/AirQualityUCI.csv from the project root."
        )

    df = pd.read_csv(path, sep=';', decimal=',')
    # Drop non-numeric and identifier columns
    drop_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'unnamed' in c.lower()]
    df = df.drop(columns=drop_cols, errors='ignore')
    # Replace sentinel missing value
    df = df.replace(-200, pd.NA).dropna().reset_index(drop=True)
    return df
