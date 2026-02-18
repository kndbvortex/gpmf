"""Validation utilities for gradual mining."""
from typing import Union
import pandas as pd
from pathlib import Path

from ..exceptions import InvalidParameterError, InvalidDataError


def validate_support(support: float, param_name: str = "support") -> None:
    """Validate support threshold.

    Args:
        support: Support value to validate (0.0-1.0 for relative, >1 for absolute)
        param_name: Name of parameter (for error messages)

    Raises:
        InvalidParameterError: If support is invalid
    """
    if not isinstance(support, (int, float)):
        raise InvalidParameterError(f"{param_name} must be a number, got {type(support)}")

    if support <= 0:
        raise InvalidParameterError(f"{param_name} must be positive, got {support}")


def validate_data(data: Union[pd.DataFrame, str]) -> None:
    """Validate input data.

    Args:
        data: Data to validate (DataFrame or file path)

    Raises:
        InvalidDataError: If data is invalid
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise InvalidDataError("DataFrame is empty")

        if data.shape[0] < 2:
            raise InvalidDataError("DataFrame must have at least 2 rows")

        if data.shape[1] < 2:
            raise InvalidDataError("DataFrame must have at least 2 columns")

    elif isinstance(data, str):
        file_path = Path(data)
        if not file_path.exists():
            raise InvalidDataError(f"File not found: {data}")

        if not file_path.is_file():
            raise InvalidDataError(f"Path is not a file: {data}")

    else:
        raise InvalidDataError(f"Data must be DataFrame or file path, got {type(data)}")


def validate_n_jobs(n_jobs: int) -> int:
    """Validate and normalize n_jobs parameter.

    Args:
        n_jobs: Number of parallel jobs

    Returns:
        Normalized n_jobs value

    Raises:
        InvalidParameterError: If n_jobs is invalid
    """
    if not isinstance(n_jobs, int):
        raise InvalidParameterError(f"n_jobs must be an integer, got {type(n_jobs)}")

    if n_jobs == 0:
        raise InvalidParameterError("n_jobs cannot be 0")

    if n_jobs == -1:
        import multiprocessing
        return multiprocessing.cpu_count()

    if n_jobs < -1:
        raise InvalidParameterError(f"n_jobs must be -1 or positive, got {n_jobs}")

    return n_jobs
