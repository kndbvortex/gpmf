"""Dataset handling for gradual pattern mining."""
import numpy as np
import pandas as pd
import csv
import logging
from typing import Union, Tuple, List, Optional
from pathlib import Path
from dateutil.parser import parse
import time
import gc
from collections import defaultdict

from ..exceptions import InvalidDataError

logger = logging.getLogger(__name__)


class GradualDataset:
    """Handles data preprocessing and preparation for gradual pattern mining.

    This class loads, cleans, and prepares datasets for gradual pattern mining.
    It supports both CSV files and pandas DataFrames as input.

    Attributes:
        thd_supp: Minimum support threshold
        equal: Whether to encode equal values as gradual
        titles: Column names
        data: Dataset as numpy array
        row_count: Number of rows
        col_count: Number of columns
        time_cols: Indices of datetime columns
        attr_cols: Indices of numeric attribute columns
        valid_bins: Valid bitmaps for gradual items
        valid_tids: Valid transaction IDs for gradual items
        no_bins: True if no valid bitmaps exist

    Example:
        >>> dataset = GradualDataset('data.csv', min_sup=0.5)
        >>> dataset.fit_bitmap()
    """

    def __init__(self, data_source: Union[pd.DataFrame, str], min_sup: float = 0.5, eq: bool = False):
        """Initialize dataset.

        Args:
            data_source: Path to CSV file or pandas DataFrame
            min_sup: Minimum support threshold (0.0-1.0 for relative, >1 for absolute)
            eq: Encode equal values as gradual

        Raises:
            InvalidDataError: If data source is invalid
        """
        if min_sup <= 0:
            raise InvalidDataError("Minimum support must be positive")

        self.thd_supp = min_sup
        self.equal = eq
        self.titles, self.data = self._read(data_source)
        self.row_count, self.col_count = self.data.shape
        self.time_cols = self._get_time_cols()
        self.attr_cols = self._get_attr_cols()
        self.valid_bins = np.array([])
        self.valid_tids = defaultdict(set)
        self.no_bins = False
        self.step_name = ''  # For T-GRAANK
        self.attr_size = 0  # For T-GRAANK

        logger.info(f"Loaded dataset: {self.row_count} rows, {self.col_count} columns")
        logger.info(f"Attribute columns: {len(self.attr_cols)}, Time columns: {len(self.time_cols)}")

    def _get_attr_cols(self) -> np.ndarray:
        """Get indices of numeric attribute columns (non-datetime)."""
        all_cols = np.arange(self.col_count)
        attr_cols = np.setdiff1d(all_cols, self.time_cols)
        return attr_cols

    def _get_time_cols(self) -> np.ndarray:
        """Get indices of datetime columns."""
        time_cols = []
        for i in range(self.col_count):
            row_data = str(self.data[0][i])
            try:
                time_ok, _ = self._test_time(row_data)
                if time_ok:
                    time_cols.append(i)
            except ValueError:
                continue
        return np.array(time_cols)

    def get_gi_bitmap(self, col: int) -> np.ndarray:
        """Compute bitmap matrix for a specific attribute.

        Args:
            col: Column index

        Returns:
            Bitmap matrix as numpy array

        Raises:
            InvalidDataError: If column is invalid
        """
        if col in self.time_cols:
            col_name = self.titles[col][1].decode() if hasattr(self.titles[col][1], 'decode') else str(self.titles[col][1])
            raise InvalidDataError(f"{col_name} is a date/time column")
        elif col >= self.col_count:
            raise InvalidDataError("Column does not exist")

        attr_data = self.data.T
        col_data = np.array(attr_data[col], dtype=float)
        with np.errstate(invalid='ignore'):
            temp_pos = np.where(col_data < col_data[:, np.newaxis], 1, 0)
        return temp_pos

    def fit_bitmap(self, attr_data: Optional[np.ndarray] = None):
        """Generate bitmaps for all numeric columns.

        Stores valid bitmaps (support >= min_sup) in valid_bins attribute.

        Args:
            attr_data: Optional stepped attribute data (for T-GRAANK)
        """
        if attr_data is None:
            attr_data = self.data.T
            self.attr_size = self.row_count
        else:
            self.attr_size = len(attr_data[self.attr_cols[0]])

        n = self.attr_size
        valid_bins = []

        for col in self.attr_cols:
            col_data = np.array(attr_data[col], dtype=float)
            incr = np.array((col, '+'), dtype='i, S1')
            decr = np.array((col, '-'), dtype='i, S1')

            with np.errstate(invalid='ignore'):
                if not self.equal:
                    temp_pos = col_data > col_data[:, np.newaxis]
                else:
                    temp_pos = col_data >= col_data[:, np.newaxis]
                    np.fill_diagonal(temp_pos, 0)

                supp = float(np.sum(temp_pos)) / float(n * (n - 1.0) / 2.0)
                if supp >= self.thd_supp:
                    valid_bins.append(np.array([incr.tolist(), temp_pos], dtype=object))
                    valid_bins.append(np.array([decr.tolist(), temp_pos.T], dtype=object))

        self.valid_bins = np.array(valid_bins)
        if len(self.valid_bins) < 3:
            self.no_bins = True
            logger.warning("Insufficient valid bitmaps generated")

        logger.info(f"Generated {len(self.valid_bins)} valid bitmaps")
        gc.collect()

    def fit_tids(self):
        """Generate transaction IDs for all numeric columns.

        Stores valid tids (support >= min_sup) in valid_tids attribute.
        """
        self.fit_bitmap()
        n = self.row_count

        for bin_obj in self.valid_bins:
            arr_ij = np.transpose(np.nonzero(bin_obj[1]))
            set_ij = {tuple(ij) for ij in arr_ij if ij[0] < ij[1]}

            symbol = bin_obj[0][1].decode() if hasattr(bin_obj[0][1], 'decode') else bin_obj[0][1]
            int_gi = int(bin_obj[0][0] + 1) if symbol == '+' else (-1 * int(bin_obj[0][0] + 1))
            tids_len = len(set_ij)

            supp = float((tids_len * 0.5) * (tids_len - 1)) / float(n * (n - 1.0) / 2.0)
            if supp >= self.thd_supp:
                self.valid_tids[int_gi] = set_ij

        logger.info(f"Generated {len(self.valid_tids)} valid transaction IDs")

    @staticmethod
    def _read(data_src: Union[pd.DataFrame, str]) -> Tuple[List, np.ndarray]:
        """Read data from CSV file or DataFrame.

        Args:
            data_src: Path to CSV or DataFrame

        Returns:
            Tuple of (titles, data array)

        Raises:
            InvalidDataError: If data cannot be read
        """
        if isinstance(data_src, pd.DataFrame):
            try:
                _ = data_src.columns.astype(float)
                data_src.loc[-1] = data_src.columns.to_numpy(dtype=float)
                data_src.index = data_src.index + 1
                data_src.sort_index(inplace=True)
                vals = [f'col_{k}' for k in np.arange(data_src.shape[1])]
                data_src.columns = vals
            except (ValueError, TypeError):
                pass

            logger.debug("Data loaded from DataFrame")
            return GradualDataset._clean_data(data_src)
        else:
            file_path = Path(data_src)
            if not file_path.exists():
                raise InvalidDataError(f"File not found: {data_src}")

            try:
                with open(file_path, 'r') as f:
                    dialect = csv.Sniffer().sniff(f.readline(), delimiters=";,' '\t")
                    f.seek(0)
                    reader = csv.reader(f, dialect)
                    raw_data = list(reader)

                if len(raw_data) <= 1:
                    raise InvalidDataError("CSV file has insufficient data")

                keys = np.arange(len(raw_data[0]))
                if raw_data[0][0].replace('.', '', 1).isdigit():
                    vals = [f'col_{k}' for k in keys]
                    header = np.array(vals, dtype='S')
                else:
                    if raw_data[0][1].replace('.', '', 1).isdigit():
                        vals = [f'col_{k}' for k in keys]
                        header = np.array(vals, dtype='S')
                    else:
                        header = np.array(raw_data[0], dtype='S')
                        raw_data = np.delete(raw_data, 0, 0)

                d_frame = pd.DataFrame(raw_data, columns=header)
                logger.debug(f"Data loaded from CSV: {file_path}")
                return GradualDataset._clean_data(d_frame)

            except Exception as error:
                raise InvalidDataError(f"Error reading CSV: {error}")

    @staticmethod
    def _test_time(date_str: str) -> Tuple[bool, Union[float, bool]]:
        """Test if a string represents a datetime.

        Args:
            date_str: String to test

        Returns:
            Tuple of (is_datetime, timestamp)
        """
        try:
            if type(int(date_str)):
                return False, False
        except ValueError:
            try:
                if type(float(date_str)):
                    return False, False
            except ValueError:
                try:
                    date_time = parse(date_str)
                    t_stamp = time.mktime(date_time.timetuple())
                    return True, t_stamp
                except ValueError:
                    raise ValueError('No valid date-time format found')

    @staticmethod
    def _clean_data(df: pd.DataFrame) -> Tuple[List, np.ndarray]:
        """Clean DataFrame by removing null values and non-numeric columns.

        Args:
            df: DataFrame to clean

        Returns:
            Tuple of (titles, cleaned data)

        Raises:
            InvalidDataError: If dataframe is empty after cleaning
        """
        df = df.dropna()

        cols_to_remove = []
        for col in df.columns:
            try:
                _ = df[col].astype(float)
            except ValueError:
                try:
                    ok, _ = GradualDataset._test_time(str(df[col].iloc[0]))
                    if not ok:
                        cols_to_remove.append(col)
                except (ValueError, IndexError):
                    cols_to_remove.append(col)
            except TypeError:
                cols_to_remove.append(col)

        df = df[[col for col in df.columns if col not in cols_to_remove]]

        if df.empty:
            raise InvalidDataError("Dataset is empty after cleaning")

        keys = np.arange(df.shape[1])
        values = np.array(df.columns, dtype='S')
        titles = list(np.rec.fromarrays((keys, values), names=('key', 'value')))

        logger.debug(f"Data cleaned: {df.shape[0]} rows, {df.shape[1]} columns")
        return titles, df.values
