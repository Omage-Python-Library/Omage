"""
Omage Data — Data loading and preprocessing
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("omage")

SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls"}


class OmageDataError(Exception):
    pass


class Data:
    """Dataset handler with chainable operations"""

    def __init__(self, data: Optional[pd.DataFrame] = None, path: Optional[str] = None):
        self.data = data
        self.path = path
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self._batch_size: int = 32
        self._target_col: Optional[str] = None   # FIX #4: flexible target column

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str) -> "Data":
        """Load data from a CSV or Excel file"""
        p = Path(path)
        if not p.exists():                               # FIX #5: missing file
            raise OmageDataError(f"File not found: '{path}'")
        if p.suffix not in SUPPORTED_FORMATS:
            raise OmageDataError(
                f"Unsupported format '{p.suffix}'. Supported: {SUPPORTED_FORMATS}"
            )
        try:
            if p.suffix == ".csv":
                data = pd.read_csv(path)
            else:
                data = pd.read_excel(path)
        except Exception as exc:
            raise OmageDataError(f"Failed to read '{path}': {exc}") from exc

        logger.info(f"Loaded data from '{path}' — shape: {data.shape}")
        return cls(data=data, path=path)

    # ------------------------------------------------------------------ #
    # Target column  (FIX #4)
    # ------------------------------------------------------------------ #
    def set_target(self, column: str) -> "Data":
        """Set the target (label) column explicitly"""
        if self.data is None:
            raise OmageDataError("No data loaded")
        if column not in self.data.columns:
            raise OmageDataError(
                f"Column '{column}' not found. Available: {list(self.data.columns)}"
            )
        self._target_col = column
        logger.info(f"Target column set to '{column}'")
        return self

    def _get_xy(self, df: pd.DataFrame):
        """Split DataFrame into features X and labels y"""
        if self._target_col:
            X = df.drop(columns=[self._target_col]).values
            y = df[self._target_col].values
        else:
            X = df.iloc[:, :-1].values   # fallback: last column
            y = df.iloc[:, -1].values
        return X, y

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    def clean(self) -> "Data":
        """Remove null values and duplicates"""
        if self.data is None:
            raise OmageDataError("No data loaded")
        before = len(self.data)
        self.data = self.data.dropna().drop_duplicates().reset_index(drop=True)
        removed = before - len(self.data)
        logger.info(f"Cleaned data — removed {removed} rows, remaining: {len(self.data)}")
        return self

    def normalize(self, min_val: float = 0, max_val: float = 1) -> "Data":
        """Min-max normalize numeric columns (excludes target column)"""
        if self.data is None:
            raise OmageDataError("No data loaded")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if self._target_col and self._target_col in numeric_cols:
            numeric_cols = numeric_cols.drop(self._target_col)

        for col in numeric_cols:
            col_min, col_max = self.data[col].min(), self.data[col].max()
            if col_max != col_min:
                self.data[col] = (
                    (self.data[col] - col_min) / (col_max - col_min)
                    * (max_val - min_val) + min_val
                )
        logger.info(f"Normalized {len(numeric_cols)} columns — range: [{min_val}, {max_val}]")
        return self

    def standardize(self) -> "Data":
        """Z-score standardize numeric columns (excludes target column)"""
        if self.data is None:
            raise OmageDataError("No data loaded")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if self._target_col and self._target_col in numeric_cols:
            numeric_cols = numeric_cols.drop(self._target_col)

        for col in numeric_cols:
            mean, std = self.data[col].mean(), self.data[col].std()
            if std != 0:
                self.data[col] = (self.data[col] - mean) / std
        logger.info(f"Standardized {len(numeric_cols)} columns")
        return self

    def shuffle(self, seed: Optional[int] = None) -> "Data":
        if self.data is None:
            raise OmageDataError("No data loaded")
        self.data = self.data.sample(frac=1, random_state=seed).reset_index(drop=True)
        logger.info(f"Shuffled data — seed: {seed or 'random'}")
        return self

    def split(
        self,
        train_ratio: float = 0.8,
        test_ratio: Optional[float] = None,
    ) -> "Data":
        if self.data is None:
            raise OmageDataError("No data loaded")
        if test_ratio is None:
            test_ratio = 1.0 - train_ratio
        if not (0 < train_ratio < 1):
            raise OmageDataError("train_ratio must be between 0 and 1")

        n = len(self.data)
        train_size = int(n * train_ratio)
        self.train_data = self.data.iloc[:train_size].reset_index(drop=True)
        self.test_data  = self.data.iloc[train_size:].reset_index(drop=True)
        logger.info(f"Split — train: {len(self.train_data)} | test: {len(self.test_data)}")
        return self

    def batch(self, size: int = 32) -> "Data":
        if size <= 0:
            raise OmageDataError("Batch size must be > 0")
        self._batch_size = size
        logger.info(f"Batch size set to {size}")
        return self

    def filter(self, condition: str) -> "Data":
        """Filter rows — pass a pandas query string, e.g. 'age > 30'"""
        if self.data is None:
            raise OmageDataError("No data loaded")
        try:
            self.data = self.data.query(condition).reset_index(drop=True)
            logger.info(f"Filter applied: '{condition}' — remaining: {len(self.data)}")
        except Exception as exc:
            raise OmageDataError(f"Invalid filter condition '{condition}': {exc}") from exc
        return self

    # ------------------------------------------------------------------ #
    # DataLoaders
    # ------------------------------------------------------------------ #
    def _make_loader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        X_np, y_np = self._get_xy(df)
        X = torch.tensor(X_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.long)
        return DataLoader(TensorDataset(X, y), batch_size=self._batch_size, shuffle=shuffle)

    def get_train_loader(self) -> DataLoader:
        if self.train_data is None:
            logger.warning("No split found — auto-splitting 80/20")
            self.split()
        return self._make_loader(self.train_data, shuffle=True)

    def get_test_loader(self) -> DataLoader:
        if self.test_data is None:
            logger.warning("No split found — auto-splitting 80/20")
            self.split()
        return self._make_loader(self.test_data, shuffle=False)

    # ------------------------------------------------------------------ #
    # Dunder
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.data) if self.data is not None else 0

    def __repr__(self) -> str:
        shape = self.data.shape if self.data is not None else "empty"
        return f"Data(shape={shape}, target='{self._target_col or 'last column'}')"


# Helper
def load(path: str) -> Data:
    """Load data from file"""
    return Data.load(path)
