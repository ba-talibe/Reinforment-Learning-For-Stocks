"""Validation utilities.

Sanity checks used between the pipeline steps. Each function raises a
``ValueError`` with an explicit message instead of letting a later step fail
with an obscure error.
"""

from __future__ import annotations

import pandas as pd


def check_required_columns(df: pd.DataFrame, required_columns: list[str], name: str) -> None:
    """Raise if ``df`` is missing one of ``required_columns``."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{name}: missing columns {missing_columns}")


def check_unique_date_ticker(df: pd.DataFrame, name: str) -> None:
    """Raise if a (Date, Ticker) pair appears more than once."""
    check_required_columns(df, ["Date", "Ticker"], name)
    duplicated_rows = df.duplicated(subset=["Date", "Ticker"])
    if duplicated_rows.any():
        raise ValueError(f"{name}: {int(duplicated_rows.sum())} duplicated (Date, Ticker) rows")


def check_no_missing_values(df: pd.DataFrame, name: str, columns: list[str] | None = None) -> None:
    """Raise if ``columns`` (default: all) contain NaN."""
    columns = list(df.columns) if columns is None else columns
    missing_per_column = df[columns].isna().sum()
    missing_per_column = missing_per_column[missing_per_column > 0]
    if not missing_per_column.empty:
        raise ValueError(f"{name}: missing values {missing_per_column.to_dict()}")


def check_numeric_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    """Raise if one of ``columns`` is not numeric (e.g. a Date or a string)."""
    non_numeric_columns = [
        column for column in columns if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric_columns:
        raise ValueError(f"{name}: non-numeric feature columns {non_numeric_columns}")


def check_aligned_rows(*named_frames: tuple[str, pd.DataFrame]) -> None:
    """Raise if the DataFrames do not have exactly the same (Date, Ticker) rows."""
    reference_name, reference_df = named_frames[0]
    reference_keys = reference_df[["Date", "Ticker"]].reset_index(drop=True)
    for name, df in named_frames[1:]:
        keys = df[["Date", "Ticker"]].reset_index(drop=True)
        if not reference_keys.equals(keys):
            raise ValueError(f"{name} is not aligned with {reference_name} on (Date, Ticker)")
