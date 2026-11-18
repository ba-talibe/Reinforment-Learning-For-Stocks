"""Data loading and transformation utilities.

Download, reshaping (wide <-> long), merging and scaling of the datasets.

Conventions:
- "wide" DataFrame: index = dates, one column per ticker.
- "long" DataFrame: one row per (Date, Ticker) pair.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.preprocessing import StandardScaler

KEY_COLUMNS = ["Date", "Ticker"]
TARGET_COLUMNS = ["Action", "CVaR"]


def download_ohlcv(tickers: Iterable[str], start, end) -> pd.DataFrame:
    """Download daily OHLCV prices from Yahoo Finance.

    :return: DataFrame with ``(Price, Ticker)`` MultiIndex columns.
    """
    import yfinance as yf

    ohlcv = yf.download(list(tickers), start=start, end=end, progress=False)
    ohlcv.index.name = "Date"
    return ohlcv


def compute_daily_returns(close_prices: pd.DataFrame) -> pd.DataFrame:
    """Daily percentage returns; rows containing a NaN are dropped."""
    returns = close_prices.pct_change().dropna()
    returns.index.name = "Date"
    return returns


def wide_to_long(wide_df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Turn a (dates x tickers) DataFrame into ``Date, Ticker, value_name`` rows.

    Missing values are dropped.
    """
    long_df = wide_df.stack().rename(value_name).reset_index()
    long_df.columns = [*KEY_COLUMNS, value_name]
    return long_df.dropna(subset=[value_name]).reset_index(drop=True)


def add_market_returns(
    indicators_long: pd.DataFrame,
    market_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Attach market-wide returns (e.g. ^GSPC, ^VIX) to every ticker row, by date.

    Dates without market returns are dropped.
    """
    market_returns = market_returns.copy()
    market_returns.index.name = "Date"
    market_returns.columns = [str(column) for column in market_returns.columns]
    market_returns.columns.name = None
    return indicators_long.merge(market_returns.reset_index(), on="Date", how="inner")


def drop_first_rows_per_ticker(long_df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Drop the first ``n_rows`` dates of each ticker (not of the whole table)."""
    long_df = long_df.sort_values(KEY_COLUMNS[::-1])
    row_number_in_ticker = long_df.groupby("Ticker").cumcount()
    return long_df[row_number_in_ticker >= n_rows].reset_index(drop=True)


def merge_features_with_targets(
    features_long: pd.DataFrame,
    action_labels_long: pd.DataFrame,
    cvar_long: pd.DataFrame,
) -> pd.DataFrame:
    """Join features, action labels and CVaR on (Date, Ticker).

    Inner joins: only rows that have features, a label and a CVaR are kept.
    """
    merged = (
        features_long
        .merge(action_labels_long, on=KEY_COLUMNS, how="inner", validate="one_to_one")
        .merge(cvar_long, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    )
    merged = merged.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    other_columns = [column for column in merged.columns if column not in KEY_COLUMNS]
    return merged[KEY_COLUMNS + other_columns]


def get_feature_columns(merged_data: pd.DataFrame) -> list[str]:
    """Columns used as model inputs (everything except keys and targets)."""
    excluded = set(KEY_COLUMNS + TARGET_COLUMNS)
    return [column for column in merged_data.columns if column not in excluded]


def scale_feature_columns(
    merged_data: pd.DataFrame,
    feature_columns: list[str],
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize the feature columns.

    If ``scaler`` is None a new StandardScaler is fitted on ``merged_data``;
    otherwise the given (already fitted) scaler is only applied, which is what
    you want for test data.
    """
    scaled_data = merged_data.copy()
    if scaler is None:
        scaler = StandardScaler().fit(scaled_data[feature_columns])
    scaled_data[feature_columns] = scaler.transform(scaled_data[feature_columns])
    return scaled_data, scaler


def split_model_inputs(
    merged_data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the merged table into ``(feature_df, action_labels_df, aligned_cvar_df)``.

    All three keep the ``Date`` and ``Ticker`` columns and share the same index.
    """
    feature_df = merged_data[KEY_COLUMNS + feature_columns]
    action_labels_df = merged_data[KEY_COLUMNS + ["Action"]]
    aligned_cvar_df = merged_data[KEY_COLUMNS + ["CVaR"]]
    return feature_df, action_labels_df, aligned_cvar_df
