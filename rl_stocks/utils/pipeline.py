"""Data pipeline for training DQN / DDQN agents on stocks.

prices -> features + CVaR labels + CVaR -> offline transitions (CVaRDataset)

This pipeline is specific to value-based, off-policy agents (DQN and Double
DQN) trained offline on historical data, with one discrete action per ticker
(Hold / Buy / Sell) and a CVaR-based reward. See ``rl_stocks.utils.transitions``
for the full list of assumptions. Other algorithms (PPO, A2C, SAC, ...) need an
interactive environment and a different pipeline.

Typical use in a notebook::

    from rl_stocks.utils.pipeline import prepare_training_data

    prepared = prepare_training_data(tickers, start, end)
    dataset = prepared["dataset"]          # CVaRDataset, ready for a DataLoader
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from rl_stocks.utils import data, indicators, transitions, validation

DEFAULT_MARKET_TICKERS = ("^GSPC", "^VIX")


def build_training_data(
    ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
    sequence_length: int = 10,
    short_cvar_window: int = 10,
    long_cvar_window: int = 30,
    cvar_alpha: float = 0.05,
    normalize_cvar_for_reward: bool = False,
) -> dict:
    """Run every pipeline step on already downloaded prices (DQN / DDQN data).

    :param ohlcv: stocks OHLCV, ``(Price, Ticker)`` MultiIndex columns.
    :param market_ohlcv: market indices OHLCV (e.g. ^GSPC, ^VIX), same format.
    :return: dict with every intermediate result (see the keys below).
    """
    # 1. Returns
    stock_returns = data.compute_daily_returns(ohlcv["Close"])
    market_returns = data.compute_daily_returns(market_ohlcv["Close"])

    # 2. CVaR and Buy / Sell / Hold labels (wide: dates x tickers)
    daily_cvar = indicators.compute_rolling_cvar(stock_returns, window=1, alpha=cvar_alpha).bfill()
    short_term_cvar = indicators.compute_rolling_cvar(stock_returns, short_cvar_window, cvar_alpha).bfill()
    long_term_cvar = indicators.compute_rolling_cvar(stock_returns, long_cvar_window, cvar_alpha).bfill()
    action_labels = indicators.generate_action_labels(short_term_cvar, long_term_cvar)

    # 3. Features (long: one row per Date x Ticker)
    indicators_long = indicators.compute_indicators_for_all_tickers(
        ohlcv, tickers=list(stock_returns.columns)
    )
    features_long = data.add_market_returns(indicators_long, market_returns)
    features_long = data.drop_first_rows_per_ticker(features_long, n_rows=sequence_length)

    # 4. Merge features with labels and CVaR on (Date, Ticker)
    merged_data = data.merge_features_with_targets(
        features_long,
        action_labels_long=data.wide_to_long(action_labels, "Action"),
        cvar_long=data.wide_to_long(daily_cvar, "CVaR"),
    )
    validation.check_unique_date_ticker(merged_data, "merged_data")
    validation.check_no_missing_values(merged_data, "merged_data")

    # 5. Scaling and split into the three model inputs
    feature_columns = data.get_feature_columns(merged_data)
    validation.check_numeric_columns(merged_data, feature_columns, "merged_data")
    scaled_data, scaler = data.scale_feature_columns(merged_data, feature_columns)
    feature_df, action_labels_df, aligned_cvar_df = data.split_model_inputs(
        scaled_data, feature_columns
    )
    validation.check_aligned_rows(
        ("feature_df", feature_df),
        ("action_labels_df", action_labels_df),
        ("aligned_cvar_df", aligned_cvar_df),
    )

    # 6. RL transitions as a PyTorch Dataset
    dataset = transitions.CVaRDataset(
        feature_df,
        action_labels_df,
        aligned_cvar_df,
        sequence_length=sequence_length,
        normalize_cvar_for_reward=normalize_cvar_for_reward,
    )

    return {
        "stock_returns": stock_returns,
        "market_returns": market_returns,
        "action_labels": action_labels,
        "action_counts": indicators.count_actions_per_ticker(action_labels),
        "merged_data": merged_data,
        "feature_columns": feature_columns,
        "scaler": scaler,
        "feature_df": feature_df,
        "action_labels_df": action_labels_df,
        "aligned_cvar_df": aligned_cvar_df,
        "dataset": dataset,
    }


def prepare_training_data(
    tickers: Iterable[str],
    start,
    end,
    market_tickers: Iterable[str] = DEFAULT_MARKET_TICKERS,
    **pipeline_options,
) -> dict:
    """Download prices from Yahoo Finance, then run :func:`build_training_data`.

    ``pipeline_options`` are forwarded to :func:`build_training_data`
    (``sequence_length``, ``short_cvar_window``, ``long_cvar_window``, ...).
    """
    ohlcv = data.download_ohlcv(tickers, start, end)
    market_ohlcv = data.download_ohlcv(market_tickers, start, end)
    return build_training_data(ohlcv, market_ohlcv, **pipeline_options)
