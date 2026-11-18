"""Technical indicator utilities.

Technical indicators (TA-Lib), rolling CVaR and CVaR-based action labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOLD = 0
BUY = 1
SELL = 2


def compute_technical_indicators(
    high_prices: pd.Series,
    low_prices: pd.Series,
    close_prices: pd.Series,
    volume: pd.Series,
) -> pd.DataFrame:
    """Compute the technical indicators of a single ticker.

    The four series must share the same DatetimeIndex. The returned DataFrame
    keeps that index (one row per trading day, one column per indicator).
    Warm-up rows, where an indicator is not defined yet, contain NaN.
    """
    import talib as ta

    high = high_prices.astype(np.float64).to_numpy()
    low = low_prices.astype(np.float64).to_numpy()
    close = close_prices.astype(np.float64).to_numpy()
    volume_values = volume.astype(np.float64).to_numpy()

    bollinger_upper, bollinger_middle, bollinger_lower = ta.BBANDS(close)

    indicators = {
        "MACD": ta.MACD(close)[0],
        "RSI": ta.RSI(close),
        "CCI": ta.CCI(high, low, close),
        "ADX": ta.ADX(high, low, close),
        "STOCH_K": ta.STOCH(high, low, close)[0],
        "WILLR": ta.WILLR(high, low, close),
        "BB_Upper": bollinger_upper,
        "BB_Middle": bollinger_middle,
        "BB_Lower": bollinger_lower,
        "MFI": ta.MFI(high, low, close, volume_values),
        "EMA": ta.EMA(close),
        "ATR": ta.ATR(high, low, close),
        "SAR": ta.SAR(high, low),
        "OBV": ta.OBV(close, volume_values),
    }
    return pd.DataFrame(indicators, index=close_prices.index)


def compute_indicators_for_all_tickers(
    ohlcv: pd.DataFrame,
    tickers: list[str] | None = None,
    drop_warmup_rows: bool = True,
) -> pd.DataFrame:
    """Compute indicators for every ticker of a yfinance multi-ticker download.

    :param ohlcv: output of ``yf.download(tickers, ...)``, with
        ``(Price, Ticker)`` MultiIndex columns ("Close", "High", ...).
    :param tickers: tickers to process (default: all tickers in ``ohlcv``).
    :param drop_warmup_rows: drop the first days where at least one indicator
        is NaN, instead of filling them with an artificial value.
    :return: long DataFrame with columns ``Date``, ``Ticker`` + one column per
        indicator.
    """
    if tickers is None:
        tickers = list(ohlcv["Close"].columns)

    indicators_per_ticker = []
    for ticker in tickers:
        ticker_indicators = compute_technical_indicators(
            high_prices=ohlcv["High"][ticker],
            low_prices=ohlcv["Low"][ticker],
            close_prices=ohlcv["Close"][ticker],
            volume=ohlcv["Volume"][ticker],
        )
        if drop_warmup_rows:
            ticker_indicators = ticker_indicators.dropna()
        ticker_indicators.index.name = "Date"
        ticker_indicators.insert(0, "Ticker", ticker)
        indicators_per_ticker.append(ticker_indicators.reset_index())

    return pd.concat(indicators_per_ticker, ignore_index=True)


def compute_rolling_cvar(
    returns: pd.DataFrame,
    window: int,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Rolling Conditional Value at Risk (expected shortfall) of each column.

    For each window, CVaR is the mean of the returns that are at or below the
    ``alpha`` percentile. With ``window=1`` the CVaR is the return itself.
    The first ``window - 1`` rows are NaN.
    """

    def window_cvar(window_returns: np.ndarray) -> float:
        value_at_risk = np.percentile(window_returns, alpha * 100)
        tail_returns = window_returns[window_returns <= value_at_risk]
        return tail_returns.mean() if len(tail_returns) > 0 else 0.0

    return returns.rolling(window=window).apply(window_cvar, raw=True)


def generate_action_labels(
    short_term_cvar: pd.DataFrame,
    long_term_cvar: pd.DataFrame,
) -> pd.DataFrame:
    """Build Buy / Sell / Hold labels by comparing two CVaR horizons.

    - short-term CVaR < long-term CVaR -> BUY (1)
    - short-term CVaR > long-term CVaR -> SELL (2)
    - equal                            -> HOLD (0)
    - missing value on either side     -> NaN

    :return: wide DataFrame (dates x tickers) of float labels.
    """
    short_term_cvar, long_term_cvar = short_term_cvar.align(long_term_cvar, join="inner")

    labels = np.select(
        [
            short_term_cvar < long_term_cvar,
            short_term_cvar > long_term_cvar,
            short_term_cvar == long_term_cvar,
        ],
        [BUY, SELL, HOLD],
        default=np.nan,
    )
    return pd.DataFrame(labels, index=short_term_cvar.index, columns=short_term_cvar.columns)


def count_actions_per_ticker(action_labels: pd.DataFrame) -> pd.DataFrame:
    """Number of Hold / Buy / Sell labels for each ticker."""
    counts = {
        "Hold (0)": (action_labels == HOLD).sum(),
        "Buy (1)": (action_labels == BUY).sum(),
        "Sell (2)": (action_labels == SELL).sum(),
    }
    return pd.DataFrame(counts)
