"""Offline tests of the data pipeline (no download, no TA-Lib needed)."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from rl_stocks.utils import data, indicators, pipeline, transitions, validation
from rl_stocks.utils.indicators import BUY, HOLD, SELL

N_DAYS = 120
TICKERS = ["AAA", "BBB", "CCC"]
MARKET_TICKERS = ["^GSPC", "^VIX"]


def make_fake_ohlcv(tickers, n_days=N_DAYS, seed=0):
    """Random-walk OHLCV with the same column layout as yf.download."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days, name="Date")
    frames = {}
    for ticker in tickers:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
        frames[("Close", ticker)] = close
        frames[("High", ticker)] = close * 1.01
        frames[("Low", ticker)] = close * 0.99
        frames[("Open", ticker)] = close
        frames[("Volume", ticker)] = rng.integers(1_000, 10_000, n_days).astype(float)
    ohlcv = pd.DataFrame(frames, index=dates)
    ohlcv.columns = pd.MultiIndex.from_tuples(ohlcv.columns, names=["Price", "Ticker"])
    return ohlcv


def fake_technical_indicators(high_prices, low_prices, close_prices, volume):
    """Stand-in for TA-Lib: two indicators with a 5-day warm-up."""
    return pd.DataFrame(
        {
            "SMA_5": close_prices.rolling(5).mean(),
            "RANGE": high_prices - low_prices,
        },
        index=close_prices.index,
    )


@pytest.fixture
def prepared(monkeypatch):
    monkeypatch.setattr(indicators, "compute_technical_indicators", fake_technical_indicators)
    return pipeline.build_training_data(
        make_fake_ohlcv(TICKERS), make_fake_ohlcv(MARKET_TICKERS, seed=1), sequence_length=10
    )


def test_generate_action_labels():
    short = pd.DataFrame({"X": [-0.03, -0.01, -0.02, np.nan]})
    long = pd.DataFrame({"X": [-0.02, -0.02, -0.02, -0.02]})
    labels = indicators.generate_action_labels(short, long)["X"].tolist()
    assert labels[:3] == [BUY, SELL, HOLD]
    assert np.isnan(labels[3])


def test_rolling_cvar_window_one_equals_returns():
    returns = pd.DataFrame({"X": [0.01, -0.02, 0.03]})
    cvar = indicators.compute_rolling_cvar(returns, window=1)
    pd.testing.assert_frame_equal(cvar, returns)


def test_drop_first_rows_per_ticker():
    long_df = pd.DataFrame(
        {"Date": list(range(5)) * 2, "Ticker": ["A"] * 5 + ["B"] * 5, "v": range(10)}
    )
    trimmed = data.drop_first_rows_per_ticker(long_df, n_rows=2)
    assert trimmed.groupby("Ticker").size().to_dict() == {"A": 3, "B": 3}
    assert trimmed.groupby("Ticker")["Date"].min().to_dict() == {"A": 2, "B": 2}


def test_compute_cvar_reward():
    actions = np.array([BUY, SELL, HOLD, BUY])
    cvar = np.array([-0.1, 0.1, 0.1, 0.1])
    np.testing.assert_allclose(
        transitions.compute_cvar_reward(actions, cvar), [-0.15, 0.12, -0.08, -0.1]
    )


def test_pipeline_outputs_are_aligned(prepared):
    feature_df = prepared["feature_df"]
    validation.check_aligned_rows(
        ("feature_df", feature_df),
        ("action_labels_df", prepared["action_labels_df"]),
        ("aligned_cvar_df", prepared["aligned_cvar_df"]),
    )
    assert set(feature_df["Ticker"]) == set(TICKERS)
    assert set(prepared["feature_columns"]) == {"SMA_5", "RANGE", *MARKET_TICKERS}
    assert not prepared["merged_data"].isna().any().any()


def test_labels_match_their_own_date(prepared):
    """Each row must carry the label of its own (Date, Ticker), not a constant."""
    merged = prepared["merged_data"].set_index(["Date", "Ticker"])
    expected = data.wide_to_long(prepared["action_labels"], "Action").set_index(["Date", "Ticker"])
    pd.testing.assert_series_equal(
        merged["Action"], expected.loc[merged.index, "Action"], check_names=False
    )


def test_cvar_dataset(prepared):
    dataset = prepared["dataset"]
    n_features = len(prepared["feature_columns"]) * len(TICKERS)

    assert isinstance(dataset, transitions.CVaRDataset)
    assert len(dataset) > 0
    assert dataset.tickers == TICKERS
    assert dataset.state_shape == (10, n_features)
    assert dataset.dones[-1].all() and not dataset.dones[:-1].any()
    # next_state of transition t is the state of transition t + 1
    assert torch.equal(dataset.states[1:], dataset.next_states[:-1])

    state, action, reward, next_state, done = dataset[0]
    assert state.dtype == torch.float32
    assert state.shape == next_state.shape == (10, n_features)
    assert action.shape == reward.shape == done.shape == (len(TICKERS),)


def test_cvar_dataset_with_dataloader(prepared):
    loader = DataLoader(prepared["dataset"], batch_size=8, shuffle=True)
    states, actions, rewards, next_states, dones = next(iter(loader))
    assert states.shape[0] == actions.shape[0] == 8
    assert states.shape[1:] == prepared["dataset"].state_shape
