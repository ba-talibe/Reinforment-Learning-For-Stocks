"""Reinforcement-learning transitions built from the prepared datasets.

Functional replacement of the notebook's ``CVaRDataset`` class: every sample is
a ``(state, action, reward, next_state, done)`` transition where

- ``state``      = features of all tickers over ``sequence_length`` days,
                   shape ``(sequence_length, n_indicators * n_tickers)``
- ``action``     = one label per ticker for the day after the window
- ``reward``     = one reward per ticker
- ``next_state`` = the same window shifted by one day
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from rl_stocks.utils.indicators import BUY, HOLD, SELL


def compute_cvar_reward(
    daily_actions: np.ndarray,
    daily_cvar: np.ndarray,
    buy_weight_on_loss: float = 1.5,
    sell_weight_on_gain: float = 1.2,
    hold_weight: float = 0.8,
    max_abs_reward: float = 3.0,
) -> np.ndarray:
    """Reward of each ticker's action given that day's CVaR.

    - BUY  -> penalty ``-weight * |CVaR|`` (heavier when CVaR < 0)
    - SELL -> bonus   ``+weight * |CVaR|`` (heavier when CVaR > 0)
    - HOLD -> small penalty ``-hold_weight * |CVaR|``

    Pass the *raw* CVaR: after min-max normalization it is never negative and
    the ``CVaR < 0`` / ``CVaR > 0`` weights lose their meaning.
    """
    cvar_magnitude = np.abs(daily_cvar)
    buy_weight = np.where(daily_cvar < 0, buy_weight_on_loss, 1.0)
    sell_weight = np.where(daily_cvar > 0, sell_weight_on_gain, 1.0)

    reward = -buy_weight * cvar_magnitude * (daily_actions == BUY)
    reward += sell_weight * cvar_magnitude * (daily_actions == SELL)
    reward += -hold_weight * cvar_magnitude * (daily_actions == HOLD)
    return np.clip(reward, -max_abs_reward, max_abs_reward)


def normalize_min_max(matrix: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Scale every value of ``matrix`` into [0, 1] using its global min / max."""
    return (matrix - matrix.min()) / (matrix.max() - matrix.min() + epsilon)


def pivot_to_daily_matrices(
    feature_df: pd.DataFrame,
    action_labels_df: pd.DataFrame,
    aligned_cvar_df: pd.DataFrame,
) -> dict:
    """Turn the long DataFrames into aligned (dates x ...) numpy matrices.

    Dates where at least one ticker has no features are dropped, so every row
    of the three matrices refers to the same trading day.
    """
    indicator_columns = [c for c in feature_df.columns if c not in ("Date", "Ticker")]

    features_by_date = (
        feature_df
        .pivot(index="Date", columns="Ticker", values=indicator_columns)
        .sort_index()
        .dropna()
    )
    trading_dates = features_by_date.index

    actions_by_date = (
        action_labels_df
        .pivot(index="Date", columns="Ticker", values="Action")
        .reindex(trading_dates)
    )
    cvar_by_date = (
        aligned_cvar_df
        .pivot(index="Date", columns="Ticker", values="CVaR")
        .reindex(index=trading_dates, columns=actions_by_date.columns)
    )

    return {
        "trading_dates": trading_dates,
        "tickers": list(actions_by_date.columns),
        "feature_matrix": features_by_date.to_numpy(dtype=np.float32),
        "action_matrix": actions_by_date.fillna(HOLD).to_numpy(dtype=np.float32),
        "cvar_matrix": cvar_by_date.fillna(0).to_numpy(dtype=np.float32),
    }


def build_transitions(
    feature_df: pd.DataFrame,
    action_labels_df: pd.DataFrame,
    aligned_cvar_df: pd.DataFrame,
    sequence_length: int = 10,
    normalize_cvar_for_reward: bool = False,
) -> dict:
    """Build every ``(state, action, reward, next_state, done)`` transition.

    :param normalize_cvar_for_reward: reproduce the notebook behaviour
        (min-max normalized CVaR in the reward). Default False: raw CVaR.
    :return: dict of numpy arrays ``states``, ``actions``, ``rewards``,
        ``next_states``, ``dones`` plus ``dates`` and ``tickers``.
    """
    daily = pivot_to_daily_matrices(feature_df, action_labels_df, aligned_cvar_df)
    feature_matrix = daily["feature_matrix"]
    action_matrix = daily["action_matrix"]
    cvar_matrix = daily["cvar_matrix"]
    if normalize_cvar_for_reward:
        cvar_matrix = normalize_min_max(cvar_matrix)

    states, actions, rewards, next_states, dates = [], [], [], [], []

    for current_day in range(sequence_length, len(feature_matrix) - 1):
        window_start = current_day - sequence_length
        daily_actions = action_matrix[current_day]
        reward = compute_cvar_reward(daily_actions, cvar_matrix[current_day])
        if np.any(np.isnan(reward)):
            continue

        states.append(feature_matrix[window_start:current_day])
        next_states.append(feature_matrix[window_start + 1:current_day + 1])
        actions.append(daily_actions)
        rewards.append(reward)
        dates.append(daily["trading_dates"][current_day])

    n_tickers = len(daily["tickers"])
    n_transitions = len(states)
    dones = np.zeros((n_transitions, n_tickers), dtype=np.float32)
    if n_transitions > 0:
        dones[-1] = 1.0

    n_features = feature_matrix.shape[1]
    return {
        "states": np.asarray(states, dtype=np.float32).reshape(-1, sequence_length, n_features),
        "actions": np.asarray(actions, dtype=np.float32).reshape(-1, n_tickers),
        "rewards": np.asarray(rewards, dtype=np.float32).reshape(-1, n_tickers),
        "next_states": np.asarray(next_states, dtype=np.float32).reshape(-1, sequence_length, n_features),
        "dones": dones,
        "dates": pd.DatetimeIndex(dates),
        "tickers": daily["tickers"],
    }


def transitions_to_tensor_dataset(transitions: dict) -> TensorDataset:
    """Wrap the transitions in a ``TensorDataset`` usable by a ``DataLoader``.

    Each item is ``(state, action, reward, next_state, done)``.
    """
    return TensorDataset(
        torch.from_numpy(transitions["states"]),
        torch.from_numpy(transitions["actions"]),
        torch.from_numpy(transitions["rewards"]),
        torch.from_numpy(transitions["next_states"]),
        torch.from_numpy(transitions["dones"]),
    )
