"""Transitions for DQN / DDQN agents, built from the prepared datasets.

Scope: this module is NOT a generic RL toolkit. In RL the data format depends
on the problem, the environment and the algorithm; everything here is designed
for **value-based, off-policy agents (DQN and Double DQN)** under these
assumptions:

- **Environment**: historical and offline. There is no simulator; the agent
  does not choose the actions stored in the transitions. Actions are the
  CVaR labels (10-day vs 30-day CVaR), so the agent learns Q-values for
  pre-computed decisions, like a replay buffer filled in advance.
- **Action space**: discrete, one action per ticker in {0 = Hold, 1 = Buy,
  2 = Sell}. A Q-network therefore outputs ``n_tickers x 3`` values.
- **Reward**: per ticker, derived from that day's CVaR (``compute_cvar_reward``).
- **State**: features of all tickers over a fixed window of past days.
- **Episode**: the whole history is one episode; ``done`` is 1 only on the
  last transition (used in the target ``r + gamma * (1 - done) * max Q(s', .)``).

Policy-gradient or actor-critic methods (PPO, A2C, SAC, ...) need an
interactive environment and on-policy rollouts, and would require a different
pipeline (e.g. a gym-style environment in ``rl_stocks.environments``).

The functions build the transitions as numpy arrays; ``CVaRDataset`` wraps
them in a PyTorch ``Dataset``. Every sample is a
``(state, action, reward, next_state, done)`` transition where

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
from torch.utils.data import Dataset

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


class CVaRDataset(Dataset):
    """Offline replay data for DQN / DDQN training, ready for a ``DataLoader``.

    Plays the role of a pre-filled replay buffer: sampling with
    ``DataLoader(dataset, shuffle=True)`` gives the decorrelated mini-batches
    DQN / DDQN expect. See the module docstring for the environment
    assumptions (offline, discrete actions per ticker, CVaR reward).

    Each item is a tuple of float32 tensors::

        state       (sequence_length, n_indicators * n_tickers)
        action      (n_tickers,)   0 = Hold, 1 = Buy, 2 = Sell
        reward      (n_tickers,)
        next_state  (sequence_length, n_indicators * n_tickers)
        done        (n_tickers,)   1.0 only on the last transition

    :param feature_df: long DataFrame ``Date, Ticker`` + scaled indicator columns.
    :param action_labels_df: long DataFrame ``Date, Ticker, Action``.
    :param aligned_cvar_df: long DataFrame ``Date, Ticker, CVaR``.
    :param sequence_length: number of past days in each state.
    :param normalize_cvar_for_reward: use min-max normalized CVaR in the
        reward (old notebook behaviour) instead of the raw CVaR.
    """

    def __init__(
        self,
        feature_df: pd.DataFrame,
        action_labels_df: pd.DataFrame,
        aligned_cvar_df: pd.DataFrame,
        sequence_length: int = 10,
        normalize_cvar_for_reward: bool = False,
    ):
        self.sequence_length = sequence_length

        rl_transitions = build_transitions(
            feature_df,
            action_labels_df,
            aligned_cvar_df,
            sequence_length=sequence_length,
            normalize_cvar_for_reward=normalize_cvar_for_reward,
        )

        self.tickers = rl_transitions["tickers"]
        self.n_tickers = len(self.tickers)
        self.dates = rl_transitions["dates"]

        self.states = torch.from_numpy(rl_transitions["states"])
        self.actions = torch.from_numpy(rl_transitions["actions"])
        self.rewards = torch.from_numpy(rl_transitions["rewards"])
        self.next_states = torch.from_numpy(rl_transitions["next_states"])
        self.dones = torch.from_numpy(rl_transitions["dones"])

    @property
    def state_shape(self) -> tuple[int, int]:
        """Shape of one state: ``(sequence_length, n_indicators * n_tickers)``."""
        return tuple(self.states.shape[1:])

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )
