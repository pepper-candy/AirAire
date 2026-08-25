"""V2.10 Gym env — V2 mechanics, long-only, 6-name Bloomberg lookback.

Price window is 5 core + HK.HSI (no SPX). Actions are target weights in
``[0, 1]`` so the policy never shorts (Futu HK has no short solution).
Volume is expected 0 (Bloomberg / V2.10 panel).

Obs dim is 932, not V2's 782 or V3's 1082. Do not load those zips here.
Do not import this from ``inference.py``. Live V2 stays on ``[-1, 1]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from gymnasium import spaces

from src.trading_env import (
    LOOKBACK_BARS,
    N_CORE,
    OHLCV_FIELDS,
    TradingEnv as _TradingEnvV2,
    _finite_array,
    _long_term_dim,
)
from src.utils import CORE_TICKERS, setup_logging

logger = setup_logging("airaire.trading_env_v2_10")

OBSERVER_TICKERS_V2_10 = ["HK.HSI"]
OBS_TICKERS = list(CORE_TICKERS) + list(OBSERVER_TICKERS_V2_10)
N_OBS = len(OBS_TICKERS)

__all__ = [
    "LOOKBACK_BARS",
    "N_OBS",
    "OBS_TICKERS",
    "OBSERVER_TICKERS_V2_10",
    "TradingEnv",
    "news_obs_slice",
    "observation_dim",
]


def observation_dim(lookback_bars: int = LOOKBACK_BARS) -> int:
    return (
        lookback_bars * N_OBS * len(OHLCV_FIELDS)
        + _long_term_dim()
        + 5
        + N_CORE
        + N_CORE
        + 1
    )


def news_obs_slice(lookback_bars: int = LOOKBACK_BARS) -> slice:
    start = lookback_bars * N_OBS * len(OHLCV_FIELDS) + _long_term_dim() + 5
    return slice(start, start + N_CORE)


class TradingEnv(_TradingEnvV2):
    """V2 step/inventory; 6-name OHLCV lookback; long-only 5-name action."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ohlcv_cube = self._build_ohlcv_cube()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim(self.lookback_bars),),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(N_CORE,), dtype=np.float32)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        action = np.clip(_finite_array(action), 0.0, 1.0)
        return super().step(action)

    def _rebalance(self, target_weights: np.ndarray, prices: np.ndarray) -> None:
        weights = np.clip(_finite_array(target_weights), 0.0, 1.0)
        super()._rebalance(weights, prices)
        self._holdings = np.maximum(self._holdings, 0.0)

    def _build_ohlcv_cube(self) -> np.ndarray:
        n = len(self.datetimes)
        cube = np.zeros((max(n, 1), N_OBS, len(OHLCV_FIELDS)), dtype=np.float64)
        if n == 0 or self.df.empty:
            return cube
        wide_close = self.df.pivot_table(index="datetime", columns="ticker", values="close", aggfunc="last")
        wide_close = wide_close.reindex(self.datetimes)
        close_mat = np.ones((n, N_OBS), dtype=np.float64)
        for j, ticker in enumerate(OBS_TICKERS):
            if ticker not in wide_close.columns:
                continue
            col = pd.to_numeric(wide_close[ticker], errors="coerce").ffill().bfill()
            vals = col.to_numpy(dtype=np.float64)
            vals = np.where(np.isfinite(vals) & (vals > 0), vals, np.nan)
            finite = np.isfinite(vals)
            if finite.any():
                close_mat[finite, j] = vals[finite]
        cube[:, :, 3] = close_mat
        for f_i, field in enumerate(OHLCV_FIELDS):
            if field == "close":
                continue
            wide = self.df.pivot_table(index="datetime", columns="ticker", values=field, aggfunc="last")
            wide = wide.reindex(self.datetimes)
            for j, ticker in enumerate(OBS_TICKERS):
                if ticker not in wide.columns:
                    continue
                col = pd.to_numeric(wide[ticker], errors="coerce").ffill().bfill()
                vals = col.to_numpy(dtype=np.float64)
                if field == "volume":
                    cube[:, j, f_i] = np.where(np.isfinite(vals), vals, 0.0)
                else:
                    vals = np.where(np.isfinite(vals) & (vals > 0), vals, np.nan)
                    finite = np.isfinite(vals)
                    if finite.any():
                        cube[finite, j, f_i] = vals[finite]
                    missing = ~finite
                    if missing.any():
                        cube[missing, j, f_i] = close_mat[missing, j]
        for f_i in range(4):
            bad = ~(np.isfinite(cube[:, :, f_i]) & (cube[:, :, f_i] > 0))
            cube[:, :, f_i] = np.where(bad, close_mat, cube[:, :, f_i])
        return cube

    def _price_window_features(self) -> np.ndarray:
        lookback = self.lookback_bars
        end = min(self._bar_index + 1, len(self._ohlcv_cube))
        start = end - lookback
        if end <= 0:
            window = np.zeros((lookback, N_OBS, len(OHLCV_FIELDS)), dtype=np.float64)
        elif start < 0:
            available = self._ohlcv_cube[:end]
            pad = np.repeat(available[:1], lookback - end, axis=0)
            window = np.concatenate([pad, available], axis=0)
        else:
            window = self._ohlcv_cube[start:end]
        window = _finite_array(window)
        last_close = window[-1, :, 3]
        last_close = np.where(last_close > 0, last_close, 1.0)
        ohlc = window[:, :, :4] / last_close.reshape(1, N_OBS, 1) - 1.0
        vol = window[:, :, 4]
        vol_den = np.nanmean(np.abs(vol), axis=0)
        vol_den = np.where(np.isfinite(vol_den) & (vol_den > 0), vol_den, 1.0)
        vol_norm = vol / vol_den.reshape(1, N_OBS)
        feat = np.concatenate([ohlc, vol_norm[:, :, None]], axis=2)
        return np.transpose(feat, (1, 0, 2)).reshape(-1).astype(np.float32)


if __name__ == "__main__":
    env = TradingEnv()
    obs, info = env.reset()
    print(
        f"v2.10 obs_dim={obs.shape} expected={observation_dim(env.lookback_bars)} "
        f"action={env.action_space.low[0]}..{env.action_space.high[0]} "
        f"shape={env.action_space.shape} n_obs_tickers={N_OBS} info={info}"
    )
