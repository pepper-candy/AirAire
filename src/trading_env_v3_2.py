"""V3.2 Gym environment — long-only twin of V3 / V3.1.

Same 1082-dim observation and ``enhanced_v3.parquet`` as V3.1.
Actions are target weights in ``[0, 1]`` so holdings cannot go short — the
constraint Futu SIMULATE already enforces. Gross exposure still capped at
``MAX_LEVERAGE`` (2x) inside the V2 ``step()``.

Do not import this from ``inference.py``. Live V2 stays on ``[-1, 1]``.
V3 / V3.1 zips are not compatible (same obs dim, different action bounds).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from gymnasium import spaces

from src.trading_env import N_CORE, _finite_array
from src.trading_env_v3 import LOOKBACK_BARS, TradingEnv as _TradingEnvV3, news_obs_slice, observation_dim
from src.utils import setup_logging

logger = setup_logging("airaire.trading_env_v3_2")

__all__ = ["LOOKBACK_BARS", "TradingEnv", "news_obs_slice", "observation_dim"]


class TradingEnv(_TradingEnvV3):
    """V3 observers; long-only 5-name action."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(N_CORE,), dtype=np.float32)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        action = np.clip(_finite_array(action), 0.0, 1.0)
        return super().step(action)

    def _rebalance(self, target_weights: np.ndarray, prices: np.ndarray) -> None:
        weights = np.clip(_finite_array(target_weights), 0.0, 1.0)
        super()._rebalance(weights, prices)
        # Targets are nonnegative; this only kills −1e-15 noise, not a short book.
        self._holdings = np.maximum(self._holdings, 0.0)


if __name__ == "__main__":
    env = TradingEnv()
    obs, info = env.reset()
    print(
        f"v3.2 obs_dim={obs.shape} expected={observation_dim(env.lookback_bars)} "
        f"action={env.action_space.low[0]}..{env.action_space.high[0]} "
        f"shape={env.action_space.shape} info={info}"
    )
