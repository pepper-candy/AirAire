"""V2.11 Gym env — same 782-dim V2 cube, Box[-1,1] head, Futu-honest HK longs.

HK names (00700 / 03690 / 03750) are clipped to ``[0, 1]`` after the policy
action. US (COST / KO) stay ``[-1, 1]``. Holdings on HK are forced ``>= 0``.

Do not change Gym action lows (SB3 will not load the V2 zip). Do not import
this from live ``inference.py``. Live V2 stays on ``src.trading_env``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from gymnasium import spaces

from src.trading_env import (
    LOOKBACK_BARS,
    N_CORE,
    TradingEnv as _TradingEnvV2,
    _finite_array,
    news_obs_slice,
    observation_dim,
)
from src.utils import CORE_TICKERS, setup_logging
from src.v2_11 import clip_hk_holdings, clip_hybrid_action, hk_us_split_ok

logger = setup_logging("airaire.trading_env_v2_11")

hk_us_split_ok()

__all__ = [
    "LOOKBACK_BARS",
    "N_CORE",
    "TradingEnv",
    "news_obs_slice",
    "observation_dim",
]


class TradingEnv(_TradingEnvV2):
    """V2 observation / Box[-1,1] action; HK shorts are no-ops in step/_rebalance."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Keep the saved V2 head. Changing lows to 0 refuses PPO.load of the paper zip.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(N_CORE,), dtype=np.float32)
        expected = observation_dim(self.lookback_bars)
        if int(self.observation_space.shape[0]) != expected:
            raise RuntimeError(
                f"V2.11 obs_dim={self.observation_space.shape[0]} expected {expected} (V2 782-dim family)."
            )

    def restore_portfolio(self, cash: float, holdings: dict[str, float] | None = None) -> None:
        if holdings is not None:
            holdings = {
                t: (max(0.0, float(v)) if str(t).startswith("HK.") else float(v))
                for t, v in holdings.items()
            }
        super().restore_portfolio(cash, holdings)
        self._holdings = clip_hk_holdings(self._holdings)

    def step(self, action: np.ndarray):
        raw = np.asarray(action, dtype=np.float64).reshape(-1)
        clipped = clip_hybrid_action(raw)
        obs, reward, terminated, truncated, info = super().step(clipped)
        info["action_raw"] = dict(zip(CORE_TICKERS, _finite_array(raw).tolist()))
        # ``action`` in info is the executed hybrid (HK >= 0, after leverage cap).
        return obs, reward, terminated, truncated, info

    def _rebalance(self, target_weights: np.ndarray, prices: np.ndarray) -> None:
        weights = clip_hybrid_action(target_weights)
        super()._rebalance(weights, prices)
        self._holdings = clip_hk_holdings(self._holdings)


if __name__ == "__main__":
    env = TradingEnv()
    obs, info = env.reset()
    short_hk = np.array([-0.8, -0.5, -0.2, -0.4, 0.6], dtype=np.float32)
    obs2, reward, terminated, truncated, info2 = env.step(short_hk)
    executed = info2.get("action", {})
    print(
        f"v2.11 obs_dim={obs.shape} expected={observation_dim(env.lookback_bars)} "
        f"action_space={env.action_space.low[0]}..{env.action_space.high[0]} "
        f"raw={short_hk.tolist()} executed={executed} hk_holdings={env._holdings[:3].tolist()}"
    )
