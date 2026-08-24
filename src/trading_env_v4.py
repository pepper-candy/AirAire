"""V4 Gym env — V2 mechanics, 7-name Bloomberg lookback, volume expected 0.

Same 1082-dim layout as V3 (5 core + HSI + SPX OHLCV). Actions stay 5 names.
V3 zips must not be loaded: they trained on TradingView volume, not zeros.
"""

from __future__ import annotations

from src.trading_env_v3 import (  # noqa: F401
    N_OBS,
    OBS_TICKERS,
    LOOKBACK_BARS,
    TradingEnv as _TradingEnvV3,
    news_obs_slice,
    observation_dim,
)
from src.utils import setup_logging

logger = setup_logging("airaire.trading_env_v4")


class TradingEnv(_TradingEnvV3):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        logger.info(
            "V4 env ready obs_dim=%s (Bloomberg 7-name lookback, volume channel should be 0).",
            getattr(self.observation_space, "shape", None),
        )
