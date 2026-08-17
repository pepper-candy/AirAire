"""Gymnasium trading environment — AirAire state space (PLAN.md §4).

``_get_obs()`` concatenates five blocks, in order:

    1. Price Window      — last LOOKBACK_BARS of OHLCV for the 5 core stocks
    2. Long-Term Features — 200MA distance, 2y vol percentile, 90d HK↔US corr
    3. Calendar Features  — dow, month, days-to Christmas / CNY / National Day
    4. News Scores        — per-ticker sentiment in [-1, 1]
    5. Inventory          — holdings ratio per core ticker + cash fraction
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.utils import (
    CORE_TICKERS,
    INITIAL_CASH,
    calendar_feature_vector,
    setup_logging,
)

logger = setup_logging("airaire.trading_env")

LOOKBACK_BARS = 30
WINDOW_DAYS = 30
N_CORE = len(CORE_TICKERS)
OHLCV_FIELDS = ("open", "high", "low", "close", "volume")
MAX_LEVERAGE = 2.0
SHARPE_WINDOW = 20
DRAWDOWN_LAMBDA = 1.0
TRADING_DAYS = 252


def _empty_price_window() -> np.ndarray:
    return np.zeros(LOOKBACK_BARS * N_CORE * len(OHLCV_FIELDS), dtype=np.float32)


def _long_term_dim() -> int:
    # 5 MA distances + 5 vol percentiles + 3 HK × 2 US = 6 correlations
    return N_CORE + N_CORE + 3 * 2


def observation_dim(lookback_bars: int = LOOKBACK_BARS) -> int:
    return (
        lookback_bars * N_CORE * len(OHLCV_FIELDS)  # (1) price window
        + _long_term_dim()  # (2) long-term
        + 5  # (3) calendar
        + N_CORE  # (4) news
        + N_CORE  # (5) inventory holdings
        + 1  # (5) cash
    )


class TradingEnv(gym.Env):
    """FinRL-style continuous-action env. One ``step()`` = one bar inside a 30-day window."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        lookback_bars: int = LOOKBACK_BARS,
        window_days: int = WINDOW_DAYS,
        initial_cash: float = INITIAL_CASH,
        news_scores: dict[str, float] | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.lookback_bars = lookback_bars
        self.window_days = window_days
        self.initial_cash = float(initial_cash)
        self.render_mode = render_mode
        self._news_scores = {t: 0.0 for t in CORE_TICKERS}
        if news_scores:
            self._news_scores.update(news_scores)

        self.df = self._prepare_panel(df)
        self.datetimes = self._unique_datetimes()
        self._bar_index = 0
        self._cash = self.initial_cash
        self._holdings = np.zeros(N_CORE, dtype=np.float64)  # shares
        self._returns: list[float] = []
        self._equity_curve: list[float] = [self.initial_cash]
        self._last_equity = self.initial_cash

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(N_CORE,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim(self.lookback_bars),),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Data prep
    # ------------------------------------------------------------------
    def _prepare_panel(self, df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or df.empty:
            logger.warning("TradingEnv received empty data; using synthetic OHLCV so the Gym interface can be exercised.")
            return _synthetic_panel()
        need = {"datetime", "ticker", "open", "high", "low", "close", "volume"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"TradingEnv df missing columns: {missing}")
        panel = df.copy()
        panel["datetime"] = pd.to_datetime(panel["datetime"])
        return panel.sort_values(["datetime", "ticker"]).reset_index(drop=True)

    def _unique_datetimes(self) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(self.df["datetime"].unique()).sort_values()
        if len(idx) < self.lookback_bars + 2:
            logger.warning(
                "Only %d unique timestamps (need > %d lookback bars). Env will truncate/hold.",
                len(idx),
                self.lookback_bars,
            )
        return idx

    def set_news_scores(self, scores: dict[str, float]) -> None:
        for ticker, value in scores.items():
            if ticker in self._news_scores:
                self._news_scores[ticker] = float(np.clip(value, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        start = self.lookback_bars
        if options and "start_index" in options:
            start = int(options["start_index"])
        self._bar_index = min(max(start, self.lookback_bars), max(len(self.datetimes) - 1, 0))
        self._cash = self.initial_cash
        self._holdings = np.zeros(N_CORE, dtype=np.float64)
        self._returns = []
        self._equity_curve = [self.initial_cash]
        self._last_equity = self.initial_cash
        obs = self._get_obs()
        return obs, {"bar_index": self._bar_index, "datetime": self._current_dt()}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != N_CORE:
            raise ValueError(f"Expected action of shape ({N_CORE},), got {action.shape}")
        action = np.clip(action, -1.0, 1.0)
        # Leverage cap: sum(|weights|) <= 2x
        abs_sum = np.abs(action).sum()
        if abs_sum > MAX_LEVERAGE:
            action = action * (MAX_LEVERAGE / abs_sum)

        prices = self._current_closes()
        prev_equity = self._mark_to_market(prices)
        self._rebalance(action, prices)
        equity = self._mark_to_market(prices)
        step_return = (equity - prev_equity) / max(prev_equity, 1e-9)
        self._returns.append(step_return)
        self._equity_curve.append(equity)
        self._last_equity = equity

        reward = self._sharpe_drawdown_reward()

        self._bar_index += 1
        terminated = self._bar_index >= len(self.datetimes) - 1
        truncated = False
        obs = self._get_obs()
        info = {
            "equity": equity,
            "cash": self._cash,
            "holdings": dict(zip(CORE_TICKERS, self._holdings.tolist())),
            "action": dict(zip(CORE_TICKERS, action.tolist())),
            "datetime": self._current_dt(),
            "reward_sharpe_dd": reward,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        prices = self._current_closes()
        logger.info(
            "t=%s equity=%.2f cash=%.2f holdings=%s",
            self._current_dt(),
            self._mark_to_market(prices),
            self._cash,
            dict(zip(CORE_TICKERS, np.round(self._holdings, 2).tolist())),
        )

    # ------------------------------------------------------------------
    # State space — five concatenated blocks
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        price_window = self._price_window_features()  # (1)
        long_term = self._long_term_features()  # (2)
        calendar = self._calendar_features()  # (3)
        news_scores = self._news_score_features()  # (4)
        inventory = self._inventory_features()  # (5)
        return np.concatenate(
            [price_window, long_term, calendar, news_scores, inventory]
        ).astype(np.float32)

    def _price_window_features(self) -> np.ndarray:
        """(1) Last ``lookback_bars`` of OHLCV for the 5 core stocks, flattened."""
        end = self._bar_index + 1
        start = max(end - self.lookback_bars, 0)
        window_ts = self.datetimes[start:end]
        chunk = self.df[self.df["datetime"].isin(window_ts)]
        blocks: list[np.ndarray] = []
        for ticker in CORE_TICKERS:
            sub = chunk[chunk["ticker"] == ticker].sort_values("datetime")
            values = sub.loc[:, list(OHLCV_FIELDS)].to_numpy(dtype=np.float64)
            if len(values) == 0:
                values = np.zeros((self.lookback_bars, len(OHLCV_FIELDS)))
            elif len(values) < self.lookback_bars:
                pad = np.repeat(values[:1], self.lookback_bars - len(values), axis=0)
                values = np.vstack([pad, values])
            else:
                values = values[-self.lookback_bars :]
            # Normalize volume and prices vs last close to keep PPO inputs scaled
            last_close = values[-1, 3] if values[-1, 3] else 1.0
            values = values.copy()
            values[:, :4] = values[:, :4] / max(last_close, 1e-9) - 1.0
            vol = values[:, 4]
            vol_den = np.nanmean(np.abs(vol)) or 1.0
            values[:, 4] = vol / vol_den
            blocks.append(values.reshape(-1))
        return np.concatenate(blocks).astype(np.float32)

    def _long_term_features(self) -> np.ndarray:
        """(2) MA distance, 2-year vol percentile, 90-day HK-tech vs US-defensive corr."""
        now = self._current_dt()
        hist = self.df[self.df["datetime"] <= now]
        closes = hist.pivot_table(index="datetime", columns="ticker", values="close", aggfunc="last").sort_index()

        ma_dist = []
        vol_pct = []
        for ticker in CORE_TICKERS:
            series = closes[ticker].dropna() if ticker in closes.columns else pd.Series(dtype=float)
            if len(series) < 2:
                ma_dist.append(0.0)
                vol_pct.append(0.5)
                continue
            px = float(series.iloc[-1])
            window_ma = min(len(series), 200)
            ma = float(series.iloc[-window_ma:].mean())
            ma_dist.append((px - ma) / ma if ma else 0.0)

            rets = series.pct_change().dropna()
            if len(rets) < 30:
                vol_pct.append(0.5)
            else:
                vol_30 = float(rets.iloc[-30:].std())
                rolling = rets.rolling(30).std().dropna()
                vol_pct.append(float((rolling <= vol_30).mean()) if len(rolling) else 0.5)

        hk = CORE_TICKERS[:3]
        us = CORE_TICKERS[3:]
        corrs: list[float] = []
        for h in hk:
            for u in us:
                if h in closes.columns and u in closes.columns:
                    pair = closes[[h, u]].dropna().iloc[-90 * 390 :]  # ~90 sessions of 1-min if dense
                    if len(pair) < 30:
                        pair = closes[[h, u]].dropna()
                    if len(pair) >= 10:
                        corrs.append(float(pair[h].pct_change().corr(pair[u].pct_change()) or 0.0))
                    else:
                        corrs.append(0.0)
                else:
                    corrs.append(0.0)

        return np.asarray(ma_dist + vol_pct + corrs, dtype=np.float32)

    def _calendar_features(self) -> np.ndarray:
        """(3) Day-of-week, month, days until major holidays."""
        dt = self._current_dt()
        ref = dt.date() if hasattr(dt, "date") else datetime.now(timezone.utc).date()
        return np.asarray(calendar_feature_vector(ref), dtype=np.float32)

    def _news_score_features(self) -> np.ndarray:
        """(4) Intraday sentiment in [-1, 1] for each core ticker."""
        return np.asarray([self._news_scores.get(t, 0.0) for t in CORE_TICKERS], dtype=np.float32)

    def _inventory_features(self) -> np.ndarray:
        """(5) Holdings as portfolio-weight ratios plus cash fraction."""
        prices = self._current_closes()
        equity = max(self._mark_to_market(prices), 1e-9)
        notionals = self._holdings * prices
        weights = notionals / equity
        cash_frac = self._cash / equity
        return np.concatenate([weights, np.asarray([cash_frac])]).astype(np.float32)

    # ------------------------------------------------------------------
    # Portfolio mechanics
    # ------------------------------------------------------------------
    def _current_dt(self):
        if len(self.datetimes) == 0:
            return pd.Timestamp.utcnow()
        i = min(self._bar_index, len(self.datetimes) - 1)
        return self.datetimes[i]

    def _current_closes(self) -> np.ndarray:
        dt = self._current_dt()
        snap = self.df[self.df["datetime"] == dt]
        prices = []
        for ticker in CORE_TICKERS:
            row = snap[snap["ticker"] == ticker]
            if row.empty:
                hist = self.df[(self.df["ticker"] == ticker) & (self.df["datetime"] <= dt)]
                prices.append(float(hist["close"].iloc[-1]) if not hist.empty else 0.0)
            else:
                prices.append(float(row["close"].iloc[-1]))
        return np.asarray(prices, dtype=np.float64)

    def _mark_to_market(self, prices: np.ndarray) -> float:
        return float(self._cash + np.dot(self._holdings, prices))

    def _rebalance(self, target_weights: np.ndarray, prices: np.ndarray) -> None:
        equity = max(self._mark_to_market(prices), 1e-9)
        target_notional = target_weights * equity
        safe_prices = np.where(prices > 0, prices, np.inf)
        target_shares = target_notional / safe_prices
        target_shares = np.where(np.isfinite(target_shares), target_shares, 0.0)
        delta = target_shares - self._holdings
        trade_cash = float(np.dot(delta, prices))
        self._cash -= trade_cash
        self._holdings = target_shares
        if self._cash < 0:
            # Scale back buys so cash never goes deeply negative in the stub.
            logger.debug("Cash went negative (%.2f); clamping to 0 for this bar.", self._cash)
            self._cash = 0.0

    def _sharpe_drawdown_reward(self) -> float:
        """Reward = rolling Sharpe − λ × max-drawdown penalty (PLAN.md §5)."""
        rets = np.asarray(self._returns[-SHARPE_WINDOW :], dtype=np.float64)
        if len(rets) < 2:
            sharpe = 0.0
        else:
            std = float(rets.std())
            sharpe = float(rets.mean() / (std + 1e-9) * np.sqrt(TRADING_DAYS))
        curve = np.asarray(self._equity_curve, dtype=np.float64)
        peak = np.maximum.accumulate(curve)
        dd = (peak - curve) / np.maximum(peak, 1e-9)
        max_dd = float(dd.max()) if len(dd) else 0.0
        return sharpe - DRAWDOWN_LAMBDA * max_dd


def _synthetic_panel(n_bars: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-02 09:30", periods=n_bars, freq="min")
    rows = []
    for ticker in CORE_TICKERS:
        px = 100.0 + rng.normal(0, 1)
        for dt in dates:
            px = max(px * (1.0 + rng.normal(0, 0.001)), 1.0)
            o = px * (1 + rng.normal(0, 0.0003))
            h = max(o, px) * (1 + abs(rng.normal(0, 0.0004)))
            l = min(o, px) * (1 - abs(rng.normal(0, 0.0004)))
            rows.append(
                {
                    "datetime": dt,
                    "ticker": ticker,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": px,
                    "volume": float(rng.integers(1_000, 50_000)),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    env = TradingEnv()
    obs, info = env.reset()
    print(f"obs_dim={obs.shape} expected={observation_dim(env.lookback_bars)} info={info}")
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    print(f"step reward={reward:.4f} equity={info2['equity']:.2f} terminated={terminated}")
