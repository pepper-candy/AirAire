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
OBS_CLIP = 10.0
REWARD_CLIP = 10.0
MAX_PRICE = 1.0e6
MAX_HOLDING = 1.0e6
MAX_EQUITY = 1.0e12
MIN_EQUITY = 1.0e-6


def _safe_float(value: object, default: float = 0.0) -> float:
    """NaN is truthy in Python, so ``x or default`` will not catch it."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(number) else number


def _finite_array(values: np.ndarray, fill: float = 0.0) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=fill, posinf=fill, neginf=fill)


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


def news_obs_slice(lookback_bars: int = LOOKBACK_BARS) -> slice:
    """Slice of the concatenated observation that holds the 5 news scores."""
    start = lookback_bars * N_CORE * len(OHLCV_FIELDS) + _long_term_dim() + 5
    return slice(start, start + N_CORE)


class TradingEnv(gym.Env):
    """FinRL-style continuous-action env. One ``step()`` = one bar inside a 30-day window."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        news_df: pd.DataFrame | None = None,
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
        self._news_live = False
        if news_scores:
            self._news_scores.update(news_scores)
            self._news_live = True

        self.df = self._prepare_panel(df)
        self.datetimes = self._unique_datetimes()
        self._close_matrix = self._build_close_matrix()
        # Precompute OHLCV + long-term blocks once. Per-step pandas (isin / pivot_table)
        # was the 30 FPS bottleneck — the GPU was idle waiting on these.
        self._ohlcv_cube = self._build_ohlcv_cube()
        self._long_term_mat = self._precompute_long_term()
        self._news_aligned = self._prepare_news(news_df)
        self._bar_index = 0
        self._cash = self.initial_cash
        self._holdings = np.zeros(N_CORE, dtype=np.float64)  # shares
        self._returns: list[float] = []
        self._equity_curve: list[float] = [self.initial_cash]
        self._last_equity = self.initial_cash
        self._last_good_prices = np.ones(N_CORE, dtype=np.float64)
        if len(self._close_matrix):
            seed_px = self._close_matrix[min(self.lookback_bars, len(self._close_matrix) - 1)]
            seed_px = np.where(np.isfinite(seed_px) & (seed_px > 0), seed_px, 1.0)
            self._last_good_prices = seed_px.astype(np.float64)

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
        for col in ("open", "high", "low", "close", "volume"):
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
        panel["volume"] = panel["volume"].fillna(0.0)
        panel = panel.sort_values(["ticker", "datetime"])
        ohlc = ["open", "high", "low", "close"]
        panel[ohlc] = panel.groupby("ticker", group_keys=False)[ohlc].ffill()
        panel[ohlc] = panel[ohlc].fillna(0.0)
        if "news_score" in panel.columns:
            panel["news_score"] = pd.to_numeric(panel["news_score"], errors="coerce")
            panel["news_score"] = panel.groupby("ticker", group_keys=False)["news_score"].ffill().fillna(0.0)
        if "sentiment_score" in panel.columns:
            panel["sentiment_score"] = pd.to_numeric(panel["sentiment_score"], errors="coerce")
            panel["sentiment_score"] = panel.groupby("ticker", group_keys=False)["sentiment_score"].ffill().fillna(0.0)
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

    def _build_close_matrix(self) -> np.ndarray:
        """datetime × CORE_TICKERS close matrix — O(1) price lookup per step."""
        n = len(self.datetimes)
        mat = np.ones((n, N_CORE), dtype=np.float64)
        if n == 0 or self.df.empty:
            return mat
        wide = self.df.pivot_table(index="datetime", columns="ticker", values="close", aggfunc="last")
        wide = wide.reindex(self.datetimes)
        for i, ticker in enumerate(CORE_TICKERS):
            if ticker in wide.columns:
                col = pd.to_numeric(wide[ticker], errors="coerce").ffill().bfill()
                vals = col.to_numpy(dtype=np.float64)
                vals = np.where(np.isfinite(vals) & (vals > 0), vals, np.nan)
                # leftover NaNs stay 1.0 so leverage math doesn't explode
                finite = np.isfinite(vals)
                if finite.any():
                    mat[finite, i] = vals[finite]
        return mat

    def _build_ohlcv_cube(self) -> np.ndarray:
        """datetime × ticker × OHLCV — sliced in ``_price_window_features`` with no pandas."""
        n = len(self.datetimes)
        cube = np.zeros((max(n, 1), N_CORE, len(OHLCV_FIELDS)), dtype=np.float64)
        if n == 0 or self.df.empty:
            return cube
        cube[:, :, 3] = self._close_matrix
        for f_i, field in enumerate(OHLCV_FIELDS):
            if field == "close":
                continue
            wide = self.df.pivot_table(index="datetime", columns="ticker", values=field, aggfunc="last")
            wide = wide.reindex(self.datetimes)
            for j, ticker in enumerate(CORE_TICKERS):
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
                    # missing OHLC: copy close so normalize does not divide by 0
                    missing = ~finite
                    if missing.any():
                        cube[missing, j, f_i] = self._close_matrix[missing, j]
        # any leftover zero OHLC → close
        for f_i in range(4):
            bad = ~(np.isfinite(cube[:, :, f_i]) & (cube[:, :, f_i] > 0))
            cube[:, :, f_i] = np.where(bad, self._close_matrix, cube[:, :, f_i])
        return cube

    def _precompute_long_term(self) -> np.ndarray:
        """MA distance, vol percentile, HK↔US corr for every bar (once per env)."""
        n = len(self.datetimes)
        if n == 0:
            return np.zeros((1, _long_term_dim()), dtype=np.float32)
        closes = pd.DataFrame(self._close_matrix, columns=CORE_TICKERS)
        ma_block = np.zeros((n, N_CORE), dtype=np.float32)
        vol_block = np.full((n, N_CORE), 0.5, dtype=np.float32)
        for j, ticker in enumerate(CORE_TICKERS):
            series = closes[ticker]
            ma = series.rolling(200, min_periods=2).mean()
            dist = (series - ma) / ma.replace(0.0, np.nan)
            ma_block[:, j] = dist.fillna(0.0).to_numpy(dtype=np.float32)

            rets = series.pct_change().replace([np.inf, -np.inf], np.nan)
            roll = rets.rolling(30).std()
            valid = roll.to_numpy(dtype=np.float64)
            for i in range(n):
                v = valid[i]
                if not np.isfinite(v):
                    continue
                hist = valid[: i + 1]
                hist = hist[np.isfinite(hist)]
                if len(hist) == 0:
                    continue
                vol_block[i, j] = float(np.mean(hist <= v))

        corr_block = np.zeros((n, 6), dtype=np.float32)
        k = 0
        for h in CORE_TICKERS[:3]:
            for u in CORE_TICKERS[3:]:
                rh = closes[h].pct_change().replace([np.inf, -np.inf], np.nan)
                ru = closes[u].pct_change().replace([np.inf, -np.inf], np.nan)
                corr = rh.expanding(min_periods=10).corr(ru)
                corr_block[:, k] = np.nan_to_num(corr.to_numpy(dtype=np.float64), nan=0.0).astype(np.float32)
                k += 1
        return np.concatenate([ma_block, vol_block, corr_block], axis=1).astype(np.float32)

    def _prepare_news(self, news_df: pd.DataFrame | None) -> np.ndarray:
        """Align sentiment onto ``self.datetimes`` (forward-fill, no NaNs).

        Preference: explicit ``news_df`` → ``news_score`` / ``sentiment_score``
        column on the price panel → zeros. Shape ``(n_bars, N_CORE)``.
        """
        n = len(self.datetimes)
        aligned = np.zeros((max(n, 1), N_CORE), dtype=np.float32)
        source = None
        if news_df is not None and not news_df.empty:
            source = news_df.copy()
        elif not self.df.empty:
            for col in ("sentiment_score", "news_score"):
                if col in self.df.columns:
                    source = self.df[["datetime", "ticker", col]].rename(columns={col: "sentiment_score"})
                    break
        if source is None or source.empty or n == 0:
            return aligned

        if "sentiment_score" not in source.columns:
            if "news_score" in source.columns:
                source = source.rename(columns={"news_score": "sentiment_score"})
            else:
                logger.warning("News frame missing sentiment_score; block (4) stays 0.")
                return aligned

        source = source.copy()
        source["datetime"] = pd.to_datetime(source["datetime"], errors="coerce")
        if getattr(source["datetime"].dt, "tz", None) is not None:
            source["datetime"] = source["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
        source["sentiment_score"] = pd.to_numeric(source["sentiment_score"], errors="coerce").clip(-1.0, 1.0)
        source = source.dropna(subset=["datetime", "ticker", "sentiment_score"])
        if source.empty:
            return aligned

        wide = source.pivot_table(index="datetime", columns="ticker", values="sentiment_score", aggfunc="last")
        union_index = wide.index.union(self.datetimes).sort_values()
        wide = wide.reindex(union_index).sort_index().ffill()
        wide = wide.reindex(self.datetimes)
        for i, ticker in enumerate(CORE_TICKERS):
            if ticker in wide.columns:
                col = wide[ticker].fillna(0.0).clip(-1.0, 1.0).to_numpy(dtype=np.float32)
                aligned[: len(col), i] = col
        nonzero = float(np.mean(np.abs(aligned) > 1e-12)) if aligned.size else 0.0
        if nonzero < 1e-12:
            logger.warning("News block is all zeros after alignment. Check data/raw/news/ or Alpha Vantage coverage.")
        else:
            logger.info(
                "News block aligned: bars=%d coverage=%.1f%% sample=%s",
                n,
                100.0 * nonzero,
                np.round(aligned[min(self.lookback_bars, max(n - 1, 0))], 3).tolist(),
            )
        return aligned

    def set_news_scores(self, scores: dict[str, float]) -> None:
        for ticker, value in scores.items():
            if ticker in self._news_scores:
                self._news_scores[ticker] = float(np.clip(value, -1.0, 1.0))
        self._news_live = True

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
        # Re-seed from the start bar so a previous episode's prices cannot leak in.
        if len(self._close_matrix):
            seed_px = self._close_matrix[min(self._bar_index, len(self._close_matrix) - 1)]
            seed_px = np.where(np.isfinite(seed_px) & (seed_px > 0), seed_px, 1.0)
            self._last_good_prices = seed_px.astype(np.float64)
        else:
            self._last_good_prices = np.ones(N_CORE, dtype=np.float64)
        obs = self._get_obs()
        return obs, {"bar_index": self._bar_index, "datetime": self._current_dt()}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != N_CORE:
            raise ValueError(f"Expected action of shape ({N_CORE},), got {action.shape}")
        action = _finite_array(action)
        action = np.clip(action, -1.0, 1.0)
        # Leverage cap: sum(|weights|) <= 2x
        abs_sum = np.abs(action).sum()
        if abs_sum > MAX_LEVERAGE:
            action = action * (MAX_LEVERAGE / abs_sum)

        # Mark at the *current* bar, rebalance, then advance and mark at the
        # *next* bar. Same-price MTM after a commission-free rebalance made
        # every step_return ≈ 0 (2026-08-19 GPU run: Sharpe always 0.0000),
        # so PPO only saw the drawdown penalty. Equity-curve P&L was already
        # valid because the next step opened at the new prices; _returns was not.
        prices_current = self._current_closes()
        prev_equity = _safe_float(self._mark_to_market(prices_current), self._last_equity)
        self._rebalance(action, prices_current)

        self._bar_index += 1
        terminated = self._bar_index >= len(self.datetimes) - 1

        prices_next = self._current_closes()
        equity = _safe_float(self._mark_to_market(prices_next), prev_equity)
        step_return = (equity - prev_equity) / max(abs(prev_equity), 1e-9)
        step_return = _safe_float(step_return, 0.0)
        self._returns.append(step_return)
        self._equity_curve.append(equity)
        self._last_equity = equity

        reward = self._sharpe_drawdown_reward()
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
        return np.clip(
            np.nan_to_num(
                np.concatenate([price_window, long_term, calendar, news_scores, inventory]).astype(np.float32),
                nan=0.0,
                posinf=OBS_CLIP,
                neginf=-OBS_CLIP,
            ),
            -OBS_CLIP,
            OBS_CLIP,
        ).astype(np.float32)

    def _price_window_features(self) -> np.ndarray:
        """(1) Last ``lookback_bars`` of OHLCV for the 5 core stocks, flattened."""
        lookback = self.lookback_bars
        end = min(self._bar_index + 1, len(self._ohlcv_cube))
        start = end - lookback
        if end <= 0:
            window = np.zeros((lookback, N_CORE, len(OHLCV_FIELDS)), dtype=np.float64)
        elif start < 0:
            available = self._ohlcv_cube[:end]
            pad = np.repeat(available[:1], lookback - end, axis=0)
            window = np.concatenate([pad, available], axis=0)
        else:
            window = self._ohlcv_cube[start:end]
        window = _finite_array(window)
        last_close = window[-1, :, 3]
        last_close = np.where(last_close > 0, last_close, 1.0)
        ohlc = window[:, :, :4] / last_close.reshape(1, N_CORE, 1) - 1.0
        vol = window[:, :, 4]
        vol_den = np.nanmean(np.abs(vol), axis=0)
        vol_den = np.where(np.isfinite(vol_den) & (vol_den > 0), vol_den, 1.0)
        vol_norm = vol / vol_den.reshape(1, N_CORE)
        feat = np.concatenate([ohlc, vol_norm[:, :, None]], axis=2)
        # ticker-major flatten (same order as the old per-ticker loop)
        return np.transpose(feat, (1, 0, 2)).reshape(-1).astype(np.float32)

    def _long_term_features(self) -> np.ndarray:
        """(2) MA distance, vol percentile, HK-tech vs US-defensive corr (precomputed)."""
        if self._long_term_mat is None or len(self._long_term_mat) == 0:
            return np.zeros(_long_term_dim(), dtype=np.float32)
        i = min(self._bar_index, len(self._long_term_mat) - 1)
        return self._long_term_mat[i].astype(np.float32, copy=False)

    def _calendar_features(self) -> np.ndarray:
        """(3) Day-of-week, month, days until major holidays."""
        dt = self._current_dt()
        ref = dt.date() if hasattr(dt, "date") else datetime.now(timezone.utc).date()
        return np.asarray(calendar_feature_vector(ref), dtype=np.float32)

    def _news_score_features(self) -> np.ndarray:
        """(4) Intraday sentiment in [-1, 1] for each core ticker.

        Live inference (``set_news_scores``) wins over the historical matrix so
        paper trading always sees the latest Alpha Vantage poll.
        """
        if self._news_live:
            return np.asarray([self._news_scores.get(t, 0.0) for t in CORE_TICKERS], dtype=np.float32)
        if self._news_aligned is not None and len(self._news_aligned):
            i = min(self._bar_index, len(self._news_aligned) - 1)
            return np.clip(self._news_aligned[i], -1.0, 1.0).astype(np.float32)
        return np.asarray([self._news_scores.get(t, 0.0) for t in CORE_TICKERS], dtype=np.float32)

    def _inventory_features(self) -> np.ndarray:
        """(5) Holdings as portfolio-weight ratios plus cash fraction."""
        prices = self._current_closes()
        equity = float(np.clip(_safe_float(self._mark_to_market(prices), self.initial_cash), MIN_EQUITY, MAX_EQUITY))
        holdings = np.clip(_finite_array(self._holdings), -MAX_HOLDING, MAX_HOLDING)
        notionals = holdings * prices
        weights = _finite_array(notionals / equity)
        cash_frac = float(np.clip(_safe_float(self._cash / equity, 1.0), -MAX_LEVERAGE, MAX_LEVERAGE))
        return np.concatenate([np.clip(weights, -MAX_LEVERAGE, MAX_LEVERAGE), np.asarray([cash_frac])]).astype(np.float32)

    # ------------------------------------------------------------------
    # Portfolio mechanics
    # ------------------------------------------------------------------
    def _current_dt(self):
        if len(self.datetimes) == 0:
            return pd.Timestamp.utcnow()
        i = min(self._bar_index, len(self.datetimes) - 1)
        return self.datetimes[i]

    def _current_closes(self) -> np.ndarray:
        if self._close_matrix is None or len(self._close_matrix) == 0:
            return np.array(self._last_good_prices, dtype=np.float64, copy=True)
        i = min(self._bar_index, len(self._close_matrix) - 1)
        raw = self._close_matrix[i]
        prices = np.where(np.isfinite(raw) & (raw > 0), raw, self._last_good_prices)
        prices = np.clip(_finite_array(prices, fill=1.0), 0.01, MAX_PRICE)
        self._last_good_prices = prices
        return prices

    def _mark_to_market(self, prices: np.ndarray) -> float:
        prices = np.clip(_finite_array(prices), 0.01, MAX_PRICE)
        holdings = np.clip(_finite_array(self._holdings), -MAX_HOLDING, MAX_HOLDING)
        cash = float(np.clip(_safe_float(self._cash, 0.0), -MAX_EQUITY, MAX_EQUITY))
        value = cash + float(np.dot(holdings, prices))
        if not np.isfinite(value) or abs(value) > MAX_EQUITY:
            return float(np.clip(self._last_equity, MIN_EQUITY, MAX_EQUITY))
        return _safe_float(value, self._last_equity)

    def _rebalance(self, target_weights: np.ndarray, prices: np.ndarray) -> None:
        prices = np.clip(_finite_array(prices), 0.01, MAX_PRICE)
        equity = float(np.clip(self._mark_to_market(prices), MIN_EQUITY, MAX_EQUITY))
        target_notional = _finite_array(target_weights) * equity
        safe_prices = np.where(prices > 0, prices, np.inf)
        target_shares = target_notional / safe_prices
        target_shares = np.where(np.isfinite(target_shares), target_shares, 0.0)
        target_shares = np.clip(target_shares, -MAX_HOLDING, MAX_HOLDING)
        delta = target_shares - self._holdings
        trade_cash = _safe_float(np.dot(delta, prices), 0.0)
        self._cash = float(np.clip(_safe_float(self._cash - trade_cash, 0.0), -MAX_EQUITY, MAX_EQUITY))
        self._holdings = target_shares
        # Negative cash is margin (allowed up to MAX_LEVERAGE). Zeroing it used to
        # wipe the liability while keeping the shares — equity then doubled every
        # max-leverage bar and evaluation reported billion-dollar books.
        if not np.isfinite(self._cash):
            logger.debug("Cash non-finite after rebalance; restoring previous mark.")
            self._cash = float(np.clip(self._last_equity - float(np.dot(self._holdings, prices)), -MAX_EQUITY, MAX_EQUITY))

    def _sharpe_drawdown_reward(self) -> float:
        """Reward = rolling Sharpe − λ × max-drawdown penalty (PLAN.md §5)."""
        rets = _finite_array(np.asarray(self._returns[-SHARPE_WINDOW :], dtype=np.float64))
        rets = np.clip(rets, -1.0, 1.0)
        if len(rets) < 2:
            sharpe = 0.0
        else:
            std = _safe_float(rets.std(), 0.0)
            sharpe = _safe_float(rets.mean() / (std + 1e-9) * np.sqrt(TRADING_DAYS), 0.0)

        curve = np.clip(
            _finite_array(np.asarray(self._equity_curve, dtype=np.float64), fill=self.initial_cash),
            MIN_EQUITY,
            MAX_EQUITY,
        )
        if len(curve) < 2:
            return 0.0
        peak = np.maximum.accumulate(curve)
        denominator = np.maximum(np.abs(peak), 1e-6)
        dd = (peak - curve) / denominator
        dd = dd[np.isfinite(dd)]
        max_dd = float(np.clip(dd.max(), 0.0, 1.0)) if len(dd) else 0.0
        reward = sharpe - DRAWDOWN_LAMBDA * max_dd
        return float(np.clip(_safe_float(reward, 0.0), -REWARD_CLIP, REWARD_CLIP))


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
