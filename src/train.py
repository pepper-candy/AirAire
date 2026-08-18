"""Phase 3: sequential 30-day rolling-window PPO training.

Slides a window of N session days forward one day at a time. Does **not**
sample random batches — each window is a contiguous slice of calendar time,
and the same PPO policy is fine-tuned as the window advances.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.data_loader import load_processed, panel_to_wide
from src.trading_env import LOOKBACK_BARS, TRADING_DAYS, TradingEnv
from src.utils import INITIAL_CASH, MODELS_DIR, setup_logging

logger = setup_logging("airaire.train")

DEFAULT_EPOCHS = 10
DEFAULT_WINDOW_DAYS = 30
PPO_N_STEPS = 2048
# 10-minute US cash session ≈ 39 bars/day (Phase 3 spec). Used only to annualize reported Sharpe.
BARS_PER_DAY = 39


@dataclass(frozen=True)
class WindowSlice:
    index: int
    start: pd.Timestamp
    end: pd.Timestamp
    df: pd.DataFrame


@dataclass
class WindowMetrics:
    window: int
    start: str
    end: str
    n_bars: int
    timesteps: int
    cumulative_return: float
    sharpe: float
    max_drawdown: float
    final_equity: float


def _sanitize_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Fill Bloomberg volume NaNs and forward-fill OHLC so PPO never sees NaN obs."""
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["volume"] = out["volume"].fillna(0.0)
    out = out.sort_values(["ticker", "datetime"])
    ohlc = ["open", "high", "low", "close"]
    out[ohlc] = out.groupby("ticker", group_keys=False)[ohlc].ffill()
    out[ohlc] = out[ohlc].fillna(0.0)
    return out.reset_index(drop=True)


def _session_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Unique trading-session dates present in the panel (normalized, sorted)."""
    stamps = pd.to_datetime(df["datetime"])
    return pd.DatetimeIndex(stamps.dt.normalize().unique()).sort_values()


def iter_windows(df: pd.DataFrame, window_days: int) -> list[WindowSlice]:
    """Build contiguous session-day windows, sliding forward by 1 day."""
    dates = _session_dates(df)
    if len(dates) < window_days:
        raise ValueError(
            f"Need at least {window_days} session days to build a window; panel has {len(dates)}."
        )
    session = pd.to_datetime(df["datetime"]).dt.normalize()
    windows: list[WindowSlice] = []
    for i in range(window_days - 1, len(dates)):
        start = dates[i - window_days + 1]
        end = dates[i]
        mask = (session >= start) & (session <= end)
        chunk = df.loc[mask].copy()
        n_ts = chunk["datetime"].nunique()
        if n_ts < LOOKBACK_BARS + 2:
            logger.warning("Skipping %s -> %s: only %d unique bars (need > %d).", start.date(), end.date(), n_ts, LOOKBACK_BARS)
            continue
        windows.append(WindowSlice(index=len(windows) + 1, start=start, end=end, df=chunk))
    if not windows:
        raise ValueError("No valid rolling windows after applying lookback filter.")
    return windows


def make_vec_env(df: pd.DataFrame, window_days: int) -> DummyVecEnv:
    def _factory() -> Monitor:
        env = TradingEnv(df=df, initial_cash=INITIAL_CASH, window_days=window_days)
        return Monitor(env)

    return DummyVecEnv([_factory])


def make_ppo(env: DummyVecEnv, seed: int) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        n_steps=PPO_N_STEPS,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256])},
    )


def timesteps_for_window(df: pd.DataFrame, epochs: int) -> int:
    n_bars = int(pd.to_datetime(df["datetime"]).nunique())
    episode = max(n_bars - LOOKBACK_BARS, 1)
    # At least one PPO rollout so learn() always performs an update.
    return max(epochs * episode, PPO_N_STEPS)


def evaluate_policy(model: PPO, df: pd.DataFrame, window_days: int) -> tuple[float, float, float, float]:
    """Walk the window once, deterministically. Returns return, Sharpe, max DD, final equity."""
    env = TradingEnv(df=df, initial_cash=INITIAL_CASH, window_days=window_days)
    obs, _ = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)

    curve = np.asarray(env._equity_curve, dtype=np.float64)
    start_eq = float(curve[0]) if len(curve) else INITIAL_CASH
    end_eq = float(curve[-1]) if len(curve) else INITIAL_CASH
    cum_ret = (end_eq / start_eq) - 1.0 if start_eq else 0.0

    rets = np.asarray(env._returns, dtype=np.float64)
    if len(rets) < 2 or not np.isfinite(rets.std()) or float(rets.std()) < 1e-12:
        sharpe = 0.0
    else:
        sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(TRADING_DAYS * BARS_PER_DAY))

    peak = np.maximum.accumulate(curve) if len(curve) else np.asarray([start_eq])
    dd = (peak - curve) / np.maximum(peak, 1e-9)
    max_dd = float(dd.max()) if len(dd) else 0.0
    return cum_ret, sharpe, max_dd, end_eq


def _save_log(rows: list[WindowMetrics], path: Path) -> None:
    frame = pd.DataFrame([r.__dict__ for r in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Wrote training log -> %s", path)


def train(
    *,
    test: bool = False,
    epochs: int = DEFAULT_EPOCHS,
    window_days: int = DEFAULT_WINDOW_DAYS,
    output: Path | None = None,
    seed: int = 42,
) -> list[WindowMetrics]:
    output_dir = Path(output) if output is not None else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = _sanitize_panel(load_processed())
    if panel.empty:
        raise FileNotFoundError("unified_data.parquet is empty. Run `python -m src.data_loader` first.")

    wide = panel_to_wide(panel, "close")
    logger.info(
        "Loaded panel rows=%d tickers=%s span=%s -> %s | wide close %s",
        len(panel),
        sorted(panel["ticker"].unique().tolist()),
        panel["datetime"].min(),
        panel["datetime"].max(),
        tuple(wide.shape),
    )

    windows = iter_windows(panel, window_days)
    if test:
        windows = windows[:1]
        logger.info("--test: training a single window %s -> %s (no checkpoints).", windows[0].start.date(), windows[0].end.date())
    else:
        logger.info("Rolling %d sequential windows of %d session days (step = 1 day).", len(windows), window_days)

    model: PPO | None = None
    metrics: list[WindowMetrics] = []
    best_sharpe = -np.inf
    best_ckpt: Path | None = None
    n_windows = len(windows)

    for win in windows:
        n_bars = int(win.df["datetime"].nunique())
        steps = timesteps_for_window(win.df, epochs)
        logger.info(
            "Window %d/%d  %s -> %s  bars=%d  timesteps=%d",
            win.index,
            n_windows,
            win.start.date(),
            win.end.date(),
            n_bars,
            steps,
        )

        vec_env = make_vec_env(win.df, window_days)
        if model is None:
            model = make_ppo(vec_env, seed=seed)
        else:
            model.set_env(vec_env)

        # Sequential fine-tune: keep timesteps counter so PPO continues in "time".
        model.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=False)

        cum_ret, sharpe, max_dd, equity = evaluate_policy(model, win.df, window_days)
        row = WindowMetrics(
            window=win.index,
            start=str(win.start.date()),
            end=str(win.end.date()),
            n_bars=n_bars,
            timesteps=steps,
            cumulative_return=cum_ret,
            sharpe=sharpe,
            max_drawdown=max_dd,
            final_equity=equity,
        )
        metrics.append(row)
        logger.info(
            "Window %d metrics  return=%.4f  sharpe=%.4f  max_dd=%.4f  equity=%.2f",
            win.index,
            cum_ret,
            sharpe,
            max_dd,
            equity,
        )

        if not test:
            ckpt = output_dir / f"checkpoint_{win.end.date()}"
            model.save(str(ckpt))
            saved = ckpt.with_suffix(".zip")
            logger.info("Saved %s", saved)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_ckpt = saved

        vec_env.close()

    if not test:
        log_path = output_dir / "training_log.csv"
        _save_log(metrics, log_path)
        if best_ckpt is not None and best_ckpt.exists():
            dest = output_dir / "best_model.zip"
            shutil.copy2(best_ckpt, dest)
            logger.info("Best Sharpe=%.4f -> copied %s to %s", best_sharpe, best_ckpt.name, dest)
    else:
        logger.info("--test complete. Checkpoints and training_log.csv were not written.")

    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AirAire sequential 30-day PPO trainer")
    p.add_argument("--test", action="store_true", help="Train a single window and skip saving models.")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Passes over each window (default: 10).")
    p.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS, help="Session days per window (default: 30).")
    p.add_argument("--output", type=Path, default=MODELS_DIR, help="Checkpoint directory (default: models/).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train(
        test=args.test,
        epochs=args.epochs,
        window_days=args.window_days,
        output=args.output,
        seed=args.seed,
    )
