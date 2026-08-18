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
import torch

from src.data_loader import load_enhanced_data, load_processed, merge_price_news, panel_to_wide
from src.trading_env import LOOKBACK_BARS, MAX_EQUITY, MIN_EQUITY, TRADING_DAYS, TradingEnv, news_obs_slice
from src.utils import (
    CORE_TICKERS,
    ENHANCED_PARQUET,
    INITIAL_CASH,
    MODELS_DIR,
    NEWS_MODELS_DIR,
    PRICE_ONLY_BEST_CHECKPOINT,
    PRICE_ONLY_MODELS_DIR,
    setup_logging,
)

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
    news_coverage: float = 0.0


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
    if "news_score" in out.columns:
        out["news_score"] = pd.to_numeric(out["news_score"], errors="coerce")
        out["news_score"] = out.groupby("ticker", group_keys=False)["news_score"].ffill().fillna(0.0)
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


def make_vec_env(
    df: pd.DataFrame,
    news_df: pd.DataFrame | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> DummyVecEnv:
    def _factory() -> Monitor:
        env = TradingEnv(
            df=df,
            news_df=news_df,
            initial_cash=INITIAL_CASH,
            window_days=window_days,
        )
        return Monitor(env)

    return DummyVecEnv([_factory])


def make_ppo(env: DummyVecEnv, seed: int, device: str = "cpu") -> PPO:
    # MlpPolicy on CUDA is slower and was the device that produced NaN loc on Colab.
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        device=device,
        n_steps=PPO_N_STEPS,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256])},
    )


def policy_has_nan(model: PPO) -> bool:
    return any(not torch.isfinite(p).all() for p in model.policy.parameters())


def timesteps_for_window(df: pd.DataFrame, epochs: int) -> int:
    n_bars = int(pd.to_datetime(df["datetime"]).nunique())
    episode = max(n_bars - LOOKBACK_BARS, 1)
    # At least one PPO rollout so learn() always performs an update.
    return max(epochs * episode, PPO_N_STEPS)


def evaluate_policy(
    model: PPO,
    df: pd.DataFrame,
    window_days: int,
    news_df: pd.DataFrame | None = None,
) -> tuple[float, float, float, float]:
    """Walk the window once, deterministically. Returns return, Sharpe, max DD, final equity."""
    env = TradingEnv(df=df, news_df=news_df, initial_cash=INITIAL_CASH, window_days=window_days)
    obs, _ = env.reset()
    if not np.all(np.isfinite(obs)):
        logger.warning("evaluate_policy: invalid initial observation. Returning flat metrics.")
        return 0.0, 0.0, 0.0, INITIAL_CASH

    terminated = truncated = False
    hold = np.zeros(env.action_space.shape, dtype=np.float32)
    while not (terminated or truncated):
        try:
            action, _ = model.predict(obs, deterministic=True)
        except ValueError as exc:
            logger.warning("evaluate_policy: predict failed (%s). Holding for the rest of the window.", exc)
            action = hold
        action = np.nan_to_num(np.asarray(action, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(action)):
            logger.warning("evaluate_policy: non-finite action. Holding.")
            action = hold
        obs, _, terminated, truncated, _ = env.step(action)
        if not np.all(np.isfinite(obs)):
            logger.warning("evaluate_policy: non-finite obs after step. Stopping walk.")
            break

    curve = np.clip(
        np.nan_to_num(np.asarray(env._equity_curve, dtype=np.float64), nan=INITIAL_CASH),
        MIN_EQUITY,
        MAX_EQUITY,
    )
    start_eq = float(curve[0]) if len(curve) else INITIAL_CASH
    end_eq = float(curve[-1]) if len(curve) else INITIAL_CASH
    if not np.isfinite(start_eq) or start_eq == 0:
        start_eq = INITIAL_CASH
    if not np.isfinite(end_eq):
        end_eq = start_eq
    cum_ret = (end_eq / start_eq) - 1.0

    rets = np.clip(np.nan_to_num(np.asarray(env._returns, dtype=np.float64), nan=0.0), -1.0, 1.0)
    if len(rets) < 2 or not np.isfinite(rets.std()) or float(rets.std()) < 1e-12:
        sharpe = 0.0
    else:
        sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(TRADING_DAYS * BARS_PER_DAY))
        if not np.isfinite(sharpe):
            sharpe = 0.0

    peak = np.maximum.accumulate(curve) if len(curve) else np.asarray([start_eq])
    dd = (peak - curve) / np.maximum(np.abs(peak), 1e-6)
    dd = dd[np.isfinite(dd)]
    max_dd = float(np.clip(dd.max(), 0.0, 1.0)) if len(dd) else 0.0
    return cum_ret, sharpe, max_dd, end_eq


def _save_log(rows: list[WindowMetrics], path: Path) -> None:
    frame = pd.DataFrame([r.__dict__ for r in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Wrote training log -> %s", path)


def _news_coverage(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    col = "news_score" if "news_score" in df.columns else ("sentiment_score" if "sentiment_score" in df.columns else None)
    if col is None:
        return 0.0
    core = df[df["ticker"].isin(CORE_TICKERS)] if "ticker" in df.columns else df
    if core.empty:
        return 0.0
    return float((pd.to_numeric(core[col], errors="coerce").abs() > 1e-12).mean())


def _window_news(news_df: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    """Slice news with a 7-day lookback so the first bars of a window still ffill."""
    if news_df is None or news_df.empty:
        return None
    lookback = pd.Timedelta(days=7)
    mask = (news_df["datetime"] >= start - lookback) & (news_df["datetime"] <= end + pd.Timedelta(days=1))
    sliced = news_df.loc[mask].copy()
    return sliced if not sliced.empty else None


def _probe_news_block(df: pd.DataFrame, news_df: pd.DataFrame | None, window_days: int) -> None:
    env = TradingEnv(df=df, news_df=news_df, initial_cash=INITIAL_CASH, window_days=window_days)
    obs, _ = env.reset()
    sl = news_obs_slice(env.lookback_bars)
    news_vec = np.asarray(obs[sl], dtype=np.float32)
    logger.info(
        "News block (4) sample=%s  nonzero=%d/%d  obs_dim=%d",
        np.round(news_vec, 4).tolist(),
        int(np.count_nonzero(np.abs(news_vec) > 1e-12)),
        len(news_vec),
        int(obs.shape[0]),
    )


def _resolve_checkpoint(
    windows: list[WindowSlice],
    resume: int,
    output_dir: Path,
    init_checkpoint: Path | None,
) -> tuple[int, Path | None]:
    """Return (1-based first window index, optional zip to load)."""
    if resume and resume > 1:
        matches = [w for w in windows if w.index == resume]
        if not matches:
            raise ValueError(f"--resume {resume} does not match any window (have 1..{windows[-1].index}).")
        prev_candidates = [w for w in windows if w.index == resume - 1]
        prev = prev_candidates[0] if prev_candidates else None
        ckpt = Path(init_checkpoint) if init_checkpoint else None
        if ckpt is None and prev is not None:
            named = f"checkpoint_{prev.end.date()}.zip"
            for folder in (output_dir, MODELS_DIR, PRICE_ONLY_MODELS_DIR):
                candidate = folder / named
                if candidate.exists():
                    ckpt = candidate
                    break
        if ckpt is None or not ckpt.exists():
            raise FileNotFoundError(
                f"--resume {resume} needs the previous window checkpoint "
                f"(looked for checkpoint_{prev.end.date() if prev is not None else '????-??-??'}.zip "
                f"in {output_dir} and {MODELS_DIR}). Pass --init-checkpoint PATH."
            )
        return resume, ckpt

    if init_checkpoint is not None:
        ckpt = Path(init_checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"--init-checkpoint not found: {ckpt}")
        return 1, ckpt
    return 1, None


def train(
    *,
    test: bool = False,
    epochs: int = DEFAULT_EPOCHS,
    window_days: int = DEFAULT_WINDOW_DAYS,
    output: Path | None = None,
    seed: int = 42,
    device: str = "cpu",
    resume: int = 0,
    init_checkpoint: Path | None = None,
    no_news: bool = False,
    force_news_fetch: bool = False,
) -> list[WindowMetrics]:
    use_news = not no_news
    if output is not None:
        output_dir = Path(output)
    else:
        output_dir = NEWS_MODELS_DIR if use_news else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_news:
        logger.info("News-integrated training. Checkpoints -> %s (price-only zips in %s are left untouched).", output_dir, MODELS_DIR)
        prices = _sanitize_panel(load_processed())
        if test:
            probe = iter_windows(prices, window_days)[0]
            from src.news_loader import load_all_news

            logger.info("--test: loading news for first window only (%s → %s).", probe.start.date(), probe.end.date())
            try:
                news_only = load_all_news(
                    probe.start - pd.Timedelta(days=7),
                    probe.end,
                    force_fetch=force_news_fetch,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("News fetch failed (%s). --test continues with zeros.", exc)
                news_only = pd.DataFrame(columns=["datetime", "ticker", "sentiment_score"])
            panel = _sanitize_panel(merge_price_news(prices, news_only))
        elif ENHANCED_PARQUET.exists() and not force_news_fetch:
            logger.info("Loading cached enhanced panel %s", ENHANCED_PARQUET)
            panel = _sanitize_panel(pd.read_parquet(ENHANCED_PARQUET))
            if "news_score" not in panel.columns:
                logger.warning("Enhanced parquet missing news_score; rebuilding.")
                panel = _sanitize_panel(load_enhanced_data(force_news_fetch=force_news_fetch))
        else:
            logger.info("Building enhanced panel (first-time Alpha Vantage backfill can take several minutes; later runs use data/raw/news/).")
            panel = _sanitize_panel(load_enhanced_data(force_news_fetch=force_news_fetch))
        if PRICE_ONLY_BEST_CHECKPOINT.exists() and init_checkpoint is None and (not resume or resume <= 1):
            logger.info(
                "Price-only baseline found at %s (Window 4 Sharpe 144.5). "
                "Pass --init-checkpoint %s to fine-tune it with news instead of training from scratch.",
                PRICE_ONLY_BEST_CHECKPOINT,
                PRICE_ONLY_BEST_CHECKPOINT,
            )
    else:
        logger.info("Price-only training (--no-news). News block (4) will be zeros.")
        panel = _sanitize_panel(load_processed())

    if panel.empty:
        raise FileNotFoundError("unified/enhanced parquet is empty. Run `python -m src.data_loader` first.")

    news_df = None
    if use_news and "news_score" in panel.columns:
        news_df = panel.loc[panel["ticker"].isin(CORE_TICKERS), ["datetime", "ticker", "news_score"]].rename(
            columns={"news_score": "sentiment_score"}
        )

    wide = panel_to_wide(panel, "close")
    logger.info(
        "Loaded panel rows=%d tickers=%s span=%s -> %s | wide close %s | news_coverage=%.1f%%",
        len(panel),
        sorted(panel["ticker"].unique().tolist()),
        panel["datetime"].min(),
        panel["datetime"].max(),
        tuple(wide.shape),
        100.0 * _news_coverage(panel),
    )

    windows = iter_windows(panel, window_days)
    if test:
        windows = windows[:1]
        logger.info("--test: training a single window %s -> %s (no checkpoints).", windows[0].start.date(), windows[0].end.date())
    else:
        logger.info("Rolling %d sequential windows of %d session days (step = 1 day).", len(windows), window_days)

    start_index, ckpt_to_load = (1, None) if test else _resolve_checkpoint(windows, resume, output_dir, init_checkpoint)
    if ckpt_to_load is not None:
        logger.info("Warm-start weights from %s  (first window index=%d)", ckpt_to_load, start_index)

    if windows:
        first = next((w for w in windows if w.index >= start_index), windows[0])
        _probe_news_block(first.df, _window_news(news_df, first.start, first.end), window_days)

    model: PPO | None = None
    metrics: list[WindowMetrics] = []
    best_sharpe = -np.inf
    best_ckpt: Path | None = None
    last_good_ckpt: Path | None = None
    n_windows = len(windows)

    for win in windows:
        if win.index < start_index:
            continue
        n_bars = int(win.df["datetime"].nunique())
        steps = timesteps_for_window(win.df, epochs)
        window_news = _window_news(news_df, win.start, win.end)
        coverage = _news_coverage(win.df)
        logger.info(
            "Window %d/%d  %s -> %s  bars=%d  timesteps=%d  news_coverage=%.1f%%",
            win.index,
            n_windows,
            win.start.date(),
            win.end.date(),
            n_bars,
            steps,
            100.0 * coverage,
        )

        vec_env = make_vec_env(win.df, window_news, window_days)
        if model is None:
            if ckpt_to_load is not None:
                model = PPO.load(str(ckpt_to_load), env=vec_env, device=device)
                logger.info("Loaded %s onto window %d env.", ckpt_to_load, win.index)
            else:
                model = make_ppo(vec_env, seed=seed, device=device)
        else:
            model.set_env(vec_env)

        # Sequential fine-tune: keep timesteps counter so PPO continues in "time".
        model.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=False)

        saved: Path | None = None
        if not test:
            ckpt = output_dir / f"checkpoint_{win.end.date()}"
            model.save(str(ckpt))
            saved = ckpt.with_suffix(".zip")
            logger.info("Saved %s", saved)

        if policy_has_nan(model):
            logger.error(
                "Window %d: policy weights contain NaN after learn(). Skipping eval; not promoting this checkpoint.",
                win.index,
            )
            if last_good_ckpt is not None and last_good_ckpt.exists():
                logger.warning("Reloading last finite checkpoint %s", last_good_ckpt)
                model = PPO.load(str(last_good_ckpt), env=vec_env, device=device)
            vec_env.close()
            continue

        cum_ret, sharpe, max_dd, equity = evaluate_policy(model, win.df, window_days, window_news)
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
            news_coverage=coverage,
        )
        metrics.append(row)
        logger.info(
            "Window %d metrics  return=%.4f  sharpe=%.4f  max_dd=%.4f  equity=%.2f  news_coverage=%.1f%%",
            win.index,
            cum_ret,
            sharpe,
            max_dd,
            equity,
            100.0 * coverage,
        )

        if saved is not None:
            last_good_ckpt = saved
            if np.isfinite(sharpe) and sharpe > best_sharpe:
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
    p.add_argument("--output", type=Path, default=None, help="Checkpoint directory (default: models/news with news, models/ if --no-news).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "auto"),
        help="PPO device. Default cpu (SB3 recommendation for MlpPolicy).",
    )
    p.add_argument(
        "--resume",
        type=int,
        default=0,
        help="1-based window index to start from (e.g. --resume 11 loads the window-10 checkpoint and continues).",
    )
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Warm-start PPO weights (same obs dim as price-only). Example: models/checkpoint_2026-04-02.zip",
    )
    p.add_argument("--no-news", action="store_true", help="Price-only baseline: news block stays zeros.")
    p.add_argument("--force-news-fetch", action="store_true", help="Re-query Alpha Vantage even if news cache covers the panel.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train(
        test=args.test,
        epochs=args.epochs,
        window_days=args.window_days,
        output=args.output,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        init_checkpoint=args.init_checkpoint,
        no_news=args.no_news,
        force_news_fetch=args.force_news_fetch,
    )
