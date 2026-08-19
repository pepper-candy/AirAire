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
from datetime import datetime
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
from src.trading_env import LOOKBACK_BARS, MAX_EQUITY, MIN_EQUITY, N_CORE, TradingEnv, news_obs_slice
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
# Clip reported eval metrics so a broken walk cannot dominate best-model selection.
EVAL_RETURN_CLIP = (-1.0, 3.0)
EVAL_SHARPE_CLIP = (-3.0, 5.0)


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
    # Optional diagnostics (defaults keep older CSV rows loadable).
    calmar: float = 0.0
    approx_kl: float = float("nan")
    entropy_loss: float = float("nan")
    ppo_updates: int = 0


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


def resolve_device(device: str) -> str:
    """Honor --device cuda/cpu, or pick CUDA when available (--device auto)."""
    requested = (device or "auto").lower()
    cuda_ok = torch.cuda.is_available()
    if requested == "auto":
        chosen = "cuda" if cuda_ok else "cpu"
        logger.info("device=auto -> %s (torch.cuda.is_available=%s)", chosen, cuda_ok)
        return chosen
    if requested == "cuda" and not cuda_ok:
        logger.warning("CUDA requested but torch.cuda.is_available() is False. Falling back to cpu.")
        return "cpu"
    logger.info("PPO device=%s  cuda_available=%s", requested, cuda_ok)
    return requested


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
    """Walk the window once, deterministically. Returns return, Sharpe, max DD, final equity.

    Sharpe is the raw mean/std of bar returns inside this window — no
    ``sqrt(252 * 39)`` annualization (that factor ~99× turned modest paths into
    Sharpe 20). A fresh ``TradingEnv`` is built and ``reset()`` so training-env
    cash/holdings cannot bleed into the metrics.
    """
    env = TradingEnv(df=df, news_df=news_df, initial_cash=INITIAL_CASH, window_days=window_days)
    obs, _ = env.reset()
    # Belt-and-suspenders: reset() already clears these; re-assert so eval never
    # inherits a previous walk if Gym wrappers change.
    env._cash = env.initial_cash
    env._holdings = np.zeros(N_CORE, dtype=np.float64)
    env._returns = []
    env._equity_curve = [env.initial_cash]
    env._last_equity = env.initial_cash
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

    # Sharpe from the equity curve (the P&L path), not env._returns.
    # 2026-08-19: step() marked to market at the same prices as the rebalance,
    # so _returns were ~0 and every window logged Sharpe=0.0000. The curve
    # already included the next bar's move — use that as the source of truth.
    if len(curve) >= 3:
        denom = np.maximum(np.abs(curve[:-1]), 1e-9)
        rets = np.clip((curve[1:] - curve[:-1]) / denom, -1.0, 1.0)
        rets = rets[np.isfinite(rets)]
    else:
        rets = np.asarray([], dtype=np.float64)
    if len(rets) < 2 or not np.isfinite(rets.std()):
        sharpe = 0.0
    else:
        # Raw window Sharpe (no annualization). Tiny std → treat as flat, not inf.
        std = float(rets.std())
        sharpe = 0.0 if std < 1e-12 else float(rets.mean() / (std + 1e-9))
        if not np.isfinite(sharpe):
            sharpe = 0.0

    peak = np.maximum.accumulate(curve) if len(curve) else np.asarray([start_eq])
    dd = (peak - curve) / np.maximum(np.abs(peak), 1e-6)
    dd = dd[np.isfinite(dd)]
    max_dd = float(np.clip(dd.max(), 0.0, 1.0)) if len(dd) else 0.0

    if (
        cum_ret < EVAL_RETURN_CLIP[0]
        or cum_ret > EVAL_RETURN_CLIP[1]
        or sharpe < EVAL_SHARPE_CLIP[0]
        or sharpe > EVAL_SHARPE_CLIP[1]
        or end_eq > start_eq * (1.0 + EVAL_RETURN_CLIP[1])
        or end_eq < start_eq * (1.0 + EVAL_RETURN_CLIP[0])
    ):
        logger.warning(
            "evaluate_policy: metrics out of range before clip  return=%.4f  sharpe=%.4f  "
            "max_dd=%.4f  equity=%.2f  (healthy: return[-1,3] sharpe[-3,5] equity~0.9e6-1.5e6)",
            cum_ret,
            sharpe,
            max_dd,
            end_eq,
        )
    cum_ret = float(np.clip(cum_ret, EVAL_RETURN_CLIP[0], EVAL_RETURN_CLIP[1]))
    sharpe = float(np.clip(sharpe, EVAL_SHARPE_CLIP[0], EVAL_SHARPE_CLIP[1]))
    end_eq = float(start_eq * (1.0 + cum_ret))
    return cum_ret, sharpe, max_dd, end_eq


def calmar_ratio(cum_ret: float, max_dd: float) -> float:
    """Return / max drawdown. Used as a Sharpe tie-break (and fallback when Sharpe≈0)."""
    return float(cum_ret) / max(float(max_dd), 1e-6)


def checkpoint_sort_key(cum_ret: float, sharpe: float, max_dd: float) -> tuple[float, float, float]:
    """Higher tuple wins when choosing ``best_model.zip``.

    The 2026-08-19 GPU run used ``sharpe > best_sharpe`` while Sharpe was always
    0.0, so Window 1 (the first 0.0 > -inf) froze as the winner. Near-zero
    Sharpe is treated as non-informative so Calmar (then raw return) decides.
    """
    s = float(sharpe) if np.isfinite(sharpe) else 0.0
    sharpe_key = 0.0 if abs(s) < 1e-8 else s
    return (sharpe_key, calmar_ratio(cum_ret, max_dd), float(cum_ret))


def _save_log(rows: list[WindowMetrics], path: Path) -> None:
    """Snapshot this run and append *new* rows to a cross-run history file.

    DeepSeek's concat(existing, entire in-memory list) on every window would
    duplicate w1, then w1+w2, … because the trainer already calls this after
    each window *and* at the end with the full ``metrics`` list. We:

    * overwrite ``path`` with the current-run snapshot (crash recovery)
    * append only unseen ``run_id``+window+start+end keys to
      ``training_log_history.csv`` so previous jobs are never wiped
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([r.__dict__ for r in rows])
    new_df.to_csv(path, index=False)

    run_id = getattr(_save_log, "_run_id", None)
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        _save_log._run_id = run_id  # type: ignore[attr-defined]

    if new_df.empty:
        logger.info("Wrote training log -> %s (rows=0)", path)
        return

    hist_df = new_df.copy()
    hist_df.insert(0, "run_id", run_id)
    history_path = path.with_name("training_log_history.csv")
    key_cols = [c for c in ("run_id", "window", "start", "end") if c in hist_df.columns]
    appended = 0
    if history_path.exists():
        existing = pd.read_csv(history_path)
        if key_cols and all(c in existing.columns for c in key_cols):
            existing_keys = set(zip(*(existing[c].astype(str) for c in key_cols)))
            row_keys = list(zip(*(hist_df[c].astype(str) for c in key_cols)))
            mask = [key not in existing_keys for key in row_keys]
            to_append = hist_df.loc[mask]
        else:
            to_append = hist_df
        if not to_append.empty:
            pd.concat([existing, to_append], ignore_index=True).to_csv(history_path, index=False)
            appended = len(to_append)
    else:
        hist_df.to_csv(history_path, index=False)
        appended = len(hist_df)

    logger.info(
        "Wrote training log -> %s (snapshot rows=%d); history %s appended=%d",
        path,
        len(new_df),
        history_path,
        appended,
    )


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
    device = resolve_device(device)
    # Fresh id so this job's rows append to training_log_history.csv without
    # colliding with a previous run's window 1..N keys.
    _save_log._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # type: ignore[attr-defined]
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
    best_key: tuple[float, float, float] | None = None
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
        calmar = calmar_ratio(cum_ret, max_dd)
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
            calmar=calmar,
        )
        metrics.append(row)
        logger.info(
            "Window %d metrics  return=%.4f  sharpe=%.4f  max_dd=%.4f  calmar=%.4f  equity=%.2f  news_coverage=%.1f%%",
            win.index,
            cum_ret,
            sharpe,
            max_dd,
            calmar,
            equity,
            100.0 * coverage,
        )

        if saved is not None:
            last_good_ckpt = saved
            hit_clip = (
                np.isclose(sharpe, EVAL_SHARPE_CLIP[0])
                or np.isclose(sharpe, EVAL_SHARPE_CLIP[1])
                or np.isclose(cum_ret, EVAL_RETURN_CLIP[0])
                or np.isclose(cum_ret, EVAL_RETURN_CLIP[1])
            )
            if hit_clip:
                logger.warning(
                    "Window %d hit eval clip bounds; not using this checkpoint for best_model.zip.",
                    win.index,
                )
            elif np.isfinite(sharpe):
                # Sharpe primary; Calmar then return as tie-break / Sharpe≈0 fallback.
                key = checkpoint_sort_key(cum_ret, sharpe, max_dd)
                if best_key is None or key > best_key:
                    best_key = key
                    best_sharpe = sharpe
                    best_ckpt = saved

        if not test:
            _save_log(metrics, output_dir / "training_log.csv")

        vec_env.close()

    if not test:
        log_path = output_dir / "training_log.csv"
        _save_log(metrics, log_path)
        if best_ckpt is not None and best_ckpt.exists():
            dest = output_dir / "best_model.zip"
            shutil.copy2(best_ckpt, dest)
            logger.info(
                "Best checkpoint Sharpe=%.4f Calmar=%.4f -> copied %s to %s",
                best_sharpe,
                best_key[1] if best_key is not None else float("nan"),
                best_ckpt.name,
                dest,
            )
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
        default="auto",
        choices=("cpu", "cuda", "auto"),
        help="PPO device. Default auto (cuda if available, else cpu).",
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
