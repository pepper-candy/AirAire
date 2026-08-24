"""GPU trainer v3 — isolated research track.

Same PPO recipe as ``train_gpu_v2`` (8 updates/window, collapse guards).
``best_model.zip`` is chosen on the next holdout session days, not on the
window that was just trained. In-sample Calmar is still printed for comparison.

Writes only to ``models/news_gpu_v3/``. Do not import this from the live trader.
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader_v3 import (  # noqa: E402
    ENHANCED_V3_PARQUET,
    NEWS_GPU_V3_MODELS_DIR,
    load_enhanced_v3,
)
from src.trading_env_v3 import LOOKBACK_BARS, TradingEnv, news_obs_slice  # noqa: E402
from src.train import (  # noqa: E402
    EVAL_RETURN_CLIP,
    EVAL_SHARPE_CLIP,
    calmar_ratio,
    evaluate_policy as _evaluate_in_sample,
)
from src.trading_env import MAX_EQUITY, MIN_EQUITY  # noqa: E402
from src.train_gpu_v2 import (  # noqa: E402
    DESIRED_PPO_UPDATES,
    GpuPpoConfig,
    train as _train_v2,
)
from src.utils import CORE_TICKERS, INITIAL_CASH, MODELS_DIR, NEWS_GPU_V2_MODELS_DIR, setup_logging  # noqa: E402

logger = setup_logging("airaire.train_gpu_v3")

DEFAULT_HOLDOUT_DAYS = 5
DEFAULT_HOLDOUT_SMOOTH = 10
MIN_HOLDOUTS_FOR_BEST = 3
_FORCE_REBUILD = False
_SKIP_FUTU = False
_HOLDOUT_DAYS = DEFAULT_HOLDOUT_DAYS
_HOLDOUT_SMOOTH = DEFAULT_HOLDOUT_SMOOTH
_FULL_PANEL: pd.DataFrame | None = None
_HOLDOUT_HISTORY: list[tuple[float, float, float]] = []
_V2_FORBIDDEN = frozenset(
    {
        NEWS_GPU_V2_MODELS_DIR.resolve(),
        (MODELS_DIR / "news").resolve(),
        (MODELS_DIR / "news_gpu").resolve(),
    }
)


def _guard_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved in _V2_FORBIDDEN:
        raise ValueError(f"V3 refuses to write to {path}. Use models/news_gpu_v3.")
    return Path(path)


def _guard_init_checkpoint(path: Path | None) -> Path | None:
    if path is None:
        return None
    text = str(Path(path)).replace("\\", "/")
    if "news_gpu_v2" in text:
        raise ValueError(
            "V3 cannot warm-start from a V2 zip — observation_dim changed "
            "(5-name OHLCV vs 5+2 observers). Train V3 from scratch (no --init-checkpoint)."
        )
    return Path(path)


def _env_thunk(df: pd.DataFrame, news_df: pd.DataFrame | None, window_days: int):
    """Must live in this module so Subproc workers import TradingEnv from v3."""

    def _init() -> Monitor:
        env = TradingEnv(
            df=df,
            news_df=news_df,
            initial_cash=INITIAL_CASH,
            window_days=window_days,
        )
        return Monitor(env)

    return _init


def make_vec_env(
    df: pd.DataFrame,
    news_df: pd.DataFrame | None,
    window_days: int,
    cfg: GpuPpoConfig,
) -> VecEnv:
    fns = [_env_thunk(df, news_df, window_days) for _ in range(cfg.n_envs)]
    if cfg.use_subproc and cfg.n_envs > 1:
        try:
            env = SubprocVecEnv(fns)
            logger.info("VecEnv=SubprocVecEnv  n_envs=%d  (V3 env)", cfg.n_envs)
            return env
        except Exception as exc:  # noqa: BLE001
            logger.warning("SubprocVecEnv failed (%s). Falling back to DummyVecEnv.", exc)
            cfg.use_subproc = False
    env = DummyVecEnv(fns)
    logger.info("VecEnv=DummyVecEnv  n_envs=%d  (V3 env)", cfg.n_envs)
    return env


def _remember_panel(panel: pd.DataFrame) -> pd.DataFrame:
    global _FULL_PANEL
    _FULL_PANEL = panel
    return panel


def _load_enhanced_for_v2_train(force_news_fetch: bool = False, **_kwargs):
    return _remember_panel(
        load_enhanced_v3(
            save=True,
            fetch_futu=not _SKIP_FUTU,
            force_news_fetch=force_news_fetch,
            force_rebuild=_FORCE_REBUILD or force_news_fetch,
        )
    )


def _load_processed_v3() -> pd.DataFrame:
    return _remember_panel(load_enhanced_v3(save=True, fetch_futu=not _SKIP_FUTU, force_rebuild=_FORCE_REBUILD))


def _panel_for_holdout() -> pd.DataFrame | None:
    if _FULL_PANEL is not None and not _FULL_PANEL.empty:
        return _FULL_PANEL
    if ENHANCED_V3_PARQUET.exists():
        return pd.read_parquet(ENHANCED_V3_PARQUET)
    return None


def _holdout_bounds(train_df: pd.DataFrame, full: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """First/last bar of the next ``_HOLDOUT_DAYS`` session dates after the training window."""
    train_end = pd.to_datetime(train_df["datetime"]).max().normalize()
    norm = pd.to_datetime(full["datetime"]).dt.normalize()
    dates = pd.DatetimeIndex(norm.unique()).sort_values()
    future = dates[dates > train_end]
    if len(future) == 0:
        return None
    chosen = future[: max(int(_HOLDOUT_DAYS), 1)]
    mask = norm.isin(chosen)
    bars = pd.to_datetime(full.loc[mask, "datetime"])
    if bars.empty:
        return None
    return pd.Timestamp(bars.min()), pd.Timestamp(bars.max())


def _curve_metrics(curve: np.ndarray) -> tuple[float, float, float, float]:
    start_eq = float(curve[0]) if len(curve) else INITIAL_CASH
    end_eq = float(curve[-1]) if len(curve) else INITIAL_CASH
    if not np.isfinite(start_eq) or start_eq == 0:
        start_eq = INITIAL_CASH
    if not np.isfinite(end_eq):
        end_eq = start_eq
    cum_ret = (end_eq / start_eq) - 1.0
    if len(curve) >= 3:
        denom = np.maximum(np.abs(curve[:-1]), 1e-9)
        rets = np.clip((curve[1:] - curve[:-1]) / denom, -1.0, 1.0)
        rets = rets[np.isfinite(rets)]
    else:
        rets = np.asarray([], dtype=np.float64)
    if len(rets) < 2 or not np.isfinite(rets.std()):
        sharpe = 0.0
    else:
        std = float(rets.std())
        sharpe = 0.0 if std < 1e-12 else float(rets.mean() / (std + 1e-9))
        if not np.isfinite(sharpe):
            sharpe = 0.0
    peak = np.maximum.accumulate(curve) if len(curve) else np.asarray([start_eq])
    dd = (peak - curve) / np.maximum(np.abs(peak), 1e-6)
    dd = dd[np.isfinite(dd)]
    max_dd = float(np.clip(dd.max(), 0.0, 1.0)) if len(dd) else 0.0
    return cum_ret, sharpe, max_dd, end_eq


def _clip_eval(cum_ret: float, sharpe: float, max_dd: float, end_eq: float) -> tuple[float, float, float, float]:
    start_eq = INITIAL_CASH
    if (
        cum_ret < EVAL_RETURN_CLIP[0]
        or cum_ret > EVAL_RETURN_CLIP[1]
        or sharpe < EVAL_SHARPE_CLIP[0]
        or sharpe > EVAL_SHARPE_CLIP[1]
        or end_eq > start_eq * (1.0 + EVAL_RETURN_CLIP[1])
        or end_eq < start_eq * (1.0 + EVAL_RETURN_CLIP[0])
    ):
        logger.warning(
            "Holdout metrics out of range before clip  return=%.4f  sharpe=%.4f  max_dd=%.4f  equity=%.2f",
            cum_ret,
            sharpe,
            max_dd,
            end_eq,
        )
    cum_ret = float(np.clip(cum_ret, EVAL_RETURN_CLIP[0], EVAL_RETURN_CLIP[1]))
    sharpe = float(np.clip(sharpe, EVAL_SHARPE_CLIP[0], EVAL_SHARPE_CLIP[1]))
    end_eq = float(start_eq * (1.0 + cum_ret))
    return cum_ret, sharpe, max_dd, end_eq


def evaluate_holdout(
    model,
    train_df: pd.DataFrame,
    window_days: int,
    news_df: pd.DataFrame | None = None,
) -> tuple[float, float, float, float]:
    """Score the policy on the next session days after ``train_df`` (not in-sample)."""
    full = _panel_for_holdout()
    if full is None or full.empty:
        logger.warning("Holdout skipped (no full panel). Returning flat metrics.")
        return 0.0, 0.0, 0.0, INITIAL_CASH
    bounds = _holdout_bounds(train_df, full)
    if bounds is None:
        logger.info("Holdout skipped (no future session days after this window).")
        return 0.0, 0.0, 0.0, INITIAL_CASH
    hold_start, hold_end = bounds
    news = news_df
    if news is None and "news_score" in full.columns:
        news = full.loc[full["ticker"].isin(CORE_TICKERS), ["datetime", "ticker", "news_score"]].rename(
            columns={"news_score": "sentiment_score"}
        )
    env = TradingEnv(df=full, news_df=news, initial_cash=INITIAL_CASH, window_days=window_days)
    obs, _ = env.reset()
    env.seek_to_datetime(hold_start)
    env.restore_portfolio(INITIAL_CASH, {t: 0.0 for t in CORE_TICKERS})
    env._returns = []
    env._equity_curve = [env.initial_cash]
    env._last_equity = env.initial_cash
    obs = env._get_obs()
    if not np.all(np.isfinite(obs)):
        logger.warning("Holdout initial obs invalid. Returning flat metrics.")
        return 0.0, 0.0, 0.0, INITIAL_CASH

    hold = np.zeros(env.action_space.shape, dtype=np.float32)
    steps = 0
    while True:
        now = pd.Timestamp(env._current_dt())
        if now > hold_end:
            break
        if env._bar_index >= len(env.datetimes) - 1:
            break
        try:
            action, _ = model.predict(obs, deterministic=True)
        except ValueError:
            action = hold
        action = np.nan_to_num(np.asarray(action, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated or not np.all(np.isfinite(obs)):
            break

    curve = np.clip(
        np.nan_to_num(np.asarray(env._equity_curve, dtype=np.float64), nan=INITIAL_CASH),
        MIN_EQUITY,
        MAX_EQUITY,
    )
    cum_ret, sharpe, max_dd, end_eq = _curve_metrics(curve)
    logger.info(
        "Holdout eval  %s → %s  steps=%d  return=%.4f  sharpe=%.4f  max_dd=%.4f  equity=%.2f  (in-sample ignored for best_model)",
        hold_start.date(),
        hold_end.date(),
        steps,
        cum_ret,
        sharpe,
        max_dd,
        end_eq,
    )
    if steps < LOOKBACK_BARS:
        logger.warning("Holdout walk too short (%d steps). Not using this window for best_model.", steps)
        return 0.0, 0.0, 0.0, INITIAL_CASH
    return _clip_eval(cum_ret, sharpe, max_dd, end_eq)


def checkpoint_sort_key_v3(cum_ret: float, sharpe: float, max_dd: float) -> tuple[float, float, float]:
    """Pick best_model from a rolling median of holdouts, not one noisy week."""
    empty = abs(float(cum_ret)) < 1e-15 and abs(float(sharpe)) < 1e-15 and abs(float(max_dd)) < 1e-15
    if empty:
        logger.info("Holdout missing/empty — this window cannot take best_model.")
        return (float("-inf"), float("-inf"), float("-inf"))
    _HOLDOUT_HISTORY.append((float(cum_ret), float(sharpe), float(max_dd)))
    recent = _HOLDOUT_HISTORY[-max(int(_HOLDOUT_SMOOTH), 1) :]
    if len(_HOLDOUT_HISTORY) < MIN_HOLDOUTS_FOR_BEST:
        logger.info(
            "Holdout history %d/%d — not enough weeks to elect best_model yet.",
            len(_HOLDOUT_HISTORY),
            MIN_HOLDOUTS_FOR_BEST,
        )
        return (float("-inf"), float("-inf"), float("-inf"))
    med_ret = float(np.median([row[0] for row in recent]))
    med_calmar = float(np.median([calmar_ratio(row[0], row[2]) for row in recent]))
    med_dd = float(np.median([row[2] for row in recent]))
    logger.info(
        "Holdout smooth n=%d  median_return=%.4f  median_calmar=%.4f  median_dd=%.4f  (this week return=%.4f)",
        len(recent),
        med_ret,
        med_calmar,
        med_dd,
        float(cum_ret),
    )
    return (med_ret, med_calmar, -med_dd)


def evaluate_policy_v3(model, df, window_days, news_df=None):
    """V3 scorer: log in-sample, return holdout metrics for best_model / training_log."""
    try:
        ins = _evaluate_in_sample(model, df, window_days, news_df)
        logger.info(
            "In-sample (not used for best_model)  return=%.4f  sharpe=%.4f  max_dd=%.4f  equity=%.2f",
            ins[0],
            ins[1],
            ins[2],
            ins[3],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("In-sample eval failed (%s).", exc)
    return evaluate_holdout(model, df, window_days, news_df)


def _patch_shared_modules() -> None:
    """Point v2 trainer helpers at V3 env/data without editing V2 files on disk."""
    import src.train as train_mod
    import src.train_gpu_v2 as v2

    train_mod.TradingEnv = TradingEnv
    train_mod.news_obs_slice = news_obs_slice
    train_mod.evaluate_policy = evaluate_policy_v3
    train_mod.checkpoint_sort_key = checkpoint_sort_key_v3
    v2.TradingEnv = TradingEnv
    v2.LOOKBACK_BARS = LOOKBACK_BARS
    v2.ENHANCED_PARQUET = ENHANCED_V3_PARQUET
    v2.NEWS_GPU_V2_MODELS_DIR = NEWS_GPU_V3_MODELS_DIR
    v2.load_enhanced_data = _load_enhanced_for_v2_train
    v2.load_processed = _load_processed_v3
    v2.make_vec_env = make_vec_env
    v2._env_thunk = _env_thunk
    v2.evaluate_policy = evaluate_policy_v3
    v2.checkpoint_sort_key = checkpoint_sort_key_v3


def train(
    *,
    test: bool = False,
    epochs: int = 10,
    window_days: int = 30,
    output: Path | None = None,
    seed: int = 42,
    device: str = "cuda",
    resume: int = 0,
    init_checkpoint: Path | None = None,
    no_news: bool = False,
    force_news_fetch: bool = False,
    force_rebuild: bool = False,
    skip_futu: bool = False,
    holdout_days: int = DEFAULT_HOLDOUT_DAYS,
    holdout_smooth: int = DEFAULT_HOLDOUT_SMOOTH,
):
    global _FORCE_REBUILD, _SKIP_FUTU, _HOLDOUT_DAYS, _HOLDOUT_SMOOTH, _HOLDOUT_HISTORY
    _FORCE_REBUILD = bool(force_rebuild)
    _SKIP_FUTU = bool(skip_futu)
    _HOLDOUT_DAYS = max(int(holdout_days), 1)
    _HOLDOUT_SMOOTH = max(int(holdout_smooth), 1)
    _HOLDOUT_HISTORY = []
    output_dir = _guard_output(output or NEWS_GPU_V3_MODELS_DIR)
    init_checkpoint = _guard_init_checkpoint(init_checkpoint)
    _patch_shared_modules()
    logger.info(
        "GPU v3  output=%s  panel=%s  desired_updates=%d  observers=HSI+SPX  "
        "window=2026-02-24..2026-08-21  fetch_futu=%s  holdout_days=%d  "
        "holdout_smooth=%d (best_model = rolling median of holdouts, not one week)",
        output_dir,
        ENHANCED_V3_PARQUET,
        DESIRED_PPO_UPDATES,
        not skip_futu,
        _HOLDOUT_DAYS,
        _HOLDOUT_SMOOTH,
    )
    return _train_v2(
        test=test,
        epochs=epochs,
        window_days=window_days,
        output=output_dir,
        seed=seed,
        device=device,
        resume=resume,
        init_checkpoint=init_checkpoint,
        no_news=no_news,
        force_news_fetch=force_news_fetch,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AirAire GPU PPO trainer v3 (isolated; does not touch V2).")
    p.add_argument("--test", action="store_true", help="Train a single window and skip saving models.")
    p.add_argument("--epochs", type=int, default=10, help="Lower-bound passes over each window (also forces 8 PPO updates).")
    p.add_argument("--window-days", type=int, default=30, help="Session days per window (default: 30).")
    p.add_argument(
        "--output",
        type=Path,
        default=NEWS_GPU_V3_MODELS_DIR,
        help="Checkpoint directory (default: models/news_gpu_v3). Cannot be a V2 folder.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda", "auto"))
    p.add_argument("--resume", type=int, default=0, help="1-based window index; loads the previous V3 checkpoint.")
    p.add_argument("--init-checkpoint", type=Path, default=None, help="Warm-start from a V3 (not V2) SB3 zip.")
    p.add_argument("--no-news", action="store_true")
    p.add_argument("--force-news-fetch", action="store_true")
    p.add_argument("--force-rebuild", action="store_true", help="Rebuild enhanced_v3.parquet from Bloomberg+TV+Futu.")
    p.add_argument("--skip-futu", action="store_true", help="Do not call OpenD for CATL volume (cache only).")
    p.add_argument(
        "--holdout-days",
        type=int,
        default=DEFAULT_HOLDOUT_DAYS,
        help="Session days AFTER each training window (not inside it). Default: 5.",
    )
    p.add_argument(
        "--holdout-smooth",
        type=int,
        default=DEFAULT_HOLDOUT_SMOOTH,
        help="Median over this many recent holdouts elects best_model (default: 10).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    freeze_support()
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
        force_rebuild=args.force_rebuild,
        skip_futu=args.skip_futu,
        holdout_days=args.holdout_days,
        holdout_smooth=args.holdout_smooth,
    )
