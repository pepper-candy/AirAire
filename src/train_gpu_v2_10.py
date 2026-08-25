"""GPU trainer v2.10 — V2 PPO recipe, long-only, Bloomberg 5 + HSI (no SPX).

Same 8-updates-per-window recipe as ``train_gpu_v2``. ``best_model.zip`` is
still elected on in-sample Calmar (the V2 protocol), not V3 holdout.

Isolated writes: ``models/news_gpu_v2_10/`` and ``enhanced_v2_10.parquet``.
Refuses V2 (782-dim, shorts allowed) and V3/V4 (1082-dim, SPX) zips.

    python -m src.train_gpu_v2_10 --device cuda --skip-futu
    python -m src.train_gpu_v2_10 --device cuda --skip-futu --test
    python -m src.train_gpu_v2_10 --device cuda --start 2026-06-15 --end 2026-08-21 --skip-futu
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import freeze_support
from pathlib import Path

import pandas as pd
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader_v2_10 import (  # noqa: E402
    DEFAULT_PANEL_END,
    ENHANCED_V2_10_PARQUET,
    NEWS_GPU_V2_10_MODELS_DIR,
    V2_10_TICKERS,
    load_enhanced_v2_10,
)
from src.trading_env_v2_10 import LOOKBACK_BARS, TradingEnv, news_obs_slice, observation_dim  # noqa: E402
from src.train_gpu_v2 import (  # noqa: E402
    DESIRED_PPO_UPDATES,
    GpuPpoConfig,
    train as _train_v2,
)
from src.utils import INITIAL_CASH, MODELS_DIR, NEWS_GPU_V2_11_MODELS_DIR, NEWS_GPU_V2_MODELS_DIR, setup_logging  # noqa: E402

logger = setup_logging("airaire.train_gpu_v2_10")

_FORCE_REBUILD = False
_SKIP_FUTU = False
_SLICE_START: pd.Timestamp | None = None
_SLICE_END: pd.Timestamp | None = None
_ORIG_SANITIZE = None
_FORBIDDEN = frozenset(
    {
        NEWS_GPU_V2_MODELS_DIR.resolve(),
        NEWS_GPU_V2_11_MODELS_DIR.resolve(),
        (MODELS_DIR / "news").resolve(),
        (MODELS_DIR / "news_gpu").resolve(),
        (MODELS_DIR / "news_gpu_v3").resolve(),
        (MODELS_DIR / "news_gpu_v3_1").resolve(),
        (MODELS_DIR / "news_gpu_v3_2").resolve(),
        (MODELS_DIR / "news_gpu_v4").resolve(),
        (MODELS_DIR / "news_gpu_v4_1").resolve(),
    }
)
_BLOCKED_INIT_DIRS = frozenset(
    {
        "news_gpu_v2",
        "news_gpu_v2_11",
        "news_gpu_v3",
        "news_gpu_v3_1",
        "news_gpu_v3_2",
        "news_gpu_v4",
        "news_gpu_v4_1",
        "news",
        "news_gpu",
    }
)


def _parse_day(raw: str | None) -> pd.Timestamp | None:
    if raw is None or str(raw).strip() == "":
        return None
    ts = pd.Timestamp(raw)
    if pd.isna(ts):
        raise ValueError(f"Bad date {raw!r}. Use YYYY-MM-DD.")
    return ts.normalize()


def _guard_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved in _FORBIDDEN:
        raise ValueError(f"V2.10 refuses to write to {path}. Use models/news_gpu_v2_10.")
    if resolved.name == "news_gpu_v2":
        raise ValueError("V2.10 refuses models/news_gpu_v2 (live V2 brain). Use models/news_gpu_v2_10.")
    return Path(path)


def _guard_init_checkpoint(path: Path | None) -> Path | None:
    if path is None:
        return None
    parts = Path(path).resolve().parts
    if "news_gpu_v2_10" in parts:
        return Path(path)
    if any(part in _BLOCKED_INIT_DIRS for part in parts):
        raise ValueError(
            "V2.10 cannot warm-start from a V2/V3/V4 zip "
            f"(obs {observation_dim()} vs 782/1082, and actions are [0, 1]). "
            "Train V2.10 from scratch (no --init-checkpoint)."
        )
    return Path(path)


def _env_thunk(df: pd.DataFrame, news_df: pd.DataFrame | None, window_days: int):
    """Must live in this module so Subproc workers import the V2.10 env."""

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
            logger.info("VecEnv=SubprocVecEnv  n_envs=%d  (V2.10 long-only, HSI only)", cfg.n_envs)
            return env
        except Exception as exc:  # noqa: BLE001
            logger.warning("SubprocVecEnv failed (%s). Falling back to DummyVecEnv.", exc)
            cfg.use_subproc = False
    env = DummyVecEnv(fns)
    logger.info("VecEnv=DummyVecEnv  n_envs=%d  (V2.10 long-only, HSI only)", cfg.n_envs)
    return env


def _slice_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if panel is None or panel.empty or (_SLICE_START is None and _SLICE_END is None):
        return panel
    dt = pd.to_datetime(panel["datetime"])
    mask = pd.Series(True, index=panel.index)
    if _SLICE_START is not None:
        mask &= dt >= _SLICE_START
    if _SLICE_END is not None:
        mask &= dt < (_SLICE_END + pd.Timedelta(days=1))
    out = panel.loc[mask].copy()
    logger.info(
        "V2.10 train slice %s → %s  rows=%d/%d",
        None if _SLICE_START is None else _SLICE_START.date(),
        None if _SLICE_END is None else _SLICE_END.date(),
        len(out),
        len(panel),
    )
    if out.empty:
        raise ValueError("Train slice is empty. Check --start / --end against enhanced_v2_10.parquet.")
    return out


def _sanitize_and_slice(panel: pd.DataFrame) -> pd.DataFrame:
    orig = _ORIG_SANITIZE
    cleaned = orig(panel) if orig is not None else panel
    if cleaned is not None and not cleaned.empty and "ticker" in cleaned.columns:
        keep = set(V2_10_TICKERS)
        cleaned = cleaned.loc[cleaned["ticker"].astype(str).isin(keep)].copy()
    return _slice_panel(cleaned)


def _load_enhanced_for_v2_train(force_news_fetch: bool = False, **_kwargs):
    return load_enhanced_v2_10(
        save=True,
        fetch_futu=not _SKIP_FUTU,
        force_rebuild=_FORCE_REBUILD or force_news_fetch,
        force_news_fetch=force_news_fetch,
    )


def _load_processed_panel() -> pd.DataFrame:
    return load_enhanced_v2_10(save=True, fetch_futu=not _SKIP_FUTU, force_rebuild=_FORCE_REBUILD)


def _patch_shared_modules(*, output_dir: Path) -> None:
    import src.train as train_mod
    import src.train_gpu_v2 as v2

    global _ORIG_SANITIZE
    if _ORIG_SANITIZE is None:
        _ORIG_SANITIZE = train_mod._sanitize_panel

    train_mod.TradingEnv = TradingEnv
    train_mod.news_obs_slice = news_obs_slice
    train_mod._sanitize_panel = _sanitize_and_slice
    v2.TradingEnv = TradingEnv
    v2.LOOKBACK_BARS = LOOKBACK_BARS
    v2.ENHANCED_PARQUET = ENHANCED_V2_10_PARQUET
    v2.NEWS_GPU_V2_MODELS_DIR = output_dir
    v2.load_enhanced_data = _load_enhanced_for_v2_train
    v2.load_processed = _load_processed_panel
    v2.make_vec_env = make_vec_env
    v2._env_thunk = _env_thunk
    v2._sanitize_panel = _sanitize_and_slice


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
    panel_start: str | None = None,
    panel_end: str | None = None,
):
    global _FORCE_REBUILD, _SKIP_FUTU, _SLICE_START, _SLICE_END
    _FORCE_REBUILD = bool(force_rebuild)
    _SKIP_FUTU = bool(skip_futu)
    _SLICE_START = _parse_day(panel_start)
    _SLICE_END = _parse_day(panel_end)
    output_dir = _guard_output(output or NEWS_GPU_V2_10_MODELS_DIR)
    init_checkpoint = _guard_init_checkpoint(init_checkpoint)
    _patch_shared_modules(output_dir=output_dir)
    logger.info(
        "GPU v2.10 long-only  output=%s  panel=%s  desired_updates=%d  observers=HSI  "
        "action=[0,1]  obs_dim=%d  train_slice=%s..%s  fetch_futu=%s",
        output_dir,
        ENHANCED_V2_10_PARQUET,
        DESIRED_PPO_UPDATES,
        observation_dim(),
        None if _SLICE_START is None else _SLICE_START.date(),
        None if _SLICE_END is None else _SLICE_END.date(),
        not skip_futu,
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
    p = argparse.ArgumentParser(
        description="AirAire GPU PPO trainer v2.10 (V2 recipe, long-only, 5 names + HSI, Bloomberg)."
    )
    p.add_argument("--test", action="store_true", help="Train a single window and skip saving models.")
    p.add_argument("--epochs", type=int, default=10, help="Lower-bound passes over each window (also forces 8 PPO updates).")
    p.add_argument("--window-days", type=int, default=30, help="Session days per window (default: 30).")
    p.add_argument(
        "--output",
        type=Path,
        default=NEWS_GPU_V2_10_MODELS_DIR,
        help="Checkpoint directory (default: models/news_gpu_v2_10). Cannot be a V2/V3/V4 folder.",
    )
    p.add_argument("--start", default=None, help="Optional train-window start (YYYY-MM-DD). Default: full panel.")
    p.add_argument(
        "--end",
        default=DEFAULT_PANEL_END,
        help="Last train calendar day (naive local-market stamps). Default 2026-08-21 so 2026-08-24 is a fair test.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda", "auto"))
    p.add_argument("--resume", type=int, default=0, help="1-based window index; loads the previous V2.10 checkpoint.")
    p.add_argument("--init-checkpoint", type=Path, default=None, help="Warm-start from a V2.10 zip only.")
    p.add_argument("--no-news", action="store_true")
    p.add_argument("--force-news-fetch", action="store_true", help="Rebuild panel and recopy V2 news_score.")
    p.add_argument("--force-rebuild", action="store_true", help="Rebuild enhanced_v2_10.parquet from Bloomberg.")
    p.add_argument("--skip-futu", action="store_true", help="Do not call OpenD (keep it on the live V2 trader).")
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
        panel_start=args.start,
        panel_end=args.end,
    )
