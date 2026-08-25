"""GPU trainer v2.11 — V2 PPO recipe, 782-dim, HK clipped long, US free to short.

Warm-start from the GPU paper zip (``run_trader.bat`` banner), not V2.10 and
not ``checkpoint_2026-08-20.zip``. Writes only ``models/news_gpu_v2_11/``.

    python -m src.train_gpu_v2_11 --device cuda --test
    python -m src.train_gpu_v2_11 --device cuda --init-checkpoint models/news_gpu_v2/best_model.zip
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

from src.trading_env_v2_11 import LOOKBACK_BARS, TradingEnv, news_obs_slice, observation_dim
from src.train_gpu_v2 import DESIRED_PPO_UPDATES, GpuPpoConfig
from src.train_gpu_v2 import train as _train_v2
from src.utils import INITIAL_CASH, NEWS_GPU_V2_11_MODELS_DIR, setup_logging
from src.v2_11 import (
    V2_11_OBS_DIM,
    guard_init_checkpoint,
    guard_output_dir,
    install_seed_into_v2_11,
    log_seed_banner,
    resolve_v2_paper_seed_zip,
)

logger = setup_logging("airaire.train_gpu_v2_11")


def _env_thunk(df: pd.DataFrame, news_df: pd.DataFrame | None, window_days: int):
    """Must live in this module so Subproc workers import the V2.11 env."""

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
            logger.info("VecEnv=SubprocVecEnv  n_envs=%d  (V2.11 HK-long / US-short)", cfg.n_envs)
            return env
        except Exception as exc:  # noqa: BLE001
            logger.warning("SubprocVecEnv failed (%s). Falling back to DummyVecEnv.", exc)
            cfg.use_subproc = False
    env = DummyVecEnv(fns)
    logger.info("VecEnv=DummyVecEnv  n_envs=%d  (V2.11 HK-long / US-short)", cfg.n_envs)
    return env


def _patch_shared_modules(*, output_dir: Path) -> None:
    import src.train as train_mod
    import src.train_gpu_v2 as v2

    train_mod.TradingEnv = TradingEnv
    train_mod.news_obs_slice = news_obs_slice
    v2.TradingEnv = TradingEnv
    v2.LOOKBACK_BARS = LOOKBACK_BARS
    v2.NEWS_GPU_V2_MODELS_DIR = output_dir
    v2.make_vec_env = make_vec_env
    v2._env_thunk = _env_thunk


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
    from_scratch: bool = False,
):
    output_dir = guard_output_dir(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if from_scratch:
        ckpt = None
        logger.warning("V2.11 --from-scratch: random PPO, not the paper zip.")
    elif init_checkpoint is None and resume <= 1:
        paper = resolve_v2_paper_seed_zip()
        install_seed_into_v2_11(paper)
        ckpt = guard_init_checkpoint(paper)
        log_seed_banner(ckpt, role="V2.11 train")
    else:
        ckpt = guard_init_checkpoint(init_checkpoint)
        if ckpt is not None:
            log_seed_banner(ckpt, role="V2.11 train")
    _patch_shared_modules(output_dir=output_dir)
    logger.info(
        "GPU v2.11 hybrid  output=%s  obs_dim=%d  action=Box[-1,1]  HK clip in env  "
        "desired_updates=%d  seed=%s",
        output_dir,
        V2_11_OBS_DIM,
        DESIRED_PPO_UPDATES,
        ckpt,
    )
    if observation_dim() != 782:
        logger.warning("observation_dim()=%d (expected 782). Check LOOKBACK / CORE_TICKERS.", observation_dim())
    return _train_v2(
        test=test,
        epochs=epochs,
        window_days=window_days,
        output=output_dir,
        seed=seed,
        device=device,
        resume=resume,
        init_checkpoint=ckpt,
        no_news=no_news,
        force_news_fetch=force_news_fetch,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AirAire GPU PPO trainer v2.11 (V2 782-dim zip family, HK-long / US-short)."
    )
    p.add_argument("--test", action="store_true", help="Train a single window and skip saving models.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument(
        "--output",
        type=Path,
        default=NEWS_GPU_V2_11_MODELS_DIR,
        help="Checkpoint directory (default: models/news_gpu_v2_11). Cannot be news_gpu_v2.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda", "auto"))
    p.add_argument("--resume", type=int, default=0)
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Warm-start zip. Default: GPU paper best_model.zip (banner path), not 2026-08-20.",
    )
    p.add_argument("--no-news", action="store_true")
    p.add_argument("--force-news-fetch", action="store_true")
    p.add_argument(
        "--from-scratch",
        action="store_true",
        help="Do not load the V2 paper zip (random policy). You almost never want this.",
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
        from_scratch=args.from_scratch,
    )
