"""GPU-optimised sequential 30-day PPO trainer (A40-2Q, 2 GB VRAM).

Same rolling-window loop as ``src.train``. The 30 FPS ceiling was the env
(pandas per step), not the PPO size — ``TradingEnv`` now precomputes numpy
features. GPU knobs still help the *update* (batch 256, net 512, n_steps 4096).

* ``SubprocVecEnv`` × 4 is optional extra collection parallelism
* ``learn(..., reset_num_timesteps=False)`` so windows share one clock
* CUDA OOM → log, empty cache, halve ``batch_size`` (then drop envs / steps)

Checkpoints go to ``models/news_gpu/`` and never touch ``models/news/``.
SB3 2.x zips from this script are **not** interchangeable with CPU 1.7.x zips.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train import (  # noqa: E402
    DEFAULT_EPOCHS,
    DEFAULT_WINDOW_DAYS,
    EVAL_RETURN_CLIP,
    EVAL_SHARPE_CLIP,
    WindowMetrics,
    WindowSlice,
    _news_coverage,
    _probe_news_block,
    _sanitize_panel,
    _save_log,
    _window_news,
    evaluate_policy,
    iter_windows,
    policy_has_nan,
    resolve_device,
)
from src.data_loader import load_enhanced_data, load_processed, merge_price_news, panel_to_wide  # noqa: E402
from src.trading_env import LOOKBACK_BARS, TradingEnv  # noqa: E402
from src.utils import CORE_TICKERS, ENHANCED_PARQUET, INITIAL_CASH, MODELS_DIR, setup_logging  # noqa: E402

logger = setup_logging("airaire.train_gpu")

NEWS_GPU_MODELS_DIR = MODELS_DIR / "news_gpu"

# Conservative A40-2Q (2 GB) starting point. Auto-downgrade on OOM.
PPO_N_STEPS = 4096
PPO_BATCH_SIZE = 256
PPO_N_ENVS = 4
PPO_NET_WIDTH = 512
PPO_MIN_BATCH = 32


@dataclass
class GpuPpoConfig:
    n_envs: int = PPO_N_ENVS
    n_steps: int = PPO_N_STEPS
    batch_size: int = PPO_BATCH_SIZE
    net_width: int = PPO_NET_WIDTH
    use_subproc: bool = True

    def rollout_size(self) -> int:
        return int(self.n_steps * self.n_envs)

    def aligned_batch(self) -> int:
        """Largest batch_size <= self.batch_size that divides n_steps * n_envs."""
        rollout = self.rollout_size()
        bs = min(max(int(self.batch_size), 1), rollout)
        while bs > 1 and rollout % bs != 0:
            bs -= 1
        return bs

    def downgrade(self) -> GpuPpoConfig | None:
        """Smaller config for CUDA OOM. Prefer halving batch_size first."""
        if self.batch_size > PPO_MIN_BATCH:
            return replace(self, batch_size=max(PPO_MIN_BATCH, self.batch_size // 2))
        if self.use_subproc:
            return replace(self, use_subproc=False)
        if self.n_envs > 1:
            return replace(self, n_envs=max(1, self.n_envs // 2))
        if self.n_steps > 2048:
            return replace(self, n_steps=2048, batch_size=min(self.batch_size, 128))
        if self.net_width > 256:
            return replace(self, net_width=256, batch_size=min(self.batch_size, 64))
        return None


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def log_gpu_memory(tag: str) -> None:
    """Log allocated/reserved VRAM and (if possible) SM utilization."""
    if not torch.cuda.is_available():
        logger.info("GPU monitor [%s]: CUDA not available", tag)
        return
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    util = _gpu_utilization()
    logger.info(
        "GPU monitor [%s]: alloc=%.0f MB  reserved=%.0f MB  total=%.0f MB  util=%s",
        tag,
        allocated,
        reserved,
        total,
        util,
    )


def _gpu_utilization() -> str:
    smi = _nvidia_smi_line()
    if smi:
        return smi
    try:
        return f"{torch.cuda.utilization()}%"
    except Exception:
        return "n/a"


def _nvidia_smi_line() -> str | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        if len(parts) >= 3:
            return f"{parts[0]}%  nvidia-smi mem={parts[1]}/{parts[2]} MB"
        return out.strip()
    except Exception:
        return None


class GpuMonitorCallback(BaseCallback):
    """Print VRAM / util every ``log_every`` PPO rollouts (SB3 'iterations')."""

    def __init__(self, log_every: int = 10, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_every = max(int(log_every), 1)
        self._rollouts = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._rollouts += 1
        if self._rollouts == 1 or self._rollouts % self.log_every == 0:
            log_gpu_memory(f"rollout_{self._rollouts}")


def _env_thunk(df: pd.DataFrame, news_df: pd.DataFrame | None, window_days: int):
    """Top-level-friendly factory; cloudpickle carries the window frames into workers."""

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
            logger.info("VecEnv=SubprocVecEnv  n_envs=%d", cfg.n_envs)
            return env
        except Exception as exc:  # noqa: BLE001
            logger.warning("SubprocVecEnv failed (%s). Falling back to DummyVecEnv for the rest of this run.", exc)
            cfg.use_subproc = False
    env = DummyVecEnv(fns)
    logger.info("VecEnv=DummyVecEnv  n_envs=%d", cfg.n_envs)
    return env


def make_ppo(env: VecEnv, seed: int, device: str, cfg: GpuPpoConfig) -> PPO:
    """Build PPO. ``reset_num_timesteps`` is a ``learn()`` flag (always False there)."""
    batch = cfg.aligned_batch()
    logger.info(
        "make_ppo  n_steps=%d  batch_size=%d  n_envs=%d  net=[%d, %d]  rollout=%d",
        cfg.n_steps,
        batch,
        cfg.n_envs,
        cfg.net_width,
        cfg.net_width,
        cfg.rollout_size(),
    )
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        device=device,
        n_steps=cfg.n_steps,
        batch_size=batch,
        learning_rate=3e-4,
        gamma=0.99,
        max_grad_norm=0.5,
        policy_kwargs={
            "net_arch": dict(pi=[cfg.net_width, cfg.net_width], vf=[cfg.net_width, cfg.net_width]),
            "activation_fn": torch.nn.ReLU,
        },
    )


def timesteps_for_window(df: pd.DataFrame, epochs: int, cfg: GpuPpoConfig) -> int:
    n_bars = int(pd.to_datetime(df["datetime"]).nunique())
    episode = max(n_bars - LOOKBACK_BARS, 1)
    # Same epoch budget as train.py (not × n_envs) so 4 workers finish ~4× faster.
    return max(epochs * episode, cfg.rollout_size())


def _resolve_checkpoint(
    windows: list[WindowSlice],
    resume: int,
    output_dir: Path,
    init_checkpoint: Path | None,
) -> tuple[int, Path | None]:
    """Resume only from ``models/news_gpu`` (or ``--init-checkpoint``). Never CPU zips."""
    if resume and resume > 1:
        matches = [w for w in windows if w.index == resume]
        if not matches:
            raise ValueError(f"--resume {resume} does not match any window (have 1..{windows[-1].index}).")
        prev_candidates = [w for w in windows if w.index == resume - 1]
        prev = prev_candidates[0] if prev_candidates else None
        ckpt = Path(init_checkpoint) if init_checkpoint else None
        if ckpt is None and prev is not None:
            candidate = output_dir / f"checkpoint_{prev.end.date()}.zip"
            if candidate.exists():
                ckpt = candidate
        if ckpt is None or not ckpt.exists():
            raise FileNotFoundError(
                f"--resume {resume} needs the previous GPU checkpoint "
                f"(looked for checkpoint_{prev.end.date() if prev is not None else '????-??-??'}.zip "
                f"in {output_dir}). Pass --init-checkpoint PATH. "
                "Do not load models/news CPU zips — SB3 1.7 vs 2.x are incompatible."
            )
        return resume, ckpt

    if init_checkpoint is not None:
        ckpt = Path(init_checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"--init-checkpoint not found: {ckpt}")
        return 1, ckpt
    return 1, None


def _load_or_create(
    vec_env: VecEnv,
    seed: int,
    device: str,
    cfg: GpuPpoConfig,
    ckpt: Path | None,
) -> PPO:
    if ckpt is None:
        return make_ppo(vec_env, seed=seed, device=device, cfg=cfg)
    try:
        loaded = PPO.load(str(ckpt), env=vec_env, device=device)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not load %s (%s). GPU trainer starts from scratch "
            "(CPU SB3 1.7 checkpoints are not compatible).",
            ckpt,
            exc,
        )
        return make_ppo(vec_env, seed=seed, device=device, cfg=cfg)

    same_hp = (
        int(loaded.n_steps) == cfg.n_steps
        and int(loaded.n_envs) == cfg.n_envs
        and int(loaded.batch_size) == cfg.aligned_batch()
    )
    if same_hp:
        logger.info("Loaded %s (n_steps=%d batch=%d n_envs=%d).", ckpt, loaded.n_steps, loaded.batch_size, loaded.n_envs)
        return loaded

    logger.warning(
        "Checkpoint hyperparams (n_steps=%d batch=%d n_envs=%d) differ from GPU config "
        "(n_steps=%d batch=%d n_envs=%d). Copying weights into a fresh PPO.",
        loaded.n_steps,
        loaded.batch_size,
        loaded.n_envs,
        cfg.n_steps,
        cfg.aligned_batch(),
        cfg.n_envs,
    )
    fresh = make_ppo(vec_env, seed=seed, device=device, cfg=cfg)
    try:
        fresh.policy.load_state_dict(loaded.policy.state_dict())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Weight copy failed (%s). Continuing with randomly initialised GPU policy.", exc)
    return fresh


def _close_env(env: VecEnv | None) -> None:
    if env is None:
        return
    try:
        env.close()
    except Exception as exc:  # noqa: BLE001
        logger.debug("vec_env.close() raised %s", exc)


def _learn_window(
    *,
    model: PPO | None,
    win: WindowSlice,
    window_news: pd.DataFrame | None,
    window_days: int,
    cfg: GpuPpoConfig,
    seed: int,
    device: str,
    steps: int,
    ckpt_to_load: Path | None,
    last_good_ckpt: Path | None,
) -> tuple[PPO, VecEnv, GpuPpoConfig]:
    """Create env + PPO and ``learn()``. On CUDA OOM, downgrade and retry."""
    gpu_cb = GpuMonitorCallback(log_every=10)
    while True:
        vec_env: VecEnv | None = None
        try:
            vec_env = make_vec_env(win.df, window_news, window_days, cfg)
            if model is None:
                warm = last_good_ckpt if last_good_ckpt is not None else ckpt_to_load
                model = _load_or_create(vec_env, seed, device, cfg, warm)
            else:
                need_rebuild = int(model.n_steps) != cfg.n_steps or int(model.n_envs) != cfg.n_envs
                if need_rebuild:
                    logger.warning("Rebuilding PPO to match downgraded config %s", cfg)
                    warm = last_good_ckpt if last_good_ckpt is not None else ckpt_to_load
                    model = _load_or_create(vec_env, seed, device, cfg, warm)
                else:
                    model.set_env(vec_env)
            log_gpu_memory(f"window_{win.index}_before_learn")
            # Sequential windows share one timestep clock — never reset to 0.
            model.learn(
                total_timesteps=steps,
                reset_num_timesteps=False,
                progress_bar=False,
                callback=gpu_cb,
            )
            log_gpu_memory(f"window_{win.index}_after_learn")
            return model, vec_env, cfg
        except Exception as exc:
            _close_env(vec_env)
            if device == "cuda":
                torch.cuda.empty_cache()
            if not _is_cuda_oom(exc):
                raise
            nxt = cfg.downgrade()
            logger.error(
                "CUDA out of memory with n_steps=%d batch_size=%d n_envs=%d net=%d subproc=%s. "
                "Freeing cache and retrying with a smaller config. Original error: %s",
                cfg.n_steps,
                cfg.aligned_batch(),
                cfg.n_envs,
                cfg.net_width,
                cfg.use_subproc,
                exc,
            )
            if nxt is None:
                raise RuntimeError(
                    "CUDA out of memory after exhausting downgrade steps "
                    "(halve batch_size → DummyVecEnv → fewer envs → n_steps=2048 → net 256). "
                    "Try --test first or close other GPU processes."
                ) from exc
            logger.warning(
                "Downgraded GPU config: n_steps=%d batch_size=%d n_envs=%d net=%d subproc=%s",
                nxt.n_steps,
                nxt.aligned_batch(),
                nxt.n_envs,
                nxt.net_width,
                nxt.use_subproc,
            )
            cfg = nxt
            model = None


def _configure_cuda() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    logger.info("CUDA device=%s  VRAM=%.2f GB  cc=%d.%d", props.name, vram_gb, props.major, props.minor)
    if vram_gb < 2.5:
        logger.info("VRAM is tight (%.2f GB). Starting at n_steps=%d batch=%d n_envs=%d; OOM will auto-downgrade.", vram_gb, PPO_N_STEPS, PPO_BATCH_SIZE, PPO_N_ENVS)
    log_gpu_memory("startup")


def train(
    *,
    test: bool = False,
    epochs: int = DEFAULT_EPOCHS,
    window_days: int = DEFAULT_WINDOW_DAYS,
    output: Path | None = None,
    seed: int = 42,
    device: str = "cuda",
    resume: int = 0,
    init_checkpoint: Path | None = None,
    no_news: bool = False,
    force_news_fetch: bool = False,
) -> list[WindowMetrics]:
    device = resolve_device(device)
    _configure_cuda()
    cfg = GpuPpoConfig()
    use_news = not no_news
    output_dir = Path(output) if output is not None else NEWS_GPU_MODELS_DIR
    if output_dir.resolve() == (MODELS_DIR / "news").resolve():
        raise ValueError(
            "Refusing to write GPU checkpoints into models/news/. "
            "Use the default models/news_gpu or another new directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_news:
        logger.info(
            "News-integrated GPU training. Checkpoints -> %s (models/news is left untouched).",
            output_dir,
        )
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
    else:
        logger.info("Price-only GPU training (--no-news). News block (4) will be zeros.")
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
        logger.info("Warm-start GPU weights from %s  (first window index=%d)", ckpt_to_load, start_index)

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
        steps = timesteps_for_window(win.df, epochs, cfg)
        window_news = _window_news(news_df, win.start, win.end)
        coverage = _news_coverage(win.df)
        logger.info(
            "Window %d/%d  %s -> %s  bars=%d  timesteps=%d  n_envs=%d  n_steps=%d  news_coverage=%.1f%%",
            win.index,
            n_windows,
            win.start.date(),
            win.end.date(),
            n_bars,
            steps,
            cfg.n_envs,
            cfg.n_steps,
            100.0 * coverage,
        )

        model, vec_env, cfg = _learn_window(
            model=model,
            win=win,
            window_news=window_news,
            window_days=window_days,
            cfg=cfg,
            seed=seed,
            device=device,
            steps=steps,
            ckpt_to_load=ckpt_to_load if model is None else None,
            last_good_ckpt=last_good_ckpt,
        )

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
            _close_env(vec_env)
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
            elif np.isfinite(sharpe) and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_ckpt = saved

        if not test:
            _save_log(metrics, output_dir / "training_log.csv")

        _close_env(vec_env)

    if not test:
        log_path = output_dir / "training_log.csv"
        _save_log(metrics, log_path)
        if best_ckpt is not None and best_ckpt.exists():
            dest = output_dir / "best_model.zip"
            shutil.copy2(best_ckpt, dest)
            logger.info("Best Sharpe=%.4f -> copied %s to %s", best_sharpe, best_ckpt.name, dest)
        log_gpu_memory("finished")
    else:
        logger.info("--test complete. Checkpoints and training_log.csv were not written.")

    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AirAire GPU PPO trainer (A40-2Q / 2GB VRAM)")
    p.add_argument("--test", action="store_true", help="Train a single window and skip saving models.")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Passes over each window (default: 10).")
    p.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS, help="Session days per window (default: 30).")
    p.add_argument(
        "--output",
        type=Path,
        default=NEWS_GPU_MODELS_DIR,
        help="Checkpoint directory (default: models/news_gpu). Will not write to models/news.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        default="cuda",
        choices=("cpu", "cuda", "auto"),
        help="PPO device. Default cuda.",
    )
    p.add_argument(
        "--resume",
        type=int,
        default=0,
        help="1-based window index to start from (loads the previous GPU checkpoint in --output).",
    )
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Warm-start from a GPU (SB3 2.x) zip. CPU models/news zips are not compatible.",
    )
    p.add_argument("--no-news", action="store_true", help="Price-only baseline: news block stays zeros.")
    p.add_argument("--force-news-fetch", action="store_true", help="Re-query Alpha Vantage even if news cache covers the panel.")
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
    )
