"""GPU trainer v2 — more PPO updates per window, collapse guards, separate artifacts.

Copy of ``src.train_gpu`` with the 2026-08-19 post-mortem applied. The original
``train_gpu.py`` is left untouched (118-window run lives in ``models/news_gpu/``).

What went wrong last time
    * ``n_steps=4096 × n_envs=4`` → ~2 PPO updates / window on a ~19k budget.
      Sequential fine-tune then entropy-collapsed after window ~67.
    * Sharpe was always 0 (env MTM bug; fixed in ``trading_env.step``).
    * ``best_model.zip`` froze on Window 1 because ``0.0 > -inf`` once.

Reviewed vs DeepSeek's TRAINING-OPTIMIZATION.md
    * Keep ``n_steps=2048`` and force **8 updates / window** (agreed).
    * Keep ``n_envs=4``, not 6: this VM has 4 vCPUs. Six Subproc workers
      oversubscribe the host and each pickles a full env; collection was
      already ~1000 fps with 4. GPU time rises because we do 8 updates, not
      because we add idle workers.
    * Do **not** naively concat the full metrics list onto training_log.csv
      every window (that duplicates rows). Shared ``_save_log`` snapshots
      plus an append-only history file.
    * Extra (from TRAINING-RESULTS, not DeepSeek): ``ent_coef``, ``target_kl``,
      Calmar tie-break for ``best_model.zip``, reload last-good on collapse,
      and a ``logs/train_*.txt`` capture that does **not** replace sys.stdout
      (SB3 2.0 ``HumanOutputFormat`` requires a real ``TextIOBase``).

Checkpoints default to ``models/news_gpu_v2/``. SB3 2.x zips only.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from multiprocessing import freeze_support
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HumanOutputFormat, Logger as SB3Logger
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
    calmar_ratio,
    checkpoint_sort_key,
    evaluate_policy,
    iter_windows,
    policy_has_nan,
    resolve_device,
)
from src.data_loader import load_enhanced_data, load_processed, merge_price_news, panel_to_wide  # noqa: E402
from src.trading_env import LOOKBACK_BARS, TradingEnv  # noqa: E402
from src.utils import (  # noqa: E402
    CORE_TICKERS,
    ENHANCED_PARQUET,
    INITIAL_CASH,
    MODELS_DIR,
    NEWS_GPU_V2_MODELS_DIR,
    is_protected_inference_artifact,
    setup_logging,
)

logger = setup_logging("airaire.train_gpu_v2")

# A40-2Q (2 GB dedicated) + 4 vCPUs. Previous GPU run: n_steps=4096, n_envs=4
# → rollout 16384, only ~2 PPO updates / ~19k window (under-trained → collapse).
# n_steps=2048 keeps the GPU fed with more frequent updates; n_envs stays 4 so
# Subproc workers match vCPU count (DeepSeek suggested 6; that oversubscribes).
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 256  # drop to 128 via OOM downgrade if needed
PPO_N_ENVS = 4
PPO_NET_WIDTH = 512
PPO_MIN_BATCH = 32
# Force this many PPO rollouts per window (CPU run had ~9; GPU v1 had ~2).
DESIRED_PPO_UPDATES = 8
# Entropy bonus + early stop of inner epochs when the policy jumps too far.
# Previous run: entropy_loss -7.08 → -0.039, approx_kl 0.007 → 0.56.
PPO_ENT_COEF = 0.01
PPO_TARGET_KL = 0.03  # SB3 aborts the epoch when approx_kl > 1.5 * target_kl
# Heuristic: treat the policy as dead and roll back weights for the next window.
COLLAPSE_KL = 0.15
COLLAPSE_ENTROPY_LOSS = -1.0  # entropy_loss is -mean(entropy); closer to 0 = dead


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
        if self.n_steps > 1024:
            return replace(self, n_steps=max(1024, self.n_steps // 2), batch_size=min(self.batch_size, 128))
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


def _start_run_log(log_path: Path) -> tuple[object, TextIO]:
    """Write airaire logging to ``log_path`` without replacing sys.stdout.

    SB3 2.0 ``HumanOutputFormat`` does ``isinstance(sys.stdout, TextIOBase)``.
    A custom Tee fails that check even with write/close (2026-08-19 crash).
    PPO tables are mirrored onto this same real file in ``_bind_sb3_logger``.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    handler = logging.StreamHandler(log_file)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    attached: list[logging.Logger] = []
    names = set(logging.root.manager.loggerDict)
    names.update({"airaire", "airaire.train_gpu_v2", "airaire.train", "airaire.trading_env", "airaire.finetune_latest"})
    for name in names:
        if not str(name).startswith("airaire"):
            continue
        log = logging.getLogger(str(name))
        log.addHandler(handler)
        attached.append(log)

    def restore() -> None:
        for log in attached:
            log.removeHandler(handler)
        handler.close()
        log_file.close()

    return restore, log_file


def _bind_sb3_logger(model: PPO, log_file: TextIO) -> None:
    """Dump PPO verbose tables to the console *and* the run log file.

    ``set_logger`` sets ``_custom_logger=True`` so ``learn()`` will not call
    ``configure_logger`` → ``HumanOutputFormat(sys.stdout)`` on a fake stream.
    Both targets are real ``TextIOBase`` objects (console + open file).
    """
    formats = [HumanOutputFormat(log_file)]
    try:
        formats.insert(0, HumanOutputFormat(sys.stdout))
    except ValueError:
        # Console is not a TextIOBase (rare). File capture still works.
        logger.warning("sys.stdout is not a TextIOBase; PPO tables go to the log file only.")
    model.set_logger(SB3Logger(folder=None, output_formats=formats))


def _ppo_train_stats(model: PPO) -> dict[str, float]:
    """Last SB3 logger scalars (approx_kl / entropy) after ``learn()``."""
    values = getattr(getattr(model, "logger", None), "name_to_value", None) or {}
    out: dict[str, float] = {}
    for key in ("train/approx_kl", "train/entropy_loss", "train/clip_fraction", "train/explained_variance"):
        if key in values:
            try:
                out[key.split("/", 1)[-1]] = float(values[key])
            except (TypeError, ValueError):
                continue
    return out


def _policy_looks_collapsed(stats: dict[str, float]) -> bool:
    """True when KL exploded *and* entropy is nearly gone (v1 windows 110–118)."""
    kl = stats.get("approx_kl")
    ent = stats.get("entropy_loss")
    if kl is None or ent is None:
        return False
    return kl > COLLAPSE_KL and ent > COLLAPSE_ENTROPY_LOSS


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
        "make_ppo  n_steps=%d  batch_size=%d  n_envs=%d  net=[%d, %d]  rollout=%d  "
        "ent_coef=%.3f  target_kl=%.3f",
        cfg.n_steps,
        batch,
        cfg.n_envs,
        cfg.net_width,
        cfg.net_width,
        cfg.rollout_size(),
        PPO_ENT_COEF,
        PPO_TARGET_KL,
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
        # Default SB3 ent_coef=0.0: sequential 118-window fine-tune killed entropy.
        ent_coef=PPO_ENT_COEF,
        # Abort remaining inner epochs if the policy step is too large (clip 0.2).
        target_kl=PPO_TARGET_KL,
        n_epochs=10,
        policy_kwargs={
            "net_arch": dict(pi=[cfg.net_width, cfg.net_width], vf=[cfg.net_width, cfg.net_width]),
            "activation_fn": torch.nn.ReLU,
        },
    )


def timesteps_for_window(df: pd.DataFrame, epochs: int, cfg: GpuPpoConfig) -> int:
    """Budget enough vec-env steps for DESIRED_PPO_UPDATES rollouts.

    SB3 performs one PPO update per ``n_steps * n_envs`` collected steps.
    ``--epochs`` stays a *lower* bound (CPU-style episode × epochs) so a
    longer run is still possible; it cannot drop the window below 8 updates.
    """
    n_bars = int(pd.to_datetime(df["datetime"]).nunique())
    episode = max(n_bars - LOOKBACK_BARS, 1)
    rollout_size = cfg.rollout_size()
    desired_steps = DESIRED_PPO_UPDATES * rollout_size
    return max(desired_steps, episode * epochs, rollout_size)


def _resolve_checkpoint(
    windows: list[WindowSlice],
    resume: int,
    output_dir: Path,
    init_checkpoint: Path | None,
) -> tuple[int, Path | None]:
    """Resume only from ``--output`` (default ``models/news_gpu_v2``). Never CPU zips."""
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
    log_file: TextIO,
) -> tuple[PPO, VecEnv, GpuPpoConfig]:
    """Create env + PPO and ``learn()``. On CUDA OOM, downgrade and retry."""
    gpu_cb = GpuMonitorCallback(log_every=1)
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
            _bind_sb3_logger(model, log_file)
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
                    "(halve batch_size → DummyVecEnv → fewer envs → halve n_steps → net 256). "
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
    # Do not replace sys.stdout. SB3 2.0 requires a real TextIOBase; a custom
    # Tee crashed HumanOutputFormat. airaire logs + PPO tables still go to
    # logs/train_*.txt via a real file object.
    log_dir = _ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_filename = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    restore_log, log_file = _start_run_log(log_filename)
    try:
        logger.info("Terminal output is also written to %s", log_filename)
        return _train_impl(
            test=test,
            epochs=epochs,
            window_days=window_days,
            output=output,
            seed=seed,
            device=device,
            resume=resume,
            init_checkpoint=init_checkpoint,
            no_news=no_news,
            force_news_fetch=force_news_fetch,
            log_file=log_file,
        )
    finally:
        restore_log()


def _train_impl(
    *,
    test: bool,
    epochs: int,
    window_days: int,
    output: Path | None,
    seed: int,
    device: str,
    resume: int,
    init_checkpoint: Path | None,
    no_news: bool,
    force_news_fetch: bool,
    log_file: TextIO,
) -> list[WindowMetrics]:
    device = resolve_device(device)
    _configure_cuda()
    cfg = GpuPpoConfig()
    logger.info(
        "GPU v2 config  n_steps=%d  n_envs=%d  batch=%d  net=%d  desired_updates=%d  "
        "rollout=%d  steps/window>=%d  ent_coef=%.3f  target_kl=%.3f",
        cfg.n_steps,
        cfg.n_envs,
        cfg.aligned_batch(),
        cfg.net_width,
        DESIRED_PPO_UPDATES,
        cfg.rollout_size(),
        DESIRED_PPO_UPDATES * cfg.rollout_size(),
        PPO_ENT_COEF,
        PPO_TARGET_KL,
    )
    use_news = not no_news
    _save_log._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # type: ignore[attr-defined]
    output_dir = Path(output) if output is not None else NEWS_GPU_V2_MODELS_DIR
    if output_dir.resolve() == (MODELS_DIR / "news").resolve():
        raise ValueError(
            "Refusing to write GPU checkpoints into models/news/. "
            "Use the default models/news_gpu_v2 or another new directory."
        )
    if output_dir.resolve() == (MODELS_DIR / "news_gpu").resolve():
        logger.warning(
            "Output is models/news_gpu (v1 artifacts). Prefer models/news_gpu_v2 so the 2026-08-19 run is kept."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_news:
        logger.info(
            "News-integrated GPU v2 training. Checkpoints -> %s (models/news and models/news_gpu are left untouched).",
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

    if test:
        start_index = 1
        ckpt_to_load = Path(init_checkpoint) if init_checkpoint is not None else None
        if ckpt_to_load is not None and not ckpt_to_load.exists():
            raise FileNotFoundError(f"--test --init-checkpoint not found: {ckpt_to_load}")
    else:
        start_index, ckpt_to_load = _resolve_checkpoint(windows, resume, output_dir, init_checkpoint)
    if ckpt_to_load is not None:
        logger.info("Warm-start GPU weights from %s  (first window index=%d)", ckpt_to_load, start_index)

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
        steps = timesteps_for_window(win.df, epochs, cfg)
        ppo_updates = max(steps // max(cfg.rollout_size(), 1), 1)
        window_news = _window_news(news_df, win.start, win.end)
        coverage = _news_coverage(win.df)
        logger.info(
            "Window %d/%d  %s -> %s  bars=%d  timesteps=%d  n_envs=%d  n_steps=%d  "
            "ppo_updates~%d  news_coverage=%.1f%%",
            win.index,
            n_windows,
            win.start.date(),
            win.end.date(),
            n_bars,
            steps,
            cfg.n_envs,
            cfg.n_steps,
            ppo_updates,
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
            log_file=log_file,
        )

        saved: Path | None = None
        if not test:
            ckpt = output_dir / f"checkpoint_{win.end.date()}"
            if is_protected_inference_artifact(ckpt):
                logger.warning(
                    "Refusing to overwrite Phase-4 golden %s.zip. "
                    "Use src.finetune_latest (Promote) to change paper-trading weights.",
                    ckpt.name,
                )
            else:
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

        stats = _ppo_train_stats(model)
        collapsed = _policy_looks_collapsed(stats)
        if stats:
            logger.info(
                "Window %d PPO  approx_kl=%s  entropy_loss=%s  clip_fraction=%s%s",
                win.index,
                f"{stats['approx_kl']:.4f}" if "approx_kl" in stats else "n/a",
                f"{stats['entropy_loss']:.4f}" if "entropy_loss" in stats else "n/a",
                f"{stats['clip_fraction']:.4f}" if "clip_fraction" in stats else "n/a",
                "  COLLAPSED — will reload last-good weights" if collapsed else "",
            )

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
            approx_kl=stats.get("approx_kl", float("nan")),
            entropy_loss=stats.get("entropy_loss", float("nan")),
            ppo_updates=ppo_updates,
        )
        metrics.append(row)
        logger.info(
            "Window %d metrics  return=%.4f  sharpe=%.4f  max_dd=%.4f  calmar=%.4f  "
            "equity=%.2f  news_coverage=%.1f%%",
            win.index,
            cum_ret,
            sharpe,
            max_dd,
            calmar,
            equity,
            100.0 * coverage,
        )

        if saved is not None and not collapsed:
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
                # Do not promote on Sharpe==0 alone (v1 picked Window 1).
                key = checkpoint_sort_key(cum_ret, sharpe, max_dd)
                if best_key is None or key > best_key:
                    best_key = key
                    best_sharpe = sharpe
                    best_ckpt = saved
        elif collapsed:
            logger.warning(
                "Window %d not eligible for best_model.zip (entropy/KL collapse).",
                win.index,
            )

        # Roll the live policy back so window t+1 is not fine-tuned on a dead brain.
        if collapsed and last_good_ckpt is not None and last_good_ckpt.exists():
            logger.warning("Reloading last healthy checkpoint %s after collapse.", last_good_ckpt)
            model = PPO.load(str(last_good_ckpt), env=vec_env, device=device)

        if not test:
            _save_log(metrics, output_dir / "training_log.csv")

        _close_env(vec_env)

    if not test:
        log_path = output_dir / "training_log.csv"
        _save_log(metrics, log_path)
        if best_ckpt is not None and best_ckpt.exists():
            dest = output_dir / "best_model.zip"
            if is_protected_inference_artifact(dest):
                logger.warning(
                    "Refusing to overwrite %s. Paper trading is promoted only via "
                    "src.finetune_latest Telegram Promote / --promote-zip.",
                    dest,
                )
            else:
                shutil.copy2(best_ckpt, dest)
                logger.info(
                    "Best checkpoint Sharpe=%.4f Calmar=%.4f -> copied %s to %s",
                    best_sharpe,
                    best_key[1] if best_key is not None else float("nan"),
                    best_ckpt.name,
                    dest,
                )
        log_gpu_memory("finished")
    else:
        logger.info("--test complete. Checkpoints and training_log.csv were not written.")

    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AirAire GPU PPO trainer v2 (A40-2Q / 2GB VRAM, 8 updates/window)")
    p.add_argument("--test", action="store_true", help="Train a single window and skip saving models.")
    p.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Lower-bound passes over each window (v2 also forces 8 PPO updates; default: 10).",
    )
    p.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS, help="Session days per window (default: 30).")
    p.add_argument(
        "--output",
        type=Path,
        default=NEWS_GPU_V2_MODELS_DIR,
        help="Checkpoint directory (default: models/news_gpu_v2). Will not write to models/news.",
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
