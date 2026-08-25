"""Daily 1-window fine-tune into models/news_gpu_v2_11 (HK-long / US-short).

Does **not** write ``models/news_gpu_v2`` or live ``best_model.zip``.
Seeds from the GPU trader banner zip (or the 20260823 dump of it).

Run after US close so the window end is a complete session:

    python -m src.finetune_v2_11 --windows 1 --device cuda
    python -m src.finetune_v2_11 --windows 1 --device cuda --telegram
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import freeze_support
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train_gpu_v2_11 import _patch_shared_modules
from src.utils import NEWS_GPU_V2_11_MODELS_DIR, setup_logging
from src.v2_11 import (
    guard_init_checkpoint,
    guard_output_dir,
    install_seed_into_v2_11,
    log_seed_banner,
    resolve_v2_paper_seed_zip,
)

logger = setup_logging("airaire.finetune_v2_11")


def _patch_finetune(*, output_dir: Path) -> None:
    import src.finetune_latest as ft

    _patch_shared_modules(output_dir=output_dir)
    # Name-only goldens (best_model.zip, checkpoint_2026-08-12.zip) live in V2.
    # Isolated v2.11 may write those filenames in its own folder.
    ft.PROTECTED_ZIPS = frozenset()
    ft.NEWS_GPU_V2_MODELS_DIR = output_dir


def finetune(
    *,
    n_windows: int = 1,
    window_days: int = 30,
    output: Path | None = None,
    checkpoint: Path | None = None,
    seed: int = 42,
    device: str = "cuda",
    refresh_futu: bool = True,
    lookback_days: int = 30,
    force_news_fetch: bool = True,
    skip_news: bool = False,
    news_days: int | None = None,
    notify_telegram: bool = False,
    promote_wait_seconds: int = 0,
):
    from src.finetune_latest import finetune as _finetune_v2

    output_dir = guard_output_dir(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_zip = resolve_v2_paper_seed_zip(explicit=checkpoint) if checkpoint is not None else resolve_v2_paper_seed_zip()
    install_seed_into_v2_11(seed_zip)
    # Prefer the newest dated zip already in v2.11; else the paper seed.
    ckpt = checkpoint
    if ckpt is None:
        dated = sorted(output_dir.glob("finetuned_*.zip")) + sorted(output_dir.glob("checkpoint_*.zip"))
        dated = [p for p in dated if p.name != "seed_from_v2_paper.zip"]
        if dated:
            ckpt = max(dated, key=lambda p: p.stat().st_mtime)
            logger.info("Continuing V2.11 from latest dated zip %s", ckpt)
        else:
            ckpt = seed_zip
    ckpt = guard_init_checkpoint(ckpt)
    log_seed_banner(ckpt, role="V2.11 fine-tune")
    _patch_finetune(output_dir=output_dir)
    logger.info(
        "V2.11 fine-tune -> %s  env=trading_env_v2_11  telegram=%s  (live news_gpu_v2 untouched)",
        output_dir,
        notify_telegram,
    )
    return _finetune_v2(
        n_windows=n_windows,
        window_days=window_days,
        output=output_dir,
        checkpoint=ckpt,
        seed=seed,
        device=device,
        refresh_futu=refresh_futu,
        lookback_days=lookback_days,
        force_news_fetch=force_news_fetch,
        skip_news=skip_news,
        news_days=news_days,
        notify_telegram=notify_telegram,
        promote_wait_seconds=promote_wait_seconds,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AirAire V2.11 daily fine-tune (isolated from live V2).")
    p.add_argument("--windows", type=int, default=1)
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--output", type=Path, default=NEWS_GPU_V2_11_MODELS_DIR)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Warm-start zip. Default: newest v2.11 dated zip, else GPU paper best_model.zip.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda", "auto"))
    p.add_argument("--no-futu", action="store_true")
    p.add_argument("--lookback-days", type=int, default=30)
    p.add_argument("--news-days", type=int, default=None)
    p.add_argument("--skip-news", action="store_true")
    p.add_argument("--cache-news", action="store_true")
    p.add_argument("--force-news-fetch", action="store_true")
    p.add_argument(
        "--telegram",
        action="store_true",
        help="Enable Promote/Keep Telegram (off by default so live V2 is not paged).",
    )
    p.add_argument("--promote-wait", type=int, default=0)
    p.add_argument("--promote-zip", type=Path, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    freeze_support()
    args = parse_args()
    output_dir = guard_output_dir(args.output)
    if args.promote_zip is not None:
        from src.finetune_latest import promote_zip_cli

        _patch_finetune(output_dir=output_dir)
        promote_zip_cli(output_dir, args.promote_zip)
        sys.exit(0)
    finetune(
        n_windows=args.windows,
        window_days=args.window_days,
        output=output_dir,
        checkpoint=args.checkpoint,
        seed=args.seed,
        device=args.device,
        refresh_futu=not args.no_futu,
        lookback_days=args.lookback_days,
        force_news_fetch=(not args.cache_news) or args.force_news_fetch,
        skip_news=args.skip_news,
        news_days=args.news_days,
        notify_telegram=bool(args.telegram),
        promote_wait_seconds=args.promote_wait,
    )
