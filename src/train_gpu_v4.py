"""GPU trainer v4 — Bloomberg 7-name, volume=0, V3 holdout protocol.

Isolated: writes ``models/news_gpu_v4/``. Refuses V2 (782-dim) and V3 (TV volume) zips.
In-sample Calmar is printed; ``best_model.zip`` follows the rolling holdout median.

    python -m src.train_gpu_v4 --device cuda --skip-futu
    python -m src.train_gpu_v4 --device cuda --start 2026-06-15 --end 2026-08-21 --skip-futu
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader_v4 import ENHANCED_V4_PARQUET, NEWS_GPU_V4_MODELS_DIR, load_enhanced_v4
from src.trading_env_v4 import TradingEnv, news_obs_slice
from src.train_gpu_v3 import NEWS_GPU_V3_1_MODELS_DIR, NEWS_GPU_V3_MODELS_DIR
import src.train_gpu_v3 as v3
from src.utils import MODELS_DIR, NEWS_GPU_V2_MODELS_DIR, setup_logging

logger = setup_logging("airaire.train_gpu_v4")

_V4_FORBIDDEN = frozenset(
    {
        NEWS_GPU_V2_MODELS_DIR.resolve(),
        NEWS_GPU_V3_MODELS_DIR.resolve(),
        NEWS_GPU_V3_1_MODELS_DIR.resolve(),
        (MODELS_DIR / "news_gpu_v3_2").resolve(),
        (MODELS_DIR / "news").resolve(),
        (MODELS_DIR / "news_gpu").resolve(),
    }
)


def _guard_output(path: Path, *, slice_run: bool = False) -> Path:
    resolved = Path(path).resolve()
    if resolved in _V4_FORBIDDEN:
        raise ValueError(f"V4 refuses to write to {path}. Use models/news_gpu_v4.")
    return Path(path)


def _guard_init_checkpoint(path: Path | None) -> Path | None:
    if path is None:
        return None
    text = str(Path(path)).replace("\\", "/")
    if "news_gpu_v2" in text:
        raise ValueError("V4 cannot warm-start from a V2 zip (782 vs 1082).")
    if "news_gpu_v3" in text:
        raise ValueError(
            "V4 cannot warm-start from a V3 zip. V3 learned TradingView volume; "
            "V4 volume is always 0. Train V4 from scratch."
        )
    return Path(path)


def _load_enhanced_v4_for_train(force_news_fetch: bool = False, **_kwargs):
    return v3._remember_panel(
        load_enhanced_v4(
            save=True,
            fetch_futu=not v3._SKIP_FUTU,
            force_rebuild=v3._FORCE_REBUILD or force_news_fetch,
        )
    )


def _load_processed_v4():
    return v3._remember_panel(load_enhanced_v4(save=True, fetch_futu=not v3._SKIP_FUTU, force_rebuild=v3._FORCE_REBUILD))


def _panel_for_holdout_v4():
    if v3._FULL_PANEL is not None and not v3._FULL_PANEL.empty:
        return v3._FULL_PANEL
    if ENHANCED_V4_PARQUET.exists():
        import pandas as pd

        return pd.read_parquet(ENHANCED_V4_PARQUET)
    return None


def _patch() -> None:
    v3.ENHANCED_V3_PARQUET = ENHANCED_V4_PARQUET
    v3.NEWS_GPU_V3_MODELS_DIR = NEWS_GPU_V4_MODELS_DIR
    v3.NEWS_GPU_V3_1_MODELS_DIR = NEWS_GPU_V4_MODELS_DIR
    v3.TradingEnv = TradingEnv
    v3.news_obs_slice = news_obs_slice
    v3.load_enhanced_v3 = load_enhanced_v4
    v3._guard_output = _guard_output
    v3._guard_init_checkpoint = _guard_init_checkpoint
    v3._load_enhanced_for_v2_train = _load_enhanced_v4_for_train
    v3._load_processed_v3 = _load_processed_v4
    v3._panel_for_holdout = _panel_for_holdout_v4


def main(argv: list[str] | None = None) -> None:
    _patch()
    args = v3.parse_args(argv)
    if args.output is None:
        args.output = NEWS_GPU_V4_MODELS_DIR
    logger.info("GPU v4  panel=%s  output default=%s  Bloomberg volume=0  observers=HSI+SPX", ENHANCED_V4_PARQUET, args.output)
    v3.train(
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
        panel_start=args.start,
        panel_end=args.end,
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
