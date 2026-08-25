"""V2.11 shared helpers — HK-long / US-short hybrid, same 782-dim V2 zip family.

Live paper stays on ``models/news_gpu_v2``. This fork never writes there.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from src.trading_env import N_CORE, observation_dim
from src.utils import (
    CORE_TICKERS,
    HK_TZ,
    MODELS_DIR,
    NEWS_GPU_V2_11_MODELS_DIR,
    NEWS_GPU_V2_MODELS_DIR,
    PROJECT_ROOT,
    setup_logging,
    ticker_market,
)

logger = setup_logging("airaire.v2_11")

STATE_V2_11_PKL = PROJECT_ROOT / "state_v2_11.pkl"
V2_11_OBS_DIM = observation_dim()
FORBIDDEN_OUTPUT_NAMES = frozenset(
    {
        "news_gpu_v2",
        "news_gpu_v2_10",
        "news_gpu_v3",
        "news_gpu_v3_1",
        "news_gpu_v3_2",
        "news_gpu_v4",
        "news_gpu_v4_1",
        "news",
        "news_gpu",
    }
)
# Wrong obs/action families. V2 paper zips (782, Box[-1,1]) are the intended seed.
BLOCKED_INIT_PARTS = frozenset(
    {
        "news_gpu_v2_10",
        "news_gpu_v3",
        "news_gpu_v3_1",
        "news_gpu_v3_2",
        "news_gpu_v4",
        "news_gpu_v4_1",
        "news",
        "news_gpu",
    }
)
# Never guess this filename. It is not the GPU trader banner zip.
FORBIDDEN_SEED_NAMES = frozenset({"checkpoint_2026-08-20.zip"})


def clip_hybrid_action(action: np.ndarray, tickers: list[str] | None = None) -> np.ndarray:
    """Keep the V2 Box[-1,1] head; HK names cannot go below 0. US unchanged."""
    names = list(tickers) if tickers is not None else list(CORE_TICKERS)
    out = np.nan_to_num(np.asarray(action, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if out.shape[0] != len(names):
        raise ValueError(f"Expected action of shape ({len(names)},), got {out.shape}")
    out = np.clip(out, -1.0, 1.0)
    for i, ticker in enumerate(names):
        if str(ticker).startswith("HK."):
            out[i] = max(0.0, float(out[i]))
    return out


def clip_hk_holdings(holdings: np.ndarray, tickers: list[str] | None = None) -> np.ndarray:
    names = list(tickers) if tickers is not None else list(CORE_TICKERS)
    out = np.asarray(holdings, dtype=np.float64).reshape(-1).copy()
    for i, ticker in enumerate(names):
        if i >= len(out):
            break
        if str(ticker).startswith("HK."):
            out[i] = max(0.0, float(out[i]))
    return out


def resolve_v2_paper_seed_zip(*, explicit: Path | None = None) -> Path:
    """Zip the GPU ``run_trader.bat`` banner loads. Never ``checkpoint_2026-08-20.zip``."""
    if explicit is not None:
        path = Path(explicit)
        if path.name in FORBIDDEN_SEED_NAMES:
            raise ValueError(
                f"Refusing {path.name}. That is not the GPU trader banner zip. "
                "Use models/news_gpu_v2/best_model.zip (or the 20260823 dump of it)."
            )
        if not path.exists():
            raise FileNotFoundError(f"V2.11 seed zip not found: {path}")
        return path

    live = NEWS_GPU_V2_MODELS_DIR / "best_model.zip"
    dump = MODELS_DIR / "news_gpu_v2_20260823135426" / "best_model.zip"
    if live.exists():
        return live
    if dump.exists():
        logger.warning(
            "models/news_gpu_v2/best_model.zip is missing on this box. "
            "Using the VM dump %s (live_best source_zip=finetuned_2026-08-22.zip, W121). "
            "Confirm this path/size against the GPU trader startup banner.",
            dump,
        )
        return dump
    raise FileNotFoundError(
        "No V2 paper zip. Expected models/news_gpu_v2/best_model.zip "
        "(what run_trader.bat loads) or models/news_gpu_v2_20260823135426/best_model.zip. "
        "Do not seed from checkpoint_2026-08-20.zip."
    )


def log_seed_banner(seed: Path, *, role: str = "V2.11 warm-start") -> None:
    resolved = Path(seed).resolve()
    exists = resolved.exists()
    size = f"{resolved.stat().st_size / (1024 * 1024):.2f} MB" if exists else ""
    logger.info("============================================================")
    logger.info("AirAire %s — seed zip (confirm vs GPU run_trader.bat banner)", role)
    logger.info("  path   : %s", resolved)
    logger.info("  exists : %s%s", exists, f"  ({size})" if size else "")
    logger.info("  role   : V2 paper brain (782-dim, Box[-1,1]). Not checkpoint_2026-08-20.zip.")
    logger.info("============================================================")


def ensure_v2_11_dir() -> Path:
    NEWS_GPU_V2_11_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return NEWS_GPU_V2_11_MODELS_DIR


def install_seed_into_v2_11(seed: Path | None = None) -> Path:
    """Copy the V2 paper zip into the isolated dir. Does not overwrite a later v2.11 best_model."""
    output = ensure_v2_11_dir()
    src = resolve_v2_paper_seed_zip(explicit=seed)
    log_seed_banner(src)
    dest_seed = output / "seed_from_v2_paper.zip"
    if not dest_seed.exists() or dest_seed.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dest_seed)
        logger.info("Copied V2 paper seed -> %s", dest_seed)
    dest_best = output / "best_model.zip"
    if not dest_best.exists():
        shutil.copy2(src, dest_best)
        logger.info("Installed V2.11 best_model.zip from seed (no v2.11 fine-tune yet).")
    manifest = {
        "seed_src": str(src.resolve()),
        "seed_name": src.name,
        "copied_at": datetime.now(tz=HK_TZ).isoformat(),
        "note": (
            "Dump/live V2 best_model.zip. Confirm against the GPU trader banner. "
            "Not models/old/news_gpu_v2_test/checkpoint_2026-08-20.zip."
        ),
        "obs_dim": V2_11_OBS_DIM,
        "n_core": N_CORE,
        "action": "Box(-1,1) with HK clipped to [0,1] in env step/_rebalance",
    }
    (output / "seed_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dest_seed


def guard_output_dir(path: Path | None) -> Path:
    output = Path(path) if path is not None else NEWS_GPU_V2_11_MODELS_DIR
    resolved = output.resolve()
    if resolved == NEWS_GPU_V2_MODELS_DIR.resolve():
        raise ValueError("V2.11 refuses to write models/news_gpu_v2 (live paper). Use models/news_gpu_v2_11.")
    if resolved.name in FORBIDDEN_OUTPUT_NAMES:
        raise ValueError(f"V2.11 refuses to write {output}. Use models/news_gpu_v2_11.")
    return output


def guard_init_checkpoint(path: Path | None) -> Path | None:
    if path is None:
        return None
    raw = Path(path)
    if raw.name in FORBIDDEN_SEED_NAMES:
        raise ValueError(
            f"V2.11 refuses {raw.name}. Seed from the GPU trader banner zip "
            "(models/news_gpu_v2/best_model.zip), not a guessed 2026-08-20 checkpoint."
        )
    parts = {str(p) for p in raw.resolve().parts}
    if parts & BLOCKED_INIT_PARTS:
        raise ValueError(
            f"V2.11 cannot load {raw} (wrong obs/action family). "
            "Warm-start from a V2 782-dim Box[-1,1] zip or a v2.11 zip."
        )
    if not raw.exists():
        raise FileNotFoundError(f"--init-checkpoint not found: {raw}")
    return raw


def refuse_wrong_inference_zip(path: Path) -> None:
    raw = Path(path)
    if raw.name in FORBIDDEN_SEED_NAMES:
        raise ValueError(
            f"Refusing {raw.name}. Seed from the GPU trader banner zip, not a guessed 2026-08-20 checkpoint."
        )
    parts = {str(p) for p in raw.resolve().parts}
    if "news_gpu_v2_11" in parts or raw.name == "seed_from_v2_paper.zip":
        return
    if "news_gpu_v2_10" in parts:
        raise ValueError(f"inference_v2_11 refuses {path} (V2.10 932-dim long-only-all-five).")
    blocked = BLOCKED_INIT_PARTS - {"news_gpu_v2"}
    # news_gpu_v2 and the dated dump are allowed (same 782-dim family).
    if parts & blocked:
        raise ValueError(f"inference_v2_11 refuses {path} (wrong family).")


def hk_us_split_ok() -> None:
    """Fail fast if CORE_TICKERS order changes under the hybrid clip."""
    markets = [ticker_market(t) for t in CORE_TICKERS]
    if markets[:3] != ["HK", "HK", "HK"] or markets[3:] != ["US", "US"]:
        raise RuntimeError(f"V2.11 expects 3 HK + 2 US CORE_TICKERS, got {CORE_TICKERS}")
