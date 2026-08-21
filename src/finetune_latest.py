"""Lightweight daily PPO fine-tune from the latest healthy GPU v2 checkpoint.

Do **not** retrain from Window 90 (entropy-collapsed history). Always warm-start
from the newest ``checkpoint_*.zip`` / ``finetuned_*.zip`` in
``models/news_gpu_v2`` — typically ``checkpoint_2026-08-18.zip`` (resurrected
Window 118, Calmar 1.83) until a later fine-tune exists.

``best_model.zip`` is the paper-trading brain. Fine-tune never copies it unless
you press **Promote** on Telegram (or run ``--promote-zip``). The comparison bar
is ``live_best.json`` (starts as Window 113 / Calmar ~2.05, then follows whatever
you promoted).

Typical GPU VM runtime: ~2-3 minutes for 1 window (8 PPO updates, same
hyperparameters as ``train_gpu_v2``), plus a short Alpha Vantage refresh of
the latest ~30 days (education 75/min) before PPO starts.

    python -m src.finetune_latest
    python -m src.finetune_latest --windows 3
    python -m src.finetune_latest --promote-wait 0
    python -m src.finetune_latest --promote-zip models/news_gpu_v2/finetuned_2026-08-21.zip
    python -m src.finetune_latest --skip-news
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from datetime import datetime
from multiprocessing import freeze_support
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from stable_baselines3 import PPO

from src.data_loader import (
    default_futu_fetch_start,
    fetch_futu_history,
    overlay_live_ohlcv,
    persist_enhanced_panel,
)
from src.train import (
    DEFAULT_WINDOW_DAYS,
    WindowMetrics,
    _news_coverage,
    _sanitize_panel,
    _window_news,
    calmar_ratio,
    evaluate_policy,
    iter_windows,
    policy_has_nan,
    resolve_device,
)
from src.train_gpu_v2 import (
    DESIRED_PPO_UPDATES,
    GpuPpoConfig,
    _close_env,
    _configure_cuda,
    _learn_window,
    _policy_looks_collapsed,
    _ppo_train_stats,
    _start_run_log,
    timesteps_for_window,
)
from src.utils import (
    CORE_TICKERS,
    ENHANCED_PARQUET,
    HK_TZ,
    NEWS_GPU_V2_MODELS_DIR,
    PROTECTED_INFERENCE_ZIPS,
    TELEGRAM_CALLBACK_KEEP,
    TELEGRAM_CALLBACK_PROMOTE,
    send_telegram_alert,
    setup_logging,
    telegram_auth,
    wait_telegram_callback,
)

logger = setup_logging("airaire.finetune_latest")

# Golden artifacts from the 2026-08-20 resurrection. Never overwrite these.
PROTECTED_ZIPS = PROTECTED_INFERENCE_ZIPS
_DATED_ZIP_RE = re.compile(r"^(checkpoint|finetuned)_(\d{4}-\d{2}-\d{2})\.zip$")
MIN_WINDOWS = 1
MAX_WINDOWS = 3

# Promotion baselines from the 2026-08-20 resurrection (training_log_history.csv).
# Compared against the *live* paper-trading brain (live_best.json), which starts
# as Window 113 / Calmar ~2.05 and moves when you Promote on Telegram (or --promote-zip).
TRADING_GOLDEN_WINDOW = 113
TRADING_GOLDEN_END = "2026-08-12"
TRADING_GOLDEN_CALMAR = 2.053856720691549  # ~2.05, paper-trading brain
TRADING_GOLDEN_ZIP = "best_model.zip"

TRAINING_SEED_WINDOW = 118
TRAINING_SEED_END = "2026-08-18"
TRAINING_SEED_CALMAR = 1.832871817457733  # continue-training seed
TRAINING_SEED_ZIP = "checkpoint_2026-08-18.zip"

FINETUNE_LOG_NAME = "finetune_log.csv"
LIVE_BEST_JSON = "live_best.json"
PINNED_ROLES = ("trading_golden", "training_seed", "live_best")
DEFAULT_PROMOTE_WAIT_SECONDS = 600
FINETUNE_LOG_COLUMNS = [
    "run_id",
    "role",
    "window",
    "start",
    "end",
    "n_bars",
    "timesteps",
    "cumulative_return",
    "sharpe",
    "max_drawdown",
    "final_equity",
    "news_coverage",
    "calmar",
    "approx_kl",
    "entropy_loss",
    "ppo_updates",
    "zip",
    "beats_trading_golden",  # same flag as beats_live_best; name kept for existing CSVs
    "beats_live_best",
]


def _hk_today() -> datetime:
    return datetime.now(tz=HK_TZ)


def resolve_latest_checkpoint(output_dir: Path, explicit: Path | None = None) -> Path:
    """Newest dated zip in ``output_dir``. Prefers ``finetuned_`` over ``checkpoint_`` on the same day.

    ``best_model.zip`` is ignored on purpose — that is the trading snapshot, not
    necessarily the most recent weights.
    """
    if explicit is not None:
        ckpt = Path(explicit)
        if not ckpt.exists():
            raise FileNotFoundError(f"--checkpoint not found: {ckpt}")
        logger.info("Using explicit fine-tune seed %s", ckpt)
        return ckpt

    ranked: list[tuple[str, int, float, Path]] = []
    for path in output_dir.glob("*.zip"):
        match = _DATED_ZIP_RE.match(path.name)
        if not match:
            continue
        kind, day = match.group(1), match.group(2)
        # finetuned_ on the same date is a later training step than checkpoint_.
        kind_rank = 1 if kind == "finetuned" else 0
        ranked.append((day, kind_rank, path.stat().st_mtime, path))

    if not ranked:
        raise FileNotFoundError(
            f"No checkpoint_YYYY-MM-DD.zip or finetuned_YYYY-MM-DD.zip in {output_dir}. "
            "Expected checkpoint_2026-08-18.zip (Window 118). "
            "Do not train from the Window-90 entropy-collapse era."
        )
    ranked.sort()
    chosen = ranked[-1][-1]
    logger.info("Latest fine-tune seed: %s", chosen)
    return chosen


def _merge_recent_news(panel: pd.DataFrame, news: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Re-score bars at/after ``cutoff``. Older ``news_score`` values stay as-is."""
    from src.data_loader import merge_price_news

    cutoff = pd.Timestamp(cutoff)
    older = panel.loc[pd.to_datetime(panel["datetime"]) < cutoff].copy()
    recent = panel.loc[pd.to_datetime(panel["datetime"]) >= cutoff].copy()
    if recent.empty:
        return panel
    recent = merge_price_news(recent.drop(columns=["news_score"], errors="ignore"), news)
    if older.empty:
        return _sanitize_panel(recent)
    return _sanitize_panel(pd.concat([older, recent], ignore_index=True))


def load_panel(
    *,
    refresh_futu: bool,
    lookback_days: int,
    force_news_fetch: bool,
    skip_news: bool = False,
    news_days: int | None = None,
) -> pd.DataFrame:
    """Load enhanced parquet, overlay Futu bars, then always refresh recent Alpha Vantage news."""
    if not ENHANCED_PARQUET.exists():
        raise FileNotFoundError(
            f"{ENHANCED_PARQUET} is missing. Build it once with `python -m src.data_loader` "
            "before daily fine-tunes."
        )
    panel = _sanitize_panel(pd.read_parquet(ENHANCED_PARQUET))
    logger.info(
        "Loaded %s rows=%d span=%s → %s",
        ENHANCED_PARQUET,
        len(panel),
        panel["datetime"].min() if not panel.empty else None,
        panel["datetime"].max() if not panel.empty else None,
    )

    now = _hk_today()
    if refresh_futu:
        start = default_futu_fetch_start(panel, now=now, lookback_days=lookback_days)
        logger.info("Refreshing Futu 10-min bars from %s to %s (offline-gap fill).", start.date(), now.date())
        try:
            live = fetch_futu_history(CORE_TICKERS, start=start, end=now, lookback_days=lookback_days)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Futu refresh failed (%s). Fine-tune continues; news refresh still runs.", exc)
            live = None
        if live is None or live.empty:
            logger.info("No new Futu bars (OpenD down or already current).")
        else:
            old_max = pd.Timestamp(panel["datetime"].max()) if not panel.empty else None
            panel = _sanitize_panel(overlay_live_ohlcv(panel, live))
            new_max = pd.Timestamp(panel["datetime"].max()) if not panel.empty else None
            logger.info("Panel after Futu overlay: last bar %s → %s", old_max, new_max)

    if skip_news:
        persist_enhanced_panel(panel)
        return panel

    news_end = pd.Timestamp(now.replace(tzinfo=None)) if getattr(now, "tzinfo", None) else pd.Timestamp(now)
    if not panel.empty:
        news_end = max(news_end, pd.Timestamp(panel["datetime"].max()))
    span = max(int(news_days if news_days is not None else lookback_days), 7)
    news_start = news_end - pd.Timedelta(days=span)
    logger.info(
        "Refreshing Alpha Vantage NEWS_SENTIMENT %s → %s (force_fetch=%s, education 75/min).",
        news_start.date(),
        news_end.date(),
        force_news_fetch,
    )
    try:
        from src.news_loader import load_all_news

        news = load_all_news(news_start, news_end, force_fetch=force_news_fetch)
        if news is not None and not news.empty:
            panel = _merge_recent_news(panel, news, news_start)
        else:
            logger.warning("News refresh returned empty. Keeping existing news_score (ffill on new Futu bars).")
    except Exception as exc:  # noqa: BLE001 — sentiment is optional if AV is down
        logger.warning("News refresh failed (%s). Forward-filled scores kept.", exc)

    persist_enhanced_panel(panel)
    return panel


def latest_windows(panel: pd.DataFrame, window_days: int, n_windows: int):
    """Return the last ``n_windows`` rolling 30-day slices (the frontier, never Window 90)."""
    windows = iter_windows(panel, window_days)
    n_windows = max(MIN_WINDOWS, min(int(n_windows), MAX_WINDOWS))
    chosen = windows[-n_windows:]
    logger.info(
        "Fine-tuning %d of %d rolling windows (last = %s → %s, index=%d). "
        "Window-90 collapse history is not replayed.",
        len(chosen),
        len(windows),
        chosen[0].start.date(),
        chosen[-1].end.date(),
        chosen[-1].index,
    )
    return chosen


def _safe_save(model: PPO, dest: Path) -> Path | None:
    dest = Path(dest)
    if dest.suffix != ".zip":
        dest = dest.with_suffix(".zip")
    if dest.name in PROTECTED_ZIPS:
        logger.warning("Refusing to overwrite golden artifact %s", dest.name)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    # SB3 appends .zip when the path has no suffix.
    model.save(str(dest.with_suffix("")))
    saved = dest.with_suffix(".zip")
    if not saved.exists():
        logger.error("model.save did not produce %s", saved)
        return None
    logger.info("Saved %s", saved)
    return saved


def _history_lookup(output_dir: Path, window: int, end: str) -> dict | None:
    """Pull a golden row from training_log_history.csv if that file is present."""
    path = output_dir / "training_log_history.csv"
    if not path.exists():
        return None
    try:
        hist = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s (%s). Using hardcoded golden Calmars.", path, exc)
        return None
    if hist.empty or "window" not in hist.columns:
        return None
    hit = hist.loc[hist["window"].astype(int) == int(window)]
    if "end" in hist.columns:
        by_end = hit.loc[hit["end"].astype(str) == str(end)]
        if not by_end.empty:
            hit = by_end
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def load_reference_calmars(output_dir: Path) -> tuple[float, float]:
    """Window 113 / 118 Calmars: history CSV if present, else the resurrection constants."""
    w113 = _history_lookup(output_dir, TRADING_GOLDEN_WINDOW, TRADING_GOLDEN_END)
    w118 = _history_lookup(output_dir, TRAINING_SEED_WINDOW, TRAINING_SEED_END)
    trading = float(w113["calmar"]) if w113 and "calmar" in w113 else TRADING_GOLDEN_CALMAR
    seed = float(w118["calmar"]) if w118 and "calmar" in w118 else TRAINING_SEED_CALMAR
    if not math.isfinite(trading):
        trading = TRADING_GOLDEN_CALMAR
    if not math.isfinite(seed):
        seed = TRAINING_SEED_CALMAR
    return trading, seed


def _default_live_best() -> dict:
    return {
        "window": TRADING_GOLDEN_WINDOW,
        "start": "2026-07-09",
        "end": TRADING_GOLDEN_END,
        "calmar": TRADING_GOLDEN_CALMAR,
        "zip": TRADING_GOLDEN_ZIP,
        "source_zip": "checkpoint_2026-08-12.zip",
        "updated_at": "",
    }


def load_live_best(output_dir: Path) -> dict:
    """Calmar bar for promotion: whatever is currently in best_model.zip.

    Seeded from Window 113 on first run. Updated only after an explicit Promote
    (Telegram button or ``--promote-zip``).
    """
    path = output_dir / LIVE_BEST_JSON
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            calmar = float(data.get("calmar", TRADING_GOLDEN_CALMAR))
            if math.isfinite(calmar):
                data["calmar"] = calmar
                return data
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Could not read %s (%s). Falling back to Window 113.", path, exc)
    live = _default_live_best()
    hist = _history_lookup(output_dir, TRADING_GOLDEN_WINDOW, TRADING_GOLDEN_END)
    if hist:
        if "calmar" in hist and _finite(hist["calmar"]):
            live["calmar"] = float(hist["calmar"])
        if "start" in hist and pd.notna(hist["start"]):
            live["start"] = str(hist["start"])
        if "end" in hist and pd.notna(hist["end"]):
            live["end"] = str(hist["end"])
        if "window" in hist and _finite(hist["window"]):
            live["window"] = int(hist["window"])
    save_live_best(output_dir, live)
    return live


def save_live_best(output_dir: Path, live: dict) -> Path:
    path = output_dir / LIVE_BEST_JSON
    payload = dict(live)
    payload["updated_at"] = datetime.now(tz=HK_TZ).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Live best bar: W%s  %s  Calmar %.4f  (source %s)",
        payload.get("window"),
        payload.get("end"),
        float(payload.get("calmar", float("nan"))),
        payload.get("source_zip") or payload.get("zip"),
    )
    return path


def _live_log_row(live: dict) -> dict:
    row = {col: None for col in FINETUNE_LOG_COLUMNS}
    row.update(
        {
            "run_id": str(live.get("updated_at") or "live"),
            "role": "live_best",
            "window": live.get("window"),
            "start": live.get("start"),
            "end": live.get("end"),
            "calmar": live.get("calmar"),
            "zip": TRADING_GOLDEN_ZIP,
            "beats_trading_golden": None,
            "beats_live_best": None,
        }
    )
    return row


def upsert_live_best_row(output_dir: Path, live: dict) -> None:
    """Replace the pinned ``live_best`` row; leave W113/W118 and daily rows alone."""
    path = output_dir / FINETUNE_LOG_NAME
    if not path.exists():
        return
    existing = pd.read_csv(path).reindex(columns=FINETUNE_LOG_COLUMNS)
    if "role" not in existing.columns:
        return
    pinned_hist = existing.loc[existing["role"].isin(["trading_golden", "training_seed"])]
    rest = existing.loc[~existing["role"].isin(PINNED_ROLES)]
    live_df = pd.DataFrame([_live_log_row(live)]).reindex(columns=FINETUNE_LOG_COLUMNS)
    combined = pd.concat([pinned_hist, live_df, rest], ignore_index=True)
    for col in ("window", "n_bars", "timesteps", "ppo_updates"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("Int64")
    combined.to_csv(path, index=False)


def apply_promotion(
    output_dir: Path,
    zip_path: Path,
    *,
    row: WindowMetrics | None,
    live_before: dict | None = None,
) -> dict:
    """Copy ``zip_path`` onto ``best_model.zip`` and raise the live Calmar bar."""
    src = Path(zip_path)
    if not src.exists():
        raise FileNotFoundError(f"Cannot promote missing zip: {src}")
    dest = output_dir / TRADING_GOLDEN_ZIP
    shutil.copy2(src, dest)
    live = {
        "window": int(row.window) if row is not None else None,
        "start": str(row.start) if row is not None else "",
        "end": str(row.end) if row is not None else "",
        "calmar": float(row.calmar) if row is not None and _finite(row.calmar) else float("nan"),
        "zip": TRADING_GOLDEN_ZIP,
        "source_zip": src.name,
    }
    save_live_best(output_dir, live)
    upsert_live_best_row(output_dir, load_live_best(output_dir))
    logger.info("Promoted %s -> %s. Next fine-tune compares against Calmar %.4f.", src.name, dest, live["calmar"])
    before = float((live_before or {}).get("calmar", TRADING_GOLDEN_CALMAR))
    send_telegram_alert(
        "AirAire: promoted to best_model.zip\n"
        f"Source: {src.name}\n"
        f"New live Calmar: {live['calmar']:.4f} (was {before:.4f})\n"
        "Paper trading will load this zip on the next inference start."
    )
    return live


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _golden_log_row(
    *,
    role: str,
    window: int,
    start: str,
    end: str,
    calmar: float,
    zip_name: str,
    history: dict | None,
) -> dict:
    row = {col: None for col in FINETUNE_LOG_COLUMNS}
    row.update(
        {
            "run_id": "resurrection_20260820",
            "role": role,
            "window": window,
            "start": start,
            "end": end,
            "calmar": calmar,
            "zip": zip_name,
            "beats_trading_golden": None,
            "beats_live_best": None,
        }
    )
    if history:
        for key in (
            "n_bars",
            "timesteps",
            "cumulative_return",
            "sharpe",
            "max_drawdown",
            "final_equity",
            "news_coverage",
            "approx_kl",
            "entropy_loss",
            "ppo_updates",
            "start",
        ):
            if key in history and pd.notna(history[key]):
                row[key] = history[key]
        if "calmar" in history and pd.notna(history["calmar"]):
            row["calmar"] = history["calmar"]
    return row


def _metrics_log_row(
    row: WindowMetrics,
    *,
    run_id: str,
    zip_name: str,
    live_calmar: float,
) -> dict:
    beats = bool(_finite(row.calmar) and float(row.calmar) > float(live_calmar))
    payload = {col: "" for col in FINETUNE_LOG_COLUMNS}
    payload.update({k: v for k, v in row.__dict__.items() if k in payload})
    payload.update(
        {
            "run_id": run_id,
            "role": "finetune",
            "zip": zip_name,
            "beats_trading_golden": beats,
            "beats_live_best": beats,
        }
    )
    return payload


def append_finetune_log(
    output_dir: Path,
    new_rows: list[dict],
    *,
    w113_calmar: float,
    w118_calmar: float,
    live: dict,
) -> Path:
    """Append today's rows.

    Pinned at the top (never shuffled into history):
      1. trading_golden — original Window 113 (museum)
      2. training_seed  — original Window 118
      3. live_best      — whatever best_model.zip currently is (moves on Promote)
    """
    path = output_dir / FINETUNE_LOG_NAME
    hist_113 = _history_lookup(output_dir, TRADING_GOLDEN_WINDOW, TRADING_GOLDEN_END)
    hist_118 = _history_lookup(output_dir, TRAINING_SEED_WINDOW, TRAINING_SEED_END)
    history_pins = [
        _golden_log_row(
            role="trading_golden",
            window=TRADING_GOLDEN_WINDOW,
            start="2026-07-09",
            end=TRADING_GOLDEN_END,
            calmar=w113_calmar,
            zip_name="checkpoint_2026-08-12.zip",
            history=hist_113,
        ),
        _golden_log_row(
            role="training_seed",
            window=TRAINING_SEED_WINDOW,
            start="2026-07-15",
            end=TRAINING_SEED_END,
            calmar=w118_calmar,
            zip_name=TRAINING_SEED_ZIP,
            history=hist_118,
        ),
    ]
    incoming = pd.DataFrame(new_rows).reindex(columns=FINETUNE_LOG_COLUMNS)

    if path.exists():
        existing = pd.read_csv(path)
        if "role" not in existing.columns:
            existing["role"] = "finetune"
        existing = existing.reindex(columns=FINETUNE_LOG_COLUMNS)
        hist_df = existing.loc[existing["role"].isin(["trading_golden", "training_seed"])]
        live_df = existing.loc[existing["role"] == "live_best"]
        rest = existing.loc[~existing["role"].isin(PINNED_ROLES)]
        if hist_df.empty:
            hist_df = pd.DataFrame(history_pins).reindex(columns=FINETUNE_LOG_COLUMNS)
        if live_df.empty:
            live_df = pd.DataFrame([_live_log_row(live)]).reindex(columns=FINETUNE_LOG_COLUMNS)
        if not incoming.empty and not rest.empty:
            key_cols = [c for c in ("run_id", "window", "start", "end") if c in rest.columns]
            if key_cols:
                seen = set(zip(*(rest[c].astype(str) for c in key_cols)))
                mask = [
                    tuple(str(v) for v in rec) not in seen
                    for rec in zip(*(incoming[c].astype(str) for c in key_cols))
                ]
                incoming = incoming.loc[mask]
        combined = pd.concat([hist_df, live_df, rest, incoming], ignore_index=True)
    else:
        combined = pd.concat(
            [
                pd.DataFrame(history_pins).reindex(columns=FINETUNE_LOG_COLUMNS),
                pd.DataFrame([_live_log_row(live)]).reindex(columns=FINETUNE_LOG_COLUMNS),
                incoming,
            ],
            ignore_index=True,
        )

    # Keep window ids as ints in the CSV (113 not 113.0).
    for col in ("window", "n_bars", "timesteps", "ppo_updates"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("Int64")

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    logger.info(
        "Appended %d fine-tune row(s) -> %s (file rows=%d; W113/W118 history + live_best pinned)",
        len(incoming),
        path,
        len(combined),
    )
    return path


def report_promotion(
    *,
    output_dir: Path,
    row: WindowMetrics | None,
    zip_path: Path | None,
    live: dict,
    w113_calmar: float,
    w118_calmar: float,
    notify: bool,
    promote_wait_seconds: int,
) -> None:
    """Terminal scoreboard vs the *live* best_model, then optional Telegram Promote/Keep.

    Window 113/118 stay in the log as history. The hurdle is ``live['calmar']``.
    ``best_model.zip`` is copied only if you press Promote (or pass ``--promote-zip`` later).
    """
    live_calmar = float(live.get("calmar", TRADING_GOLDEN_CALMAR))
    new_calmar = float(row.calmar) if row is not None and _finite(row.calmar) else float("nan")
    beats = bool(_finite(new_calmar) and new_calmar > live_calmar)
    zip_name = zip_path.name if zip_path is not None else "(not saved)"
    win_label = f"W{row.window}  {row.start} → {row.end}" if row is not None else "n/a"
    live_label = f"W{live.get('window')}  end {live.get('end')}"

    logger.info("============================================================")
    logger.info("PROMOTION CHECK  (best_model.zip changes only if you Promote)")
    logger.info(
        "  History W113     end %s  Calmar %.4f  (original champion, museum)",
        TRADING_GOLDEN_END,
        w113_calmar,
    )
    logger.info(
        "  History W118     end %s  Calmar %.4f  (training seed)",
        TRAINING_SEED_END,
        w118_calmar,
    )
    logger.info(
        "  LIVE best_model  %s  Calmar %.4f  (%s)",
        live_label,
        live_calmar,
        live.get("source_zip") or live.get("zip"),
    )
    if row is None:
        logger.info("  This fine-tune  %s  Calmar n/a  (%s)", win_label, zip_name)
        logger.info("  Verdict: no usable window. Keep live best_model.zip.")
    else:
        logger.info("  This fine-tune  %s  Calmar %.4f  (%s)", win_label, new_calmar, zip_name)
        if beats:
            logger.info(
                "  Verdict: BEATS live best (%.4f > %.4f). Waiting for Telegram Promote / Keep.",
                new_calmar,
                live_calmar,
            )
        else:
            logger.info(
                "  Verdict: KEEP live best_model.zip (new %.4f is not above %.4f). No Telegram.",
                new_calmar,
                live_calmar,
            )
    logger.info("============================================================")

    if not beats or zip_path is None or row is None:
        return
    cli = f"python -m src.finetune_latest --promote-zip {zip_path}"
    if not notify:
        logger.info("Telegram skipped (--no-telegram). Later: %s", cli)
        return
    if telegram_auth() is None:
        logger.warning("Telegram unset. Live best unchanged. Later: %s", cli)
        return
    if promote_wait_seconds <= 0:
        send_telegram_alert(
            "AirAire fine-tune beat the live best_model\n"
            f"New: W{row.window}  {row.start} → {row.end}  Calmar {new_calmar:.4f}\n"
            f"Live: {live_label}  Calmar {live_calmar:.4f}\n"
            f"Zip: {zip_path.name}\n"
            "best_model.zip was NOT changed (wait disabled).\n"
            f"To promote: {cli}"
        )
        return

    markup = {
        "inline_keyboard": [
            [
                {"text": "Promote to best_model", "callback_data": TELEGRAM_CALLBACK_PROMOTE},
                {"text": "Keep current", "callback_data": TELEGRAM_CALLBACK_KEEP},
            ]
        ]
    }
    sent = send_telegram_alert(
        "AirAire fine-tune beat the live best_model\n"
        f"New: W{row.window}  {row.start} → {row.end}  Calmar {new_calmar:.4f}\n"
        f"Live: {live_label}  Calmar {live_calmar:.4f}\n"
        f"Zip: {zip_path.name}\n"
        f"Press Promote to copy this zip onto best_model.zip and raise the bar.\n"
        f"Press Keep (or wait {promote_wait_seconds}s) to leave trading as-is.\n"
        f"Later: {cli}",
        reply_markup=markup,
    )
    if not sent:
        logger.warning("Telegram prompt failed. Live best unchanged. Later: %s", cli)
        return

    logger.info(
        "Waiting up to %ss for Telegram Promote / Keep (Ctrl+C keeps the current best_model)...",
        promote_wait_seconds,
    )
    try:
        choice = wait_telegram_callback(
            timeout_seconds=promote_wait_seconds,
            allowed=(TELEGRAM_CALLBACK_PROMOTE, TELEGRAM_CALLBACK_KEEP),
        )
    except KeyboardInterrupt:
        logger.info("Wait interrupted. Live best_model.zip unchanged.")
        send_telegram_alert("AirAire: promote wait cancelled. best_model.zip unchanged.")
        return

    if choice == TELEGRAM_CALLBACK_PROMOTE:
        apply_promotion(output_dir, zip_path, row=row, live_before=live)
    elif choice == TELEGRAM_CALLBACK_KEEP:
        logger.info("Telegram Keep: live best_model.zip unchanged.")
        send_telegram_alert("AirAire: Keep received. best_model.zip unchanged.")
    else:
        logger.info("No Telegram answer in %ss. Live best unchanged. Later: %s", promote_wait_seconds, cli)
        send_telegram_alert(
            f"AirAire: no Promote/Keep in {promote_wait_seconds}s. best_model.zip unchanged.\n{cli}"
        )


def finetune(
    *,
    n_windows: int = 1,
    window_days: int = DEFAULT_WINDOW_DAYS,
    output: Path | None = None,
    checkpoint: Path | None = None,
    seed: int = 42,
    device: str = "cuda",
    refresh_futu: bool = True,
    lookback_days: int = 30,
    force_news_fetch: bool = True,
    skip_news: bool = False,
    news_days: int | None = None,
    notify_telegram: bool = True,
    promote_wait_seconds: int = DEFAULT_PROMOTE_WAIT_SECONDS,
) -> list[WindowMetrics]:
    if not MIN_WINDOWS <= int(n_windows) <= MAX_WINDOWS:
        raise ValueError(f"--windows must be {MIN_WINDOWS}–{MAX_WINDOWS} (got {n_windows}).")

    output_dir = Path(output) if output is not None else NEWS_GPU_V2_MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(device)
    _configure_cuda()
    cfg = GpuPpoConfig()
    today = _hk_today().date()
    log_dir = _ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_filename = log_dir / f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    restore_log, log_file = _start_run_log(log_filename)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        logger.info("Fine-tune log -> %s", log_filename)
        logger.info(
            "GPU v2 settings  n_steps=%d  n_envs=%d  batch=%d  updates/window=%d  "
            "ent_coef kept from checkpoint / make_ppo",
            cfg.n_steps,
            cfg.n_envs,
            cfg.aligned_batch(),
            DESIRED_PPO_UPDATES,
        )

        ckpt = resolve_latest_checkpoint(output_dir, checkpoint)
        panel = load_panel(
            refresh_futu=refresh_futu,
            lookback_days=lookback_days,
            force_news_fetch=force_news_fetch,
            skip_news=skip_news,
            news_days=news_days,
        )
        if panel.empty:
            raise FileNotFoundError("Enhanced panel is empty after Futu overlay.")

        news_df = None
        if "news_score" in panel.columns:
            news_df = panel.loc[panel["ticker"].isin(CORE_TICKERS), ["datetime", "ticker", "news_score"]].rename(
                columns={"news_score": "sentiment_score"}
            )

        windows = latest_windows(panel, window_days, n_windows)
        model: PPO | None = None
        last_good: Path | None = ckpt if ckpt.exists() else None
        metrics: list[WindowMetrics] = []
        last_saved: Path | None = None
        last_saved_row: WindowMetrics | None = None

        for win in windows:
            n_bars = int(win.df["datetime"].nunique())
            steps = timesteps_for_window(win.df, epochs=1, cfg=cfg)
            window_news = _window_news(news_df, win.start, win.end)
            coverage = _news_coverage(win.df)
            logger.info(
                "Fine-tune window %d  %s → %s  bars=%d  timesteps=%d  news_coverage=%.1f%%  seed=%s",
                win.index,
                win.start.date(),
                win.end.date(),
                n_bars,
                steps,
                100.0 * coverage,
                ckpt.name if model is None else "in-memory",
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
                ckpt_to_load=ckpt if model is None else None,
                last_good_ckpt=last_good,
                log_file=log_file,
            )

            if policy_has_nan(model):
                logger.error("Window %d: NaN weights after learn(). Aborting save.", win.index)
                if last_good is not None and last_good.exists():
                    model = PPO.load(str(last_good), env=vec_env, device=device)
                _close_env(vec_env)
                continue

            stats = _ppo_train_stats(model)
            collapsed = _policy_looks_collapsed(stats)
            if stats:
                logger.info(
                    "Fine-tune PPO  approx_kl=%s  entropy_loss=%s%s",
                    f"{stats['approx_kl']:.4f}" if "approx_kl" in stats else "n/a",
                    f"{stats['entropy_loss']:.4f}" if "entropy_loss" in stats else "n/a",
                    "  COLLAPSED — not saving these weights" if collapsed else "",
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
                ppo_updates=max(steps // max(cfg.rollout_size(), 1), 1),
            )
            metrics.append(row)
            logger.info(
                "Fine-tune metrics  return=%.4f  sharpe=%.4f  max_dd=%.4f  calmar=%.4f  equity=%.2f",
                cum_ret,
                sharpe,
                max_dd,
                calmar,
                equity,
            )

            if collapsed:
                if last_good is not None and last_good.exists():
                    logger.warning("Reloading last healthy checkpoint %s", last_good)
                    model = PPO.load(str(last_good), env=vec_env, device=device)
                _close_env(vec_env)
                continue

            dated = output_dir / f"finetuned_{today}.zip"
            saved = _safe_save(model, dated)
            if saved is not None:
                last_saved = saved
                last_good = saved
                last_saved_row = row

            # Keep a window-end checkpoint for future --resume, but never clobber goldens.
            window_ckpt = output_dir / f"checkpoint_{win.end.date()}.zip"
            if window_ckpt.name not in PROTECTED_ZIPS:
                extra = _safe_save(model, window_ckpt)
                if extra is not None:
                    last_good = extra

            _close_env(vec_env)

        w113_calmar, w118_calmar = load_reference_calmars(output_dir)
        live = load_live_best(output_dir)
        live_calmar = float(live.get("calmar", TRADING_GOLDEN_CALMAR))
        log_rows = [
            _metrics_log_row(
                m,
                run_id=run_id,
                zip_name=last_saved.name if last_saved is not None else "",
                live_calmar=live_calmar,
            )
            for m in metrics
        ]
        append_finetune_log(
            output_dir,
            log_rows,
            w113_calmar=w113_calmar,
            w118_calmar=w118_calmar,
            live=live,
        )
        report_promotion(
            output_dir=output_dir,
            row=last_saved_row,
            zip_path=last_saved,
            live=live,
            w113_calmar=w113_calmar,
            w118_calmar=w118_calmar,
            notify=notify_telegram,
            promote_wait_seconds=promote_wait_seconds,
        )
        if last_saved is not None:
            logger.info(
                "Fine-tune complete. Paper-trading zip is %s. Candidate weights: %s",
                output_dir / "best_model.zip",
                last_saved,
            )
        else:
            logger.warning("Fine-tune finished with no new zip (collapse, NaN, or protected-path skip).")
        return metrics
    finally:
        restore_log()


def _row_from_finetune_log(output_dir: Path, zip_name: str) -> WindowMetrics | None:
    path = output_dir / FINETUNE_LOG_NAME
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "zip" not in df.columns:
        return None
    hit = df.loc[df["zip"].astype(str) == zip_name]
    if "role" in df.columns:
        fin = hit.loc[hit["role"].astype(str) == "finetune"]
        if not fin.empty:
            hit = fin
    if hit.empty:
        return None
    last = hit.iloc[-1]
    try:
        return WindowMetrics(
            window=int(last.get("window", 0) or 0),
            start=str(last.get("start", "")),
            end=str(last.get("end", "")),
            n_bars=int(last.get("n_bars", 0) or 0),
            timesteps=int(last.get("timesteps", 0) or 0),
            cumulative_return=float(last.get("cumulative_return", 0) or 0),
            sharpe=float(last.get("sharpe", 0) or 0),
            max_drawdown=float(last.get("max_drawdown", 0) or 0),
            final_equity=float(last.get("final_equity", 0) or 0),
            news_coverage=float(last.get("news_coverage", 0) or 0),
            calmar=float(last["calmar"]),
            approx_kl=float(last.get("approx_kl", float("nan"))),
            entropy_loss=float(last.get("entropy_loss", float("nan"))),
            ppo_updates=int(last.get("ppo_updates", 0) or 0),
        )
    except (TypeError, ValueError):
        return None


def promote_zip_cli(output_dir: Path, zip_path: Path) -> None:
    """Manual Promote after you missed the Telegram wait (or skipped Telegram)."""
    output_dir = Path(output_dir)
    src = Path(zip_path)
    if not src.exists():
        alt = output_dir / src.name
        if alt.exists():
            src = alt
        else:
            raise FileNotFoundError(f"--promote-zip not found: {zip_path}")
    live = load_live_best(output_dir)
    row = _row_from_finetune_log(output_dir, src.name)
    if row is None:
        logger.warning("No finetune_log row for %s. Promoting with unknown Calmar is refused.", src.name)
        raise SystemExit(2)
    live_calmar = float(live.get("calmar", TRADING_GOLDEN_CALMAR))
    if _finite(row.calmar) and float(row.calmar) <= live_calmar:
        logger.warning(
            "This zip Calmar %.4f does not beat live %.4f. Promoting anyway because --promote-zip is explicit.",
            float(row.calmar),
            live_calmar,
        )
    apply_promotion(output_dir, src, row=row, live_before=live)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AirAire daily PPO fine-tune (latest 1–3 windows, GPU v2 settings)")
    p.add_argument(
        "--windows",
        type=int,
        default=1,
        help="How many of the latest rolling 30-day windows to train (1–3, default 1 ≈ 2–3 min on GPU).",
    )
    p.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS, help="Session days per window (default: 30).")
    p.add_argument(
        "--output",
        type=Path,
        default=NEWS_GPU_V2_MODELS_DIR,
        help="Checkpoint directory (default: models/news_gpu_v2).",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Warm-start zip. Default: newest checkpoint_/finetuned_ in --output (not best_model.zip).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda", "auto"))
    p.add_argument(
        "--no-futu",
        action="store_true",
        help="Skip the OpenD 10-min refresh (news refresh still runs unless --skip-news).",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Max calendar days to pull from Futu when filling an offline gap (default: 30).",
    )
    p.add_argument(
        "--news-days",
        type=int,
        default=None,
        help="Calendar days of Alpha Vantage NEWS_SENTIMENT to re-query before PPO (default: same as --lookback-days).",
    )
    p.add_argument(
        "--skip-news",
        action="store_true",
        help="Do not call Alpha Vantage; keep existing news_score on the parquet.",
    )
    p.add_argument(
        "--cache-news",
        action="store_true",
        help="Skip the AV API if the local cache already covers the news window.",
    )
    p.add_argument(
        "--force-news-fetch",
        action="store_true",
        help="Re-query Alpha Vantage even if cache covers the range (this is now the default).",
    )
    p.add_argument(
        "--no-telegram",
        action="store_true",
        help="Do not ping Telegram or wait for Promote/Keep even if Calmar beats the live best.",
    )
    p.add_argument(
        "--promote-wait",
        type=int,
        default=DEFAULT_PROMOTE_WAIT_SECONDS,
        metavar="SECONDS",
        help="Seconds to wait for a Telegram Promote/Keep tap after a beating run (default: 600). 0 = notify only.",
    )
    p.add_argument(
        "--promote-zip",
        type=Path,
        default=None,
        help="Copy this zip onto best_model.zip now and raise the live Calmar bar (skips training).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    freeze_support()
    args = parse_args()
    if args.promote_zip is not None:
        promote_zip_cli(args.output, args.promote_zip)
        sys.exit(0)
    finetune(
        n_windows=args.windows,
        window_days=args.window_days,
        output=args.output,
        checkpoint=args.checkpoint,
        seed=args.seed,
        device=args.device,
        refresh_futu=not args.no_futu,
        lookback_days=args.lookback_days,
        force_news_fetch=(not args.cache_news) or args.force_news_fetch,
        skip_news=args.skip_news,
        news_days=args.news_days,
        notify_telegram=not args.no_telegram,
        promote_wait_seconds=args.promote_wait,
    )
