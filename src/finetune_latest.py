"""Lightweight daily PPO fine-tune from the latest healthy GPU v2 checkpoint.

Do **not** retrain from Window 90 (entropy-collapsed history). Always warm-start
from the newest ``checkpoint_*.zip`` / ``finetuned_*.zip`` in
``models/news_gpu_v2`` — typically ``checkpoint_2026-08-18.zip`` (resurrected
Window 118, Calmar 1.83) until a later fine-tune exists.

``best_model.zip`` (Window 113, Calmar 2.05) is the paper-trading brain and is
not overwritten here.

Typical GPU VM runtime: ~2-3 minutes for 1 window (8 PPO updates, same
hyperparameters as ``train_gpu_v2``).

    python -m src.finetune_latest
    python -m src.finetune_latest --windows 3
    python -m src.finetune_latest --no-futu --device cpu
"""

from __future__ import annotations

import argparse
import math
import re
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
    send_telegram_alert,
    setup_logging,
)

logger = setup_logging("airaire.finetune_latest")

# Golden artifacts from the 2026-08-20 resurrection. Never overwrite these.
PROTECTED_ZIPS = frozenset(
    {
        "best_model.zip",  # paper trading (Window 113, Calmar 2.05)
        "checkpoint_2026-08-12.zip",  # same weights as best_model
        "checkpoint_2026-08-18.zip",  # continue-training seed (Window 118, Calmar 1.83)
    }
)
_DATED_ZIP_RE = re.compile(r"^(checkpoint|finetuned)_(\d{4}-\d{2}-\d{2})\.zip$")
MIN_WINDOWS = 1
MAX_WINDOWS = 3

# Promotion baselines from the 2026-08-20 resurrection (training_log_history.csv).
# Compared against the latest fine-tune window; never auto-copies best_model.zip.
TRADING_GOLDEN_WINDOW = 113
TRADING_GOLDEN_END = "2026-08-12"
TRADING_GOLDEN_CALMAR = 2.053856720691549  # ~2.05, paper-trading brain
TRADING_GOLDEN_ZIP = "best_model.zip"

TRAINING_SEED_WINDOW = 118
TRAINING_SEED_END = "2026-08-18"
TRAINING_SEED_CALMAR = 1.832871817457733  # continue-training seed
TRAINING_SEED_ZIP = "checkpoint_2026-08-18.zip"

FINETUNE_LOG_NAME = "finetune_log.csv"
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
    "beats_trading_golden",
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


def load_panel(*, refresh_futu: bool, lookback_days: int, force_news_fetch: bool) -> pd.DataFrame:
    """Load enhanced parquet, optionally append missing Futu 10-min bars, ffill news."""
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
    if not refresh_futu:
        return panel

    now = _hk_today()
    start = default_futu_fetch_start(panel, now=now, lookback_days=lookback_days)
    logger.info("Refreshing Futu 10-min bars from %s to %s (offline-gap fill).", start.date(), now.date())
    try:
        live = fetch_futu_history(CORE_TICKERS, start=start, end=now, lookback_days=lookback_days)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Futu refresh failed (%s). Fine-tune continues on the cached parquet.", exc)
        return panel

    if live is None or live.empty:
        logger.info("No new Futu bars (OpenD down or already current). Using cached parquet.")
        return panel

    old_max = pd.Timestamp(panel["datetime"].max()) if not panel.empty else None
    panel = _sanitize_panel(overlay_live_ohlcv(panel, live))
    new_max = pd.Timestamp(panel["datetime"].max()) if not panel.empty else None
    logger.info("Panel after Futu overlay: last bar %s → %s", old_max, new_max)

    # Pull news only for the newly added span so we do not re-backfill 2 years.
    if old_max is not None and new_max is not None and new_max > old_max:
        try:
            from src.news_loader import load_all_news
            from src.data_loader import merge_price_news

            news = load_all_news(old_max - pd.Timedelta(days=2), new_max, force_fetch=force_news_fetch)
            if news is not None and not news.empty:
                panel = _sanitize_panel(merge_price_news(panel.drop(columns=["news_score"], errors="ignore"), news))
        except Exception as exc:  # noqa: BLE001 — sentiment is optional for a 2-minute job
            logger.warning("News refresh for new bars failed (%s). Forward-filled scores kept.", exc)

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
    trading_calmar: float,
) -> dict:
    beats = bool(_finite(row.calmar) and float(row.calmar) > float(trading_calmar))
    payload = {col: "" for col in FINETUNE_LOG_COLUMNS}
    payload.update({k: v for k, v in row.__dict__.items() if k in payload})
    payload.update(
        {
            "run_id": run_id,
            "role": "finetune",
            "zip": zip_name,
            "beats_trading_golden": beats,
        }
    )
    return payload


def append_finetune_log(
    output_dir: Path,
    new_rows: list[dict],
    *,
    trading_calmar: float,
    seed_calmar: float,
) -> Path:
    """Append today's rows. Goldens (W113, W118) stay pinned as the first two data rows.

    ``training_log_history.csv`` is left chronological — we do **not** move 113/118
    around in that file. This dedicated log is the one you open for promotion.
    """
    path = output_dir / FINETUNE_LOG_NAME
    hist_113 = _history_lookup(output_dir, TRADING_GOLDEN_WINDOW, TRADING_GOLDEN_END)
    hist_118 = _history_lookup(output_dir, TRAINING_SEED_WINDOW, TRAINING_SEED_END)
    goldens = [
        _golden_log_row(
            role="trading_golden",
            window=TRADING_GOLDEN_WINDOW,
            start="2026-07-09",
            end=TRADING_GOLDEN_END,
            calmar=trading_calmar,
            zip_name=TRADING_GOLDEN_ZIP,
            history=hist_113,
        ),
        _golden_log_row(
            role="training_seed",
            window=TRAINING_SEED_WINDOW,
            start="2026-07-15",
            end=TRAINING_SEED_END,
            calmar=seed_calmar,
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
        golden_df = existing.loc[existing["role"].isin(["trading_golden", "training_seed"])]
        rest = existing.loc[~existing["role"].isin(["trading_golden", "training_seed"])]
        if golden_df.empty:
            golden_df = pd.DataFrame(goldens).reindex(columns=FINETUNE_LOG_COLUMNS)
        # Dedup today's append against what is already on disk.
        if not incoming.empty and not rest.empty:
            key_cols = [c for c in ("run_id", "window", "start", "end") if c in rest.columns]
            if key_cols:
                seen = set(zip(*(rest[c].astype(str) for c in key_cols)))
                mask = [
                    tuple(str(v) for v in rec) not in seen
                    for rec in zip(*(incoming[c].astype(str) for c in key_cols))
                ]
                incoming = incoming.loc[mask]
        combined = pd.concat([golden_df, rest, incoming], ignore_index=True)
    else:
        combined = pd.concat(
            [pd.DataFrame(goldens).reindex(columns=FINETUNE_LOG_COLUMNS), incoming],
            ignore_index=True,
        )

    # Keep window ids as ints in the CSV (113 not 113.0).
    for col in ("window", "n_bars", "timesteps", "ppo_updates"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("Int64")

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    logger.info(
        "Appended %d fine-tune row(s) -> %s (file rows=%d; W113/W118 stay at the top)",
        len(incoming),
        path,
        len(combined),
    )
    return path


def report_promotion(
    *,
    row: WindowMetrics | None,
    zip_path: Path | None,
    trading_calmar: float,
    seed_calmar: float,
    notify: bool,
) -> None:
    """Terminal scoreboard + Telegram only when the new Calmar beats Window 113.

    Never copies ``best_model.zip``. The operator still promotes by hand.
    """
    new_calmar = float(row.calmar) if row is not None and _finite(row.calmar) else float("nan")
    beats = bool(_finite(new_calmar) and new_calmar > trading_calmar)
    zip_name = zip_path.name if zip_path is not None else "(not saved)"
    win_label = f"W{row.window}  {row.start} → {row.end}" if row is not None else "n/a"

    logger.info("============================================================")
    logger.info("PROMOTION CHECK  (best_model.zip is never overwritten)")
    logger.info(
        "  Trading golden  W%d  end %s  Calmar %.4f  (%s)",
        TRADING_GOLDEN_WINDOW,
        TRADING_GOLDEN_END,
        trading_calmar,
        TRADING_GOLDEN_ZIP,
    )
    logger.info(
        "  Training seed   W%d  end %s  Calmar %.4f  (%s)",
        TRAINING_SEED_WINDOW,
        TRAINING_SEED_END,
        seed_calmar,
        TRAINING_SEED_ZIP,
    )
    if row is None:
        logger.info("  This fine-tune  %s  Calmar n/a  (%s)", win_label, zip_name)
        logger.info("  Verdict: no usable window (collapse / NaN). Keep %s.", TRADING_GOLDEN_ZIP)
    else:
        logger.info(
            "  This fine-tune  %s  Calmar %.4f  (%s)",
            win_label,
            new_calmar,
            zip_name,
        )
        if beats:
            logger.info(
                "  Verdict: BEATS Window 113 (%.4f > %.4f). "
                "If you want it live, copy %s over %s yourself.",
                new_calmar,
                trading_calmar,
                zip_name,
                TRADING_GOLDEN_ZIP,
            )
        else:
            logger.info(
                "  Verdict: KEEP %s (new %.4f is not above %.4f). No Telegram.",
                TRADING_GOLDEN_ZIP,
                new_calmar,
                trading_calmar,
            )
    logger.info("============================================================")

    if not beats or not notify:
        if beats and not notify:
            logger.info("Telegram skipped (--no-telegram).")
        return
    if zip_path is None or row is None:
        return

    message = (
        "AirAire fine-tune: Calmar beat Window 113\n"
        f"New: W{row.window}  {row.start} → {row.end}  Calmar {new_calmar:.4f}\n"
        f"Golden: W{TRADING_GOLDEN_WINDOW}  {TRADING_GOLDEN_END}  Calmar {trading_calmar:.4f}\n"
        f"Zip: {zip_path}\n"
        f"best_model.zip was NOT changed.\n"
        f"To promote: copy {zip_path.name} over best_model.zip"
    )
    sent = send_telegram_alert(message)
    if sent:
        logger.info("Telegram promotion ping sent.")
    else:
        logger.warning(
            "Telegram ping not delivered (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env). "
            "The terminal verdict above is the source of truth."
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
    force_news_fetch: bool = False,
    notify_telegram: bool = True,
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

        trading_calmar, seed_calmar = load_reference_calmars(output_dir)
        log_rows = [
            _metrics_log_row(
                m,
                run_id=run_id,
                zip_name=last_saved.name if last_saved is not None else "",
                trading_calmar=trading_calmar,
            )
            for m in metrics
        ]
        append_finetune_log(
            output_dir,
            log_rows,
            trading_calmar=trading_calmar,
            seed_calmar=seed_calmar,
        )
        report_promotion(
            row=last_saved_row,
            zip_path=last_saved,
            trading_calmar=trading_calmar,
            seed_calmar=seed_calmar,
            notify=notify_telegram,
        )
        if last_saved is not None:
            logger.info(
                "Fine-tune complete. Trading still uses %s ; new weights are %s",
                output_dir / "best_model.zip",
                last_saved,
            )
        else:
            logger.warning("Fine-tune finished with no new zip (collapse, NaN, or protected-path skip).")
        return metrics
    finally:
        restore_log()


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
        help="Skip the OpenD 10-min refresh (use enhanced_data.parquet as-is).",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Max calendar days to pull from Futu when filling an offline gap (default: 30).",
    )
    p.add_argument("--force-news-fetch", action="store_true", help="Re-query Alpha Vantage for newly added bars.")
    p.add_argument(
        "--no-telegram",
        action="store_true",
        help="Do not ping Telegram even if the new Calmar beats Window 113.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    freeze_support()
    args = parse_args()
    finetune(
        n_windows=args.windows,
        window_days=args.window_days,
        output=args.output,
        checkpoint=args.checkpoint,
        seed=args.seed,
        device=args.device,
        refresh_futu=not args.no_futu,
        lookback_days=args.lookback_days,
        force_news_fetch=args.force_news_fetch,
        notify_telegram=not args.no_telegram,
    )
