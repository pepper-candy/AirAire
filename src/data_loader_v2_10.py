"""V2.10 panel: Bloomberg OHLC for 5 core + HSI only, volume forced to 0, V2 news.

Bloomberg tape currently ends ~2026-08-18 10:00 HKT. Futu fills the gap through
the last US cash close on 2026-08-21 (04:00 HKT on 2026-08-22). The panel is
clipped to calendar 2026-08-21 so 2026-08-24 stays a fair test.

Does not write enhanced_data.parquet, enhanced_v3.parquet, or enhanced_v4.parquet.
Does not keep US.SPX even if the Bloomberg folder has it.

    python -m src.data_loader_v2_10 --force-rebuild
    python -m src.data_loader_v2_10 --force-rebuild --skip-futu
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.data_loader import (
    STANDARD_COLUMNS,
    fetch_futu_history,
    load_bloomberg,
    merge_price_news,
    overlay_live_ohlcv,
)
from src.futu_codes import FUTU_KLINE_ALIASES_V2_10
from src.utils import (
    CORE_TICKERS,
    DATA_ENHANCED,
    ENHANCED_PARQUET,
    HK_TZ,
    MODELS_DIR,
    setup_logging,
)

logger = setup_logging("airaire.data_loader_v2_10")

ENHANCED_V2_10_PARQUET = DATA_ENHANCED / "enhanced_v2_10.parquet"
NEWS_GPU_V2_10_MODELS_DIR = MODELS_DIR / "news_gpu_v2_10"
OBSERVER_TICKERS_V2_10 = ["HK.HSI"]
V2_10_TICKERS = list(CORE_TICKERS) + list(OBSERVER_TICKERS_V2_10)

# Last completed HK + US session before the 2026-08-24 fair test.
# US 16:00 ET on 2026-08-21 == 04:00 HKT on 2026-08-22. Panel clocks are naive
# local-market (HK HKT, US ET), so clipping by calendar date 2026-08-21 is the
# honest cutoff — do not use a single HKT timestamp across both books.
DEFAULT_PANEL_END = "2026-08-21"
DEFAULT_FUTU_START = "2026-08-17"
# OpenD date bound (inclusive). drop_incomplete uses 04:00 HKT 22 Aug as "now".
DEFAULT_FUTU_END = "2026-08-22"
US_FRIDAY_CLOSE_HKT = datetime(2026, 8, 22, 4, 0, 0)


def _zero_volume(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["volume"] = 0.0
    return out


def _drop_spx(panel: pd.DataFrame) -> pd.DataFrame:
    if panel is None or panel.empty or "ticker" not in panel.columns:
        return panel
    keep = set(V2_10_TICKERS)
    tickers = panel["ticker"].astype(str)
    dropped = sorted({t for t in tickers.unique().tolist() if t not in keep})
    if dropped:
        logger.info("V2.10 dropped non-universe tickers: %s", dropped)
    return panel.loc[tickers.isin(keep)].copy()


def _clip_panel_end(panel: pd.DataFrame, end_day: str | None) -> pd.DataFrame:
    """Keep bars whose naive local-market calendar date is <= end_day."""
    if panel is None or panel.empty or not end_day:
        return panel
    end = pd.Timestamp(end_day).normalize()
    dt = pd.to_datetime(panel["datetime"], errors="coerce")
    keep = dt.dt.normalize() <= end
    out = panel.loc[keep].copy()
    dropped = int((~keep).sum())
    logger.info(
        "V2.10 clipped to <= %s  kept=%d dropped=%d  last_bar=%s",
        end.date(),
        len(out),
        dropped,
        out["datetime"].max() if not out.empty else None,
    )
    return out


def _copy_v2_news(panel: pd.DataFrame) -> pd.DataFrame:
    """HSI has no AV ticker feed. Core names copy V2 parquet news_score."""
    out = panel.copy()
    out["news_score"] = 0.0
    if not ENHANCED_PARQUET.exists():
        logger.warning("No V2 enhanced parquet at %s — news_score stays 0.", ENHANCED_PARQUET)
        return out
    v2 = pd.read_parquet(ENHANCED_PARQUET)
    if "news_score" not in v2.columns:
        logger.warning("V2 parquet has no news_score — V2.10 news stays 0.")
        return out
    news = v2.loc[v2["ticker"].astype(str).isin(CORE_TICKERS), ["datetime", "ticker", "news_score"]].copy()
    news["datetime"] = pd.to_datetime(news["datetime"], errors="coerce")
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    merged = out.drop(columns=["news_score"]).merge(news, on=["datetime", "ticker"], how="left")
    merged["news_score"] = pd.to_numeric(merged["news_score"], errors="coerce")
    merged["news_score"] = merged.groupby("ticker", group_keys=False)["news_score"].ffill().fillna(0.0)
    logger.info(
        "Copied V2 news_score onto V2.10 core names. HSI stays 0. coverage=%.1f%%",
        100.0 * float((merged["news_score"].abs() > 1e-12).mean()) if len(merged) else 0.0,
    )
    return merged


def _attach_news(panel: pd.DataFrame, *, force_news_fetch: bool = False) -> pd.DataFrame:
    """V2 parquet first, then as-of merge from the local news cache for the Futu gap.

    Does not re-download the 2-year Alpha Vantage history unless ``force_news_fetch``.
    ``data/raw/news/`` already covers through 2026-08-22 on this project.
    """
    base = _copy_v2_news(panel)
    if panel is None or panel.empty:
        return base
    try:
        from src.news_loader import load_all_news
    except Exception as exc:  # noqa: BLE001
        logger.warning("News loader unavailable (%s). Using V2 parquet / ffill only.", exc)
        return base
    start = pd.to_datetime(base["datetime"], errors="coerce").min()
    end = pd.to_datetime(base["datetime"], errors="coerce").max()
    if pd.isna(start) or pd.isna(end):
        return base
    try:
        news = load_all_news(start, end, force_fetch=force_news_fetch)
    except Exception as exc:  # noqa: BLE001
        logger.warning("load_all_news failed (%s). Using V2 parquet / ffill only.", exc)
        return base
    if news is None or news.empty:
        logger.warning("News cache empty for %s → %s. Futu-gap bars will ffill last V2 score.", start, end)
        return base
    prices = base.drop(columns=["news_score"], errors="ignore")
    merged = merge_price_news(prices, news)
    logger.info(
        "Merged news cache onto V2.10 panel. coverage=%.1f%%",
        100.0 * float((merged["news_score"].abs() > 1e-12).mean()) if len(merged) else 0.0,
    )
    return merged


def _parse_day_end(raw: str | None) -> datetime:
    """Futu API / drop_incomplete 'now'. Date-only 2026-08-22 → 04:00 HKT (US Fri close)."""
    if not raw:
        return US_FRIDAY_CLOSE_HKT
    s = str(raw).strip().replace(" ", "T")
    if len(s) <= 10:
        if s >= DEFAULT_FUTU_END:
            return US_FRIDAY_CLOSE_HKT
        return datetime.fromisoformat(f"{s}T23:59:59")
    return datetime.fromisoformat(s)


def _parse_day_start(raw: str | None) -> datetime:
    if not raw:
        return datetime(2026, 8, 17)
    s = str(raw).strip()
    if len(s) <= 10:
        return datetime.fromisoformat(f"{s}T00:00:00")
    return datetime.fromisoformat(s.replace(" ", "T"))


def fetch_v2_10_futu_overlay(start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
    """OHLC from OpenD for 5 core + HSI. Volume is discarded later. No SPX."""
    frames: list[pd.DataFrame] = []
    end = end or US_FRIDAY_CLOSE_HKT
    start = start or datetime(2026, 8, 17)
    for train_code in V2_10_TICKERS:
        aliases = FUTU_KLINE_ALIASES_V2_10.get(train_code, (train_code,))
        got = None
        used = None
        for code in aliases:
            part = fetch_futu_history(tickers=[code], start=start, end=end)
            if part is not None and not part.empty:
                got = part.copy()
                got["ticker"] = train_code
                used = code
                break
        if got is None:
            logger.warning("V2.10 Futu overlay: no klines for %s (tried %s).", train_code, aliases)
            continue
        if used != train_code:
            logger.info("V2.10 Futu klines %s remapped → %s (%d rows).", used, train_code, len(got))
        frames.append(got)
    if not frames:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return _drop_spx(pd.concat(frames, ignore_index=True))


def _finish_panel(
    panel: pd.DataFrame,
    *,
    panel_end: str | None,
    force_news_fetch: bool,
) -> pd.DataFrame:
    panel = _zero_volume(_drop_spx(panel))
    panel = _clip_panel_end(panel, panel_end)
    panel = _attach_news(panel, force_news_fetch=force_news_fetch)
    panel = _drop_spx(panel).sort_values(["datetime", "ticker"]).reset_index(drop=True)
    tickers = sorted(panel["ticker"].astype(str).unique().tolist())
    logger.info(
        "V2.10 panel rows=%d tickers=%s span=%s → %s volume forced 0 (no SPX, clipped)",
        len(panel),
        tickers,
        panel["datetime"].min() if not panel.empty else None,
        panel["datetime"].max() if not panel.empty else None,
    )
    missing_core = [t for t in CORE_TICKERS if t not in tickers]
    missing_obs = [t for t in OBSERVER_TICKERS_V2_10 if t not in tickers]
    if missing_core:
        logger.warning("V2.10 panel missing core: %s", missing_core)
    if missing_obs:
        logger.warning("V2.10 panel missing observers: %s", missing_obs)
    return panel


def load_enhanced_v2_10(
    *,
    save: bool = True,
    fetch_futu: bool = True,
    force_rebuild: bool = False,
    force_news_fetch: bool = False,
    panel_end: str | None = DEFAULT_PANEL_END,
    futu_start: str | None = DEFAULT_FUTU_START,
    futu_end: str | None = DEFAULT_FUTU_END,
    **_kwargs,
) -> pd.DataFrame:
    futu_start_dt = _parse_day_start(futu_start)
    futu_end_dt = _parse_day_end(futu_end)

    if ENHANCED_V2_10_PARQUET.exists() and not force_rebuild:
        cached = _drop_spx(pd.read_parquet(ENHANCED_V2_10_PARQUET))
        logger.info("Loading cached %s (%d rows).", ENHANCED_V2_10_PARQUET, len(cached))
        if fetch_futu:
            live = fetch_v2_10_futu_overlay(start=futu_start_dt, end=futu_end_dt)
            if live is not None and not live.empty:
                cached = overlay_live_ohlcv(cached, live, now=futu_end_dt)
        cached = _finish_panel(cached, panel_end=panel_end, force_news_fetch=force_news_fetch)
        if save:
            ENHANCED_V2_10_PARQUET.parent.mkdir(parents=True, exist_ok=True)
            cached.to_parquet(ENHANCED_V2_10_PARQUET, index=False)
            logger.info("Updated %s. Aug 24 was not added.", ENHANCED_V2_10_PARQUET)
        return cached

    prices = _drop_spx(load_bloomberg())
    if prices.empty:
        raise FileNotFoundError("V2.10 Bloomberg backbone is empty. Need data/raw/bloomberg/*.csv")
    missing = [t for t in V2_10_TICKERS if t not in set(prices["ticker"].astype(str))]
    if "HK.HSI" in missing:
        raise FileNotFoundError(
            "V2.10 needs Bloomberg HSI (HSI_10min.csv). SPX is not used and should not be copied for this run."
        )
    if missing:
        logger.warning("Bloomberg missing %s — V2.10 cube will ffill/zero those names.", missing)

    panel = _zero_volume(prices)

    if fetch_futu:
        live = fetch_v2_10_futu_overlay(start=futu_start_dt, end=futu_end_dt)
        if live is not None and not live.empty:
            panel = overlay_live_ohlcv(panel, live, now=futu_end_dt)
            logger.info("Futu overlay applied %s → %s (will clip to %s).", futu_start_dt, futu_end_dt, panel_end)
        else:
            logger.warning("Futu overlay returned no rows. Panel will stop at Bloomberg (~18 Aug morning).")

    panel = _finish_panel(panel, panel_end=panel_end, force_news_fetch=force_news_fetch)

    if save and not panel.empty:
        ENHANCED_V2_10_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(ENHANCED_V2_10_PARQUET, index=False)
        logger.info("Wrote %s. V2/V3/V4 parquets were not touched.", ENHANCED_V2_10_PARQUET)
    return panel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build data/enhanced/enhanced_v2_10.parquet (Bloomberg 5+HSI, Futu gap, clip 2026-08-21)."
    )
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--skip-futu", action="store_true", help="Bloomberg only (stops ~18 Aug 10:00 HKT).")
    p.add_argument("--panel-end", default=DEFAULT_PANEL_END, help="Keep naive calendar dates <= this day. Default 2026-08-21.")
    p.add_argument("--futu-start", default=DEFAULT_FUTU_START, help="OpenD history start (default 2026-08-17).")
    p.add_argument("--futu-end", default=DEFAULT_FUTU_END, help="OpenD history end date (default 2026-08-22 = US Fri close).")
    p.add_argument("--force-news-fetch", action="store_true", help="Re-query Alpha Vantage. Default uses data/raw/news cache.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    load_enhanced_v2_10(
        save=True,
        fetch_futu=not args.skip_futu,
        force_rebuild=args.force_rebuild,
        force_news_fetch=args.force_news_fetch,
        panel_end=args.panel_end,
        futu_start=args.futu_start,
        futu_end=args.futu_end,
    )
