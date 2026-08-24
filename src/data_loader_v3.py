"""V3 panel builder — isolated from the live V2 parquet.

Does **not** write ``data/enhanced/enhanced_data.parquet`` or anything under
``models/news_gpu_v2/``.

Design (same 6-month regime as V2, extra information only):

* Prices stay on the Bloomberg / V2 clock (2026-02-24 → 2026-08-21).
* Volume is overlaid from TradingView (6 names) and Futu (CATL only).
* ``TSE_DLY_3750`` is ignored (Tokyo 3750, not HK.03750).
* HSI TradingView volume is empty — stays 0.
* Session filter uses ``is_hk_market_open`` / ``is_us_market_open`` (DST-safe)
  on the *volume sources only*. Bloomberg US afternoon stamps are stored as
  1:30–4:00 without PM; filtering those with the helpers would drop the
  whole US afternoon. We keep V2 prices and match US volume with a 12h shift.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.data_loader import (
    STANDARD_COLUMNS,
    fetch_futu_history,
    load_bloomberg,
    load_ticker_csv,
    merge_price_news,
)
from src.utils import (
    ALL_TICKERS,
    CORE_TICKERS,
    DATA_ENHANCED,
    DATA_RAW_BLOOMBERG,
    ENHANCED_PARQUET,
    HK_TZ,
    MODELS_DIR,
    OBSERVER_TICKERS,
    PROJECT_ROOT,
    US_TZ,
    is_hk_market_open,
    is_us_market_open,
    setup_logging,
    ticker_market,
)

logger = setup_logging("airaire.data_loader_v3")

ENHANCED_V3_PARQUET = DATA_ENHANCED / "enhanced_v3.parquet"
NEWS_GPU_V3_MODELS_DIR = MODELS_DIR / "news_gpu_v3"
DATA_RAW_TRADINGVIEW = PROJECT_ROOT / "data" / "raw" / "tradingview"
DATA_RAW_FUTU_V3 = PROJECT_ROOT / "data" / "raw" / "futu" / "v3"
CATL_FUTU_CACHE = DATA_RAW_FUTU_V3 / "03750_HK_10min.csv"

V3_START = pd.Timestamp("2026-02-24")
V3_END = pd.Timestamp("2026-08-21 23:59:59")
VOLUME_MATCH_TOLERANCE = pd.Timedelta(minutes=10)
US_AFTERNOON_SHIFT = pd.Timedelta(hours=12)

# Exact names from the friend export, plus globs so a hash-suffix rename still hits.
TV_FILES: dict[str, str] = {
    "US.COST": "BATS_COST, 10_ee846.csv",
    "US.KO": "BATS_KO, 10_e396d.csv",
    "HK.00700": "HKEX_DLY_700, 10_46fa9.csv",
    "HK.03690": "HKEX_DLY_3690, 10_ccc44.csv",
    "HK.HSI": "HSI_HSI, 10_1a47d.csv",
    "US.SPX": "SP_DLY_SPX, 10_beeae.csv",
}
TV_GLOBS: dict[str, tuple[str, ...]] = {
    "US.COST": ("BATS_COST*.csv",),
    "US.KO": ("BATS_KO*.csv",),
    "HK.00700": ("HKEX_DLY_700*.csv",),
    "HK.03690": ("HKEX_DLY_3690*.csv",),
    "HK.HSI": ("HSI_HSI*.csv",),
    "US.SPX": ("SP_DLY_SPX*.csv",),
}
TV_VOLUME_TICKERS = tuple(TV_FILES.keys())
IGNORE_TV_PREFIXES = ("TSE_DLY_3750",)
CATL_TICKER = "HK.03750"
_TV_DIR_OVERRIDE: Path | None = None


def clip_v3_window(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "datetime" not in df.columns:
        return df if df is not None else pd.DataFrame(columns=STANDARD_COLUMNS)
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime"])
    mask = (out["datetime"] >= V3_START) & (out["datetime"] <= V3_END)
    return out.loc[mask].reset_index(drop=True)


def bar_is_regular_session(ts, ticker: str) -> bool:
    """DST-safe cash-session check. ``ts`` should be timezone-aware (UTC or local)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        zone = HK_TZ if ticker_market(ticker) == "HK" else US_TZ
        t = t.tz_localize(zone)
    dt = t.to_pydatetime()
    if ticker_market(ticker) == "HK":
        return is_hk_market_open(dt)
    return is_us_market_open(dt)


def filter_regular_session(df: pd.DataFrame, *, datetime_is_utc: bool) -> pd.DataFrame:
    """Drop pre/post/lunch bars using the existing DST-safe helpers."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame(columns=STANDARD_COLUMNS)
    keep = []
    for ticker, part in df.groupby("ticker", sort=False):
        times = pd.to_datetime(part["datetime"], errors="coerce", utc=datetime_is_utc)
        if datetime_is_utc:
            mask = [_safe_session(ts, str(ticker)) for ts in times]
        else:
            mask = [bar_is_regular_session(ts, str(ticker)) for ts in times]
        keep.append(part.loc[mask])
    if not keep:
        return df.iloc[0:0].reset_index(drop=True)
    out = pd.concat(keep, ignore_index=True)
    dropped = len(df) - len(out)
    if dropped:
        logger.info("Regular-session filter dropped %d / %d source bars.", dropped, len(df))
    return out.reset_index(drop=True)


def _safe_session(ts, ticker: str) -> bool:
    if pd.isna(ts):
        return False
    return bar_is_regular_session(ts, ticker)


def resolve_tv_dir(tv_dir: Path | str | None = None) -> Path:
    if tv_dir is not None:
        return Path(tv_dir)
    if _TV_DIR_OVERRIDE is not None:
        return _TV_DIR_OVERRIDE
    env = os.getenv("AIRAIR_TV_DIR", "").strip()
    if env:
        return Path(env)
    return DATA_RAW_TRADINGVIEW


def _tv_path(ticker: str, tv_dir: Path | None = None) -> Path | None:
    folder = resolve_tv_dir(tv_dir)
    exact = TV_FILES.get(ticker)
    if exact:
        path = folder / exact
        if path.exists():
            return path
    for pattern in TV_GLOBS.get(ticker, ()):
        hits = sorted(
            p
            for p in folder.glob(pattern)
            if p.is_file() and not any(p.name.startswith(prefix) for prefix in IGNORE_TV_PREFIXES)
        )
        if hits:
            return hits[0]
    return None


def load_tradingview_ticker(path: Path, ticker: str) -> pd.DataFrame:
    """Unix UTC TradingView CSV → naive local-market clock, regular session only."""
    if any(path.name.startswith(prefix) for prefix in IGNORE_TV_PREFIXES):
        logger.warning("Ignoring %s (not HK.03750).", path.name)
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    raw = pd.read_csv(path)
    if raw.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    raw.columns = [str(c).replace("\ufeff", "").strip() for c in raw.columns]
    if "time" not in raw.columns:
        logger.warning("%s: no 'time' column (%s).", path.name, list(raw.columns))
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    utc = pd.to_datetime(raw["time"], unit="s", utc=True, errors="coerce")
    vol_col = "Volume" if "Volume" in raw.columns else ("volume" if "volume" in raw.columns else None)
    frame = pd.DataFrame(
        {
            "datetime": utc,
            "ticker": ticker,
            "open": pd.to_numeric(raw["open"], errors="coerce") if "open" in raw.columns else 0.0,
            "high": pd.to_numeric(raw["high"], errors="coerce") if "high" in raw.columns else 0.0,
            "low": pd.to_numeric(raw["low"], errors="coerce") if "low" in raw.columns else 0.0,
            "close": pd.to_numeric(raw["close"], errors="coerce") if "close" in raw.columns else 0.0,
            "volume": pd.to_numeric(raw[vol_col], errors="coerce") if vol_col else 0.0,
        }
    )
    frame = frame.dropna(subset=["datetime"])
    frame = filter_regular_session(frame, datetime_is_utc=True)
    if frame.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    zone = HK_TZ if ticker_market(ticker) == "HK" else US_TZ
    local = pd.to_datetime(frame["datetime"], utc=True).dt.tz_convert(zone).dt.tz_localize(None)
    frame = frame.copy()
    frame["datetime"] = local
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    return frame[STANDARD_COLUMNS].sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)


def load_tradingview_volume(tv_dir: Path | str | None = None) -> pd.DataFrame:
    folder = resolve_tv_dir(tv_dir)
    present = sorted(p.name for p in folder.glob("*.csv")) if folder.exists() else []
    logger.info("TradingView dir=%s exists=%s csv_count=%d", folder, folder.exists(), len(present))
    if present:
        logger.info("TradingView CSVs found: %s", present)
    elif not folder.exists():
        logger.warning(
            "TradingView folder does not exist. Copy the 6 exports into %s "
            "(not TSE_DLY_3750), or rerun with --tv-dir PATH.",
            folder,
        )
    frames: list[pd.DataFrame] = []
    for ticker in TV_VOLUME_TICKERS:
        path = _tv_path(ticker, folder)
        if path is None:
            logger.warning("TradingView CSV missing for %s under %s.", ticker, folder)
            continue
        part = load_tradingview_ticker(path, ticker)
        if part.empty:
            logger.warning("TradingView %s yielded 0 regular-session rows (%s).", ticker, path.name)
            continue
        nz = float((pd.to_numeric(part["volume"], errors="coerce") > 0).mean())
        logger.info(
            "TradingView %s: %d session bars, volume>0=%.1f%% (%s → %s)",
            ticker,
            len(part),
            100.0 * nz,
            part["datetime"].min(),
            part["datetime"].max(),
        )
        frames.append(part)
    if not frames:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _load_cached_catl() -> pd.DataFrame:
    if not CATL_FUTU_CACHE.exists():
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return load_ticker_csv(CATL_FUTU_CACHE, CATL_TICKER)


def fetch_catl_futu_volume(*, fetch: bool = True) -> pd.DataFrame:
    """Recent HK.03750 volume from OpenD. Cached under data/raw/futu/v3/ (not V2 latest/)."""
    cached = _load_cached_catl()
    if not fetch:
        return cached
    try:
        live = fetch_futu_history(
            [CATL_TICKER],
            start=V3_START,
            end=V3_END,
            lookback_days=220,
        )
    except Exception as exc:  # noqa: BLE001 — OpenD is optional
        logger.warning("Futu CATL fetch failed (%s). Using cache only.", exc)
        return cached
    if live is None or live.empty:
        logger.warning("Futu CATL returned no rows. Using cache only (%d rows).", len(cached))
        return cached
    live = live.loc[live["ticker"] == CATL_TICKER].copy()
    live["datetime"] = pd.to_datetime(live["datetime"], errors="coerce")
    live = filter_regular_session(live, datetime_is_utc=False)
    if live.empty:
        return cached
    DATA_RAW_FUTU_V3.mkdir(parents=True, exist_ok=True)
    live[STANDARD_COLUMNS].to_csv(CATL_FUTU_CACHE, index=False)
    logger.info("Wrote CATL Futu volume cache %s (%d rows).", CATL_FUTU_CACHE, len(live))
    return live


def _nearest_volume(base_times: pd.Series, src: pd.DataFrame, market: str) -> pd.Series:
    left = pd.DataFrame(
        {
            "datetime": pd.to_datetime(base_times, errors="coerce"),
            "_i": range(len(base_times)),
        }
    )
    left = left.dropna(subset=["datetime"]).sort_values("datetime")
    right = src[["datetime", "volume"]].copy()
    right["datetime"] = pd.to_datetime(right["datetime"], errors="coerce")
    right["volume"] = pd.to_numeric(right["volume"], errors="coerce")
    right = right.dropna(subset=["datetime"]).sort_values("datetime")
    if left.empty or right.empty:
        return pd.Series(np.nan, index=base_times.index, dtype="float64")

    matched = pd.merge_asof(
        left,
        right,
        on="datetime",
        direction="nearest",
        tolerance=VOLUME_MATCH_TOLERANCE,
    )
    if market == "US":
        shifted = right.copy()
        shifted["datetime"] = shifted["datetime"] - US_AFTERNOON_SHIFT
        shifted = shifted.sort_values("datetime")
        alt = pd.merge_asof(
            left,
            shifted,
            on="datetime",
            direction="nearest",
            tolerance=VOLUME_MATCH_TOLERANCE,
        )
        matched["volume"] = matched["volume"].where(matched["volume"].notna(), alt["volume"])
    matched = matched.sort_values("_i")
    out = pd.Series(np.nan, index=base_times.index, dtype="float64")
    positions = matched["_i"].to_numpy()
    out.iloc[positions] = pd.to_numeric(matched["volume"], errors="coerce").to_numpy()
    return out


def overlay_volume_nearest(base: pd.DataFrame, volume_src: pd.DataFrame) -> pd.DataFrame:
    """Copy volume onto ``base`` OHLC. Never changes open/high/low/close."""
    if base is None or base.empty or volume_src is None or volume_src.empty:
        return base
    out = base.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    src = volume_src.copy()
    src["datetime"] = pd.to_datetime(src["datetime"], errors="coerce")
    src["volume"] = pd.to_numeric(src["volume"], errors="coerce")
    src = src.dropna(subset=["datetime", "ticker"])

    for ticker, g_base in out.groupby("ticker", sort=False):
        g_src = src.loc[src["ticker"] == ticker]
        if g_src.empty:
            continue
        market = ticker_market(str(ticker))
        vol = _nearest_volume(g_base["datetime"], g_src, market)
        hit = vol.notna()
        if not bool(hit.any()):
            logger.info("Volume overlay %s: 0 bars matched (tolerance %s).", ticker, VOLUME_MATCH_TOLERANCE)
            continue
        hit_np = hit.to_numpy()
        out.loc[g_base.index[hit_np], "volume"] = vol.to_numpy()[hit_np]
        nz = float((pd.to_numeric(vol.loc[hit], errors="coerce") > 0).mean())
        logger.info(
            "Volume overlay %s: matched %d / %d bars (%.1f%% of matches have volume>0).",
            ticker,
            int(hit.sum()),
            len(g_base),
            100.0 * nz,
        )
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    return out


def _price_backbone() -> pd.DataFrame:
    """V2-comparable prices: Bloomberg 7 names, clipped to the shared window."""
    if ENHANCED_PARQUET.exists():
        panel = pd.read_parquet(ENHANCED_PARQUET)
        logger.info("V3 price backbone from V2 enhanced parquet (%d rows).", len(panel))
    else:
        panel = load_bloomberg(DATA_RAW_BLOOMBERG)
        logger.info("V3 price backbone from Bloomberg CSVs (%d rows).", len(panel))
    panel = panel.copy()
    panel["datetime"] = pd.to_datetime(panel["datetime"], errors="coerce")
    have = set(panel["ticker"].astype(str).unique()) if not panel.empty else set()
    missing = [t for t in ALL_TICKERS if t not in have]
    if missing:
        extra = load_bloomberg(DATA_RAW_BLOOMBERG)
        extra = extra.loc[extra["ticker"].isin(missing)]
        if not extra.empty:
            keep_cols = [c for c in extra.columns if c in panel.columns or c in STANDARD_COLUMNS]
            panel = pd.concat([panel, extra[keep_cols]], ignore_index=True, sort=False)
            logger.info("Appended missing tickers from Bloomberg: %s", missing)
    panel = clip_v3_window(panel)
    for col in ("open", "high", "low", "close", "volume"):
        if col not in panel.columns:
            panel[col] = 0.0
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel["volume"] = 0.0
    return panel.sort_values(["ticker", "datetime"]).reset_index(drop=True)


def overlay_recent_news(
    panel: pd.DataFrame,
    *,
    news_days: int = 14,
    force_fetch: bool = True,
) -> pd.DataFrame:
    """Refresh Alpha Vantage on the last ``news_days`` only. Older news_score stays.

    Writes nothing by itself. Safe to run while V2 paper-trades — it only
    mutates the in-memory frame you pass in.
    """
    if panel is None or panel.empty:
        return panel
    out = panel.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    news_end = pd.Timestamp(out["datetime"].max())
    news_start = news_end - pd.Timedelta(days=max(int(news_days), 7))
    older = out.loc[out["datetime"] < news_start].copy()
    recent = out.loc[out["datetime"] >= news_start].copy()
    if recent.empty:
        logger.info("No bars on/after %s — news overlay skipped.", news_start)
        return out
    logger.info(
        "Refreshing Alpha Vantage NEWS_SENTIMENT %s → %s (%d recent bars, older news kept).",
        news_start.date(),
        news_end.date(),
        len(recent),
    )
    try:
        from src.news_loader import load_all_news

        news = load_all_news(news_start, news_end, force_fetch=force_fetch)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Recent news fetch failed (%s). Keeping existing news_score.", exc)
        return out
    if news is None or news.empty:
        logger.warning("Recent news fetch returned empty. Keeping existing news_score.")
        return out
    updated = merge_price_news(recent.drop(columns=["news_score"], errors="ignore"), news)
    if older.empty:
        merged = updated
    else:
        merged = pd.concat([older, updated], ignore_index=True)
    merged = merged.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    col = pd.to_numeric(merged["news_score"], errors="coerce") if "news_score" in merged.columns else None
    if col is not None:
        recent_mask = merged["datetime"] >= news_start
        nz = float((col.loc[recent_mask].abs() > 1e-12).mean()) if recent_mask.any() else 0.0
        logger.info("Recent news overlay coverage: %.1f%% of bars since %s have |news_score|>0.", 100.0 * nz, news_start.date())
    return merged


def refresh_recent_news_v3(*, news_days: int = 14, save: bool = True) -> pd.DataFrame:
    """Load ``enhanced_v3.parquet``, overlay last-N-days news, write it back. Does not touch V2."""
    if not ENHANCED_V3_PARQUET.exists():
        raise FileNotFoundError(
            f"{ENHANCED_V3_PARQUET} is missing. Build it first with `python -m src.data_loader_v3`."
        )
    panel = pd.read_parquet(ENHANCED_V3_PARQUET)
    logger.info("Loaded %s (%d rows) for recent news overlay.", ENHANCED_V3_PARQUET, len(panel))
    panel = overlay_recent_news(panel, news_days=news_days, force_fetch=True)
    if save and not panel.empty:
        ENHANCED_V3_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(ENHANCED_V3_PARQUET, index=False)
        logger.info("Wrote %s after recent news overlay. V2 parquet was not touched.", ENHANCED_V3_PARQUET)
    return panel


def _attach_news(panel: pd.DataFrame, *, force_news_fetch: bool = False) -> pd.DataFrame:
    if panel.empty:
        return panel
    if ENHANCED_PARQUET.exists() and not force_news_fetch:
        old = pd.read_parquet(ENHANCED_PARQUET)
        if "news_score" in old.columns:
            news = old[["datetime", "ticker", "news_score"]].copy()
            news["datetime"] = pd.to_datetime(news["datetime"], errors="coerce")
            merged = panel.drop(columns=["news_score"], errors="ignore").merge(
                news,
                on=["datetime", "ticker"],
                how="left",
            )
            merged["news_score"] = pd.to_numeric(merged["news_score"], errors="coerce").fillna(0.0)
            logger.info("Copied news_score from V2 enhanced parquet (no AV refetch).")
            return merged
    try:
        from src.news_loader import load_all_news

        news_df = load_all_news(V3_START, V3_END, force_fetch=force_news_fetch)
    except Exception as exc:  # noqa: BLE001
        logger.warning("News load failed (%s). news_score=0.0.", exc)
        news_df = None
    return merge_price_news(panel, news_df)


def load_enhanced_v3(
    *,
    save: bool = True,
    fetch_futu: bool = True,
    force_news_fetch: bool = False,
    force_rebuild: bool = False,
    tv_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Build the V3 panel. Writes ``enhanced_v3.parquet`` only."""
    global _TV_DIR_OVERRIDE
    if tv_dir is not None:
        _TV_DIR_OVERRIDE = Path(tv_dir)
    if ENHANCED_V3_PARQUET.exists() and not force_rebuild and not force_news_fetch:
        cached = pd.read_parquet(ENHANCED_V3_PARQUET)
        logger.info("Loading cached %s (%d rows).", ENHANCED_V3_PARQUET, len(cached))
        return cached

    prices = _price_backbone()
    if prices.empty:
        raise FileNotFoundError("V3 price backbone is empty. Need Bloomberg / enhanced_data.parquet.")

    tv = load_tradingview_volume(tv_dir)
    tv = clip_v3_window(tv)
    panel = overlay_volume_nearest(prices, tv)

    catl = fetch_catl_futu_volume(fetch=fetch_futu)
    catl = clip_v3_window(catl)
    if not catl.empty:
        panel = overlay_volume_nearest(panel, catl)
    else:
        logger.warning("CATL volume stays 0 (no Futu cache/live rows). OHLC is still Bloomberg.")

    panel = _attach_news(panel, force_news_fetch=force_news_fetch)
    panel = panel.sort_values(["datetime", "ticker"]).reset_index(drop=True)

    tickers = sorted(panel["ticker"].astype(str).unique().tolist())
    vol = pd.to_numeric(panel["volume"], errors="coerce")
    logger.info(
        "V3 panel rows=%d tickers=%s span=%s → %s volume>0=%.1f%%",
        len(panel),
        tickers,
        panel["datetime"].min() if not panel.empty else None,
        panel["datetime"].max() if not panel.empty else None,
        100.0 * float((vol > 0).mean()) if len(vol) else 0.0,
    )
    missing_core = [t for t in CORE_TICKERS if t not in tickers]
    missing_obs = [t for t in OBSERVER_TICKERS if t not in tickers]
    if missing_core:
        logger.warning("V3 panel missing core tickers: %s", missing_core)
    if missing_obs:
        logger.warning("V3 panel missing observers: %s", missing_obs)

    if save and not panel.empty:
        ENHANCED_V3_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(ENHANCED_V3_PARQUET, index=False)
        logger.info("Wrote %s (%d rows). V2 parquet was not touched.", ENHANCED_V3_PARQUET, len(panel))
    return panel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build data/enhanced/enhanced_v3.parquet (does not touch V2).")
    p.add_argument("--force-rebuild", action="store_true", help="Ignore cached enhanced_v3.parquet.")
    p.add_argument("--skip-futu", action="store_true", help="Do not call OpenD; use CATL cache only.")
    p.add_argument("--force-news-fetch", action="store_true", help="Re-query Alpha Vantage for the FULL V3 window (slow; do not use).")
    p.add_argument(
        "--news-days",
        type=int,
        default=0,
        help="If >0, only refresh the last N days of news on the existing V3 parquet (no price rebuild).",
    )
    p.add_argument(
        "--tv-dir",
        type=Path,
        default=None,
        help="Folder with TradingView CSVs (default: data/raw/tradingview).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if int(args.news_days or 0) > 0 and not args.force_rebuild:
        refresh_recent_news_v3(news_days=int(args.news_days), save=True)
    else:
        load_enhanced_v3(
            save=True,
            fetch_futu=not args.skip_futu,
            force_news_fetch=args.force_news_fetch,
            force_rebuild=args.force_rebuild,
            tv_dir=args.tv_dir,
        )
