"""Load Bloomberg 2-year CSVs, overlay Futu incremental files, write unified parquet.

Bloomberg exports live in ``data/raw/bloomberg/``. Column names vary by HP export
(``date`` / ``Date`` / ``PX_LAST`` / ...). Everything is normalized to:

    datetime, ticker, open, high, low, close, volume
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.utils import (
    ALL_TICKERS,
    BAR_MINUTES,
    BLOOMBERG_FILES,
    BLOOMBERG_STALE_DAYS,
    CORE_TICKERS,
    DATA_RAW_BLOOMBERG,
    DATA_RAW_FUTU,
    ENHANCED_PARQUET,
    FUTU_FILES,
    FUTU_HOST,
    FUTU_PORT,
    HK_TZ,
    UNIFIED_PARQUET,
    RateLimiter,
    market_naive_now,
    setup_logging,
)

logger = setup_logging("airaire.data_loader")

STANDARD_COLUMNS = ["datetime", "ticker", "open", "high", "low", "close", "volume"]
ENHANCED_COLUMNS = STANDARD_COLUMNS + ["news_score"]

# Map messy Bloomberg / Futu headers onto the canonical names (case-insensitive).
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "datetime": (
        "datetime",
        "date_time",
        "timestamp",
        "date/time",
        "date time",
        "dates",
        "date",
        "time",
    ),
    "open": ("open", "px_open", "open_price", "o", "px last open"),
    "high": ("high", "px_high", "high_price", "h"),
    "low": ("low", "px_low", "low_price", "l"),
    "close": ("close", "px_last", "px_close", "close_price", "last", "adj_close", "adj close", "c"),
    "volume": ("volume", "px_volume", "vol", "turnover", "v", "size"),
}

DATE_ONLY_ALIASES = ("date", "dates", "trade_date", "tradedate")
TIME_ONLY_ALIASES = ("time", "times", "trade_time", "tradetime")


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def _lookup_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    return None


def normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Coerce a raw Bloomberg/Futu CSV frame into the standard OHLCV schema."""
    df = _strip_columns(df)
    cols = list(df.columns)

    date_col = _lookup_column(cols, DATE_ONLY_ALIASES)
    time_col = _lookup_column(cols, TIME_ONLY_ALIASES)
    datetime_col = _lookup_column(cols, COLUMN_ALIASES["datetime"])

    out = pd.DataFrame()
    if date_col and time_col and date_col != time_col:
        out["datetime"] = pd.to_datetime(
            df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(),
            errors="coerce",
        )
    elif datetime_col:
        out["datetime"] = pd.to_datetime(df[datetime_col], errors="coerce")
    else:
        raise ValueError(f"{ticker}: no date/datetime column in {cols}")

    for field in ("open", "high", "low", "close", "volume"):
        src = _lookup_column(cols, COLUMN_ALIASES[field])
        if src is None:
            logger.warning("%s: missing '%s' column; filling 0. Headers=%s", ticker, field, cols)
            out[field] = 0.0
        else:
            out[field] = pd.to_numeric(df[src], errors="coerce")

    out["ticker"] = ticker
    out = out.dropna(subset=["datetime"])
    out = out.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    return out[STANDARD_COLUMNS].reset_index(drop=True)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def load_ticker_csv(path: Path, ticker: str) -> pd.DataFrame:
    if not path.exists():
        logger.warning("CSV not found for %s: %s", ticker, path)
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    raw = _read_csv(path)
    if raw.empty:
        logger.warning("Empty CSV for %s: %s", ticker, path)
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    # 自动检测无表头：如果第一列名包含日期分隔符，则重新读取并指定列名
    first_col_name = str(raw.columns[0])   # 添加这一行！
    if '/' in first_col_name or '-' in first_col_name or ':' in first_col_name:
        raw = pd.read_csv(path, header=None, names=['datetime', 'open', 'high', 'low', 'close', 'volume'])

    return normalize_ohlcv(raw, ticker)


def check_bloomberg_freshness(df: pd.DataFrame, ticker: str, stale_days: int = BLOOMBERG_STALE_DAYS) -> bool:
    """Return True if data is fresh. Logs the exact PLAN.md warning when stale."""
    if df.empty:
        return False
    last_ts = pd.Timestamp(df["datetime"].max())
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    else:
        last_ts = last_ts.tz_convert("UTC")
    now = pd.Timestamp.now(tz="UTC")
    age = now - last_ts
    if age > pd.Timedelta(days=stale_days):
        logger.warning(
            "Bloomberg data stale. Please update manually. ticker=%s last_timestamp=%s age_days=%.1f",
            ticker,
            last_ts.isoformat(),
            age.total_seconds() / 86400.0,
        )
        return False
    logger.info("%s Bloomberg last bar %s (age %.1f days) — fresh.", ticker, last_ts.isoformat(), age.total_seconds() / 86400.0)
    return True


def load_bloomberg(data_dir: Path | None = None) -> pd.DataFrame:
    """Load and concatenate the 2-year Bloomberg base for all 7 tickers."""
    data_dir = data_dir or DATA_RAW_BLOOMBERG
    frames: list[pd.DataFrame] = []
    any_stale = False
    for ticker, filename in BLOOMBERG_FILES.items():
        frame = load_ticker_csv(data_dir / filename, ticker)
        if frame.empty:
            continue
        if not check_bloomberg_freshness(frame, ticker):
            any_stale = True
        frames.append(frame)
        logger.info("Loaded Bloomberg %s: %d rows (%s → %s)", ticker, len(frame), frame["datetime"].min(), frame["datetime"].max())
    if any_stale:
        logger.warning("Bloomberg data stale. Please update manually.")
    if not frames:
        logger.warning("No Bloomberg CSVs found under %s", data_dir)
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def load_futu_latest(data_dir: Path | None = None) -> pd.DataFrame:
    """Load Futu incremental 30-day CSVs if present."""
    data_dir = data_dir or DATA_RAW_FUTU
    frames: list[pd.DataFrame] = []
    for ticker, filename in FUTU_FILES.items():
        frame = load_ticker_csv(data_dir / filename, ticker)
        if frame.empty:
            continue
        frames.append(frame)
        logger.info("Loaded Futu %s: %d rows (%s → %s)", ticker, len(frame), frame["datetime"].min(), frame["datetime"].max())
    if not frames:
        logger.info("No Futu incremental CSVs under %s (Bloomberg-only merge).", data_dir)
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def merge_bloomberg_futu(bloomberg: pd.DataFrame, futu: pd.DataFrame) -> pd.DataFrame:
    """Overlay Futu on Bloomberg. On timestamp collisions, Futu (more recent feed) wins."""
    if bloomberg.empty and futu.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    if futu.empty:
        merged = bloomberg.copy()
    elif bloomberg.empty:
        merged = futu.copy()
    else:
        merged = pd.concat([bloomberg, futu], ignore_index=True)
        merged = merged.sort_values(["ticker", "datetime"])
        # last = Futu row when both sources share a timestamp
        merged = merged.drop_duplicates(subset=["ticker", "datetime"], keep="last")
    return merged.sort_values(["ticker", "datetime"]).reset_index(drop=True)


def load_unified(
    bloomberg_dir: Path | None = None,
    futu_dir: Path | None = None,
    save: bool = True,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Full Phase 1 pipeline: Bloomberg → stale check → Futu overlay → parquet."""
    bloomberg = load_bloomberg(bloomberg_dir)
    futu = load_futu_latest(futu_dir)
    unified = merge_bloomberg_futu(bloomberg, futu)
    logger.info(
        "Unified panel: %d rows, tickers=%s, span=%s → %s",
        len(unified),
        sorted(unified["ticker"].unique().tolist()) if not unified.empty else [],
        unified["datetime"].min() if not unified.empty else None,
        unified["datetime"].max() if not unified.empty else None,
    )
    if save and not unified.empty:
        path = output_path or UNIFIED_PARQUET
        path.parent.mkdir(parents=True, exist_ok=True)
        unified.to_parquet(path, index=False)
        logger.info("Wrote %s", path)
    return unified


def load_processed(path: Path | None = None) -> pd.DataFrame:
    path = path or UNIFIED_PARQUET
    if not path.exists():
        logger.warning("Processed parquet missing (%s); running load_unified().", path)
        return load_unified(save=True, output_path=path)
    return pd.read_parquet(path)


def _naive_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed.dt.tz_localize(None)


def merge_price_news(price_df: pd.DataFrame, news_df: pd.DataFrame | None) -> pd.DataFrame:
    """As-of merge news onto each (datetime, ticker) price bar, then forward-fill."""
    if price_df is None or price_df.empty:
        return pd.DataFrame(columns=ENHANCED_COLUMNS)
    prices = price_df.copy()
    prices["datetime"] = _naive_datetime(prices["datetime"])
    prices = prices.dropna(subset=["datetime"]).sort_values(["ticker", "datetime"]).reset_index(drop=True)

    if news_df is None or news_df.empty:
        prices["news_score"] = 0.0
        logger.warning("No news rows to merge — news_score filled with 0.0.")
        return prices

    news = news_df.copy()
    score_col = "sentiment_score" if "sentiment_score" in news.columns else "news_score"
    if score_col not in news.columns:
        prices["news_score"] = 0.0
        logger.warning("news_df missing sentiment_score/news_score — filling 0.0.")
        return prices
    news["datetime"] = _naive_datetime(news["datetime"])
    news["sentiment_score"] = pd.to_numeric(news[score_col], errors="coerce").clip(-1.0, 1.0)
    news = news.dropna(subset=["datetime", "ticker", "sentiment_score"])
    news = news.sort_values(["ticker", "datetime"]).drop_duplicates(subset=["ticker", "datetime"], keep="last")

    merged_parts: list[pd.DataFrame] = []
    for ticker, g_price in prices.groupby("ticker", sort=False):
        g_price = g_price.sort_values("datetime")
        g_news = news.loc[news["ticker"] == ticker, ["datetime", "sentiment_score"]].sort_values("datetime")
        if g_news.empty:
            part = g_price.copy()
            part["news_score"] = 0.0
            merged_parts.append(part)
            continue
        part = pd.merge_asof(
            g_price,
            g_news,
            on="datetime",
            direction="backward",
        )
        part["news_score"] = part["sentiment_score"].ffill().fillna(0.0)
        part = part.drop(columns=["sentiment_score"])
        merged_parts.append(part)

    enhanced = pd.concat(merged_parts, ignore_index=True).sort_values(["datetime", "ticker"]).reset_index(drop=True)
    core = enhanced[enhanced["ticker"].isin(CORE_TICKERS)]
    if core.empty:
        coverage = 0.0
    else:
        coverage = float((core["news_score"].abs() > 1e-12).mean())
    per_ticker = (
        enhanced[enhanced["ticker"].isin(CORE_TICKERS)]
        .groupby("ticker")["news_score"]
        .apply(lambda s: float((s.abs() > 1e-12).mean()))
        .to_dict()
    )
    logger.info(
        "News coverage: %.1f%% of core-ticker bars have |news_score| > 0  per-ticker=%s",
        100.0 * coverage,
        {k: f"{100.0 * v:.1f}%" for k, v in per_ticker.items()},
    )
    return enhanced


def load_enhanced_data(
    bloomberg_dir: Path | None = None,
    news_df: pd.DataFrame | None = None,
    *,
    save: bool = True,
    force_news_fetch: bool = False,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Load unified price data, merge last-10-headline news scores, write enhanced parquet.

    Returns columns: datetime, ticker, open, high, low, close, volume, news_score.
    """
    if bloomberg_dir is not None:
        prices = load_unified(bloomberg_dir=bloomberg_dir, save=True)
    else:
        prices = load_processed()

    if news_df is None:
        from src.news_loader import load_all_news

        if prices.empty:
            start = end = None
        else:
            start = prices["datetime"].min()
            end = prices["datetime"].max()
        try:
            news_df = load_all_news(start, end, force_fetch=force_news_fetch)
        except Exception as exc:  # noqa: BLE001 — news is optional; never block price training
            logger.warning("News fetch failed (%s). Continuing with news_score=0.0.", exc)
            news_df = pd.DataFrame(columns=["datetime", "ticker", "sentiment_score"])

    enhanced = merge_price_news(prices, news_df)
    if save and not enhanced.empty:
        path = output_path or ENHANCED_PARQUET
        path.parent.mkdir(parents=True, exist_ok=True)
        enhanced.to_parquet(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(enhanced))
    return enhanced


def panel_to_wide(df: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Pivot long OHLCV to a datetime × ticker matrix for a single field."""
    if df.empty:
        return pd.DataFrame(columns=ALL_TICKERS)
    wide = df.pivot_table(index="datetime", columns="ticker", values=field, aggfunc="last")
    return wide.sort_index()


# ---------------------------------------------------------------------------
# Live Futu 10-min bars (inference catch-up + daily fine-tune)
# ---------------------------------------------------------------------------
FUTU_KLINE_LOOKBACK_DAYS = 30
_FUTU_KLINE_MAX_PAGES = 20
_futu_history_limiter = RateLimiter()


def _naive_hk_ts(value: datetime | pd.Timestamp | str | None, default: datetime | None = None) -> pd.Timestamp:
    """HK wall-clock Timestamp with tz stripped.

    Panel CSVs / parquet store naive local-market clocks. Comparing those to a
    tz-aware ``now`` raises ``TypeError: Cannot compare tz-naive and tz-aware``.
    Convert into Asia/Hong_Kong first so a UTC caller still maps to the same
    calendar day, then drop ``tzinfo`` (keep the HK wall clock).
    """
    if value is None:
        value = default or datetime.now()
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = pd.Timestamp(ts.tz_convert(HK_TZ).replace(tzinfo=None))
    return ts


def _as_date_str(value: datetime | pd.Timestamp | str | None, default: datetime | None = None) -> str:
    return _naive_hk_ts(value, default).strftime("%Y-%m-%d")


def _coerce_now(now: datetime | pd.Timestamp | None) -> datetime | None:
    if now is None:
        return None
    to_pydt = getattr(now, "to_pydatetime", None)
    if callable(to_pydt):
        return to_pydt()
    return now if isinstance(now, datetime) else None


def drop_incomplete_klines(
    df: pd.DataFrame | None,
    now: datetime | pd.Timestamp | None = None,
    bar_minutes: int = BAR_MINUTES,
) -> pd.DataFrame:
    """Drop forming candles: keep a row only when bar_start + 10 minutes <= now.

    Futu ``request_history_kline`` usually includes the in-progress 10-min bar
    (HK 09:30 at 09:31). Training never saw those stubs. Per-ticker clocks:
    HK bars vs Asia/Hong_Kong, US bars vs America/New_York.
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame(columns=STANDARD_COLUMNS)
    if "datetime" not in df.columns or "ticker" not in df.columns:
        return df
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    now_dt = _coerce_now(now)
    kept: list[pd.DataFrame] = []
    dropped = 0
    for ticker, part in out.groupby("ticker", sort=False):
        now_local = pd.Timestamp(market_naive_now(str(ticker), now_dt))
        close_at = part["datetime"] + pd.Timedelta(minutes=bar_minutes)
        mask = close_at.notna() & (close_at <= now_local)
        dropped += int((~mask).sum())
        if mask.any():
            kept.append(part.loc[mask])
    if dropped:
        logger.info(
            "Dropped %d incomplete %d-min kline(s) still forming at %s.",
            dropped,
            bar_minutes,
            now_dt or "wall clock",
        )
    if not kept:
        return out.iloc[0:0].reset_index(drop=True)
    if len(kept) == 1:
        return kept[0].sort_values(["datetime", "ticker"]).reset_index(drop=True)
    return pd.concat(kept, ignore_index=True).sort_values(["datetime", "ticker"]).reset_index(drop=True)


def _klines_to_panel(ticker: str, data: pd.DataFrame) -> pd.DataFrame:
    """Map a Futu ``request_history_kline`` frame onto STANDARD_COLUMNS."""
    if data is None or len(data) == 0:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    time_col = "time_key" if "time_key" in data.columns else None
    if time_col is None:
        logger.warning("Futu kline for %s has no time_key column (%s).", ticker, list(data.columns))
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(data[time_col], errors="coerce"),
            "ticker": ticker,
            "open": pd.to_numeric(data["open"], errors="coerce") if "open" in data.columns else 0.0,
            "high": pd.to_numeric(data["high"], errors="coerce") if "high" in data.columns else 0.0,
            "low": pd.to_numeric(data["low"], errors="coerce") if "low" in data.columns else 0.0,
            "close": pd.to_numeric(data["close"], errors="coerce") if "close" in data.columns else 0.0,
            "volume": pd.to_numeric(data["volume"], errors="coerce") if "volume" in data.columns else 0.0,
        }
    )
    out = out.dropna(subset=["datetime"])
    out["volume"] = out["volume"].fillna(0.0)
    return out[STANDARD_COLUMNS].reset_index(drop=True)


def fetch_futu_history(
    tickers: list[str] | None = None,
    start: datetime | pd.Timestamp | str | None = None,
    end: datetime | pd.Timestamp | str | None = None,
    *,
    quote_ctx: Any = None,
    limiter: RateLimiter | None = None,
    host: str = FUTU_HOST,
    port: int = FUTU_PORT,
    lookback_days: int = FUTU_KLINE_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Pull 10-minute OHLCV from OpenD for ``tickers``.

    Used when the operator has been offline (inference catch-up) or when the
    daily fine-tune needs bars newer than ``enhanced_data.parquet``. Returns an
    empty frame if ``futu-api`` / OpenD is unavailable — callers must tolerate that.
    Timestamps stay naive local-market time, matching Bloomberg/Futu CSVs
    (HK = Beijing, US = Eastern).
    """
    tickers = list(tickers or CORE_TICKERS)
    end_ts = _naive_hk_ts(end)
    if start is None:
        start_ts = end_ts - pd.Timedelta(days=max(int(lookback_days), 1))
    else:
        start_ts = _naive_hk_ts(start)
    start_str = _as_date_str(start_ts)
    end_str = _as_date_str(end_ts)
    limiter = limiter or _futu_history_limiter

    try:
        from futu import AuType, KLType, OpenQuoteContext, RET_OK
    except ImportError:
        logger.warning("futu-api is not installed — skipping live kline fetch.")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    own_ctx = quote_ctx is None
    ctx = quote_ctx
    frames: list[pd.DataFrame] = []
    try:
        if ctx is None:
            limiter.acquire()
            ctx = OpenQuoteContext(host=host, port=port)
        for ticker in tickers:
            page_req_key = None
            ticker_frames: list[pd.DataFrame] = []
            for page in range(_FUTU_KLINE_MAX_PAGES):
                limiter.acquire()
                try:
                    ret, data, page_req_key = ctx.request_history_kline(
                        ticker,
                        start=start_str,
                        end=end_str,
                        ktype=KLType.K_10M,
                        autype=AuType.QFQ,
                        max_count=1000,
                        page_req_key=page_req_key,
                    )
                except Exception as exc:  # noqa: BLE001 — OpenD can drop mid-page
                    logger.warning("request_history_kline(%s) raised %s", ticker, exc)
                    break
                if ret != RET_OK:
                    logger.warning("request_history_kline(%s) failed: %s", ticker, data)
                    break
                part = _klines_to_panel(ticker, data if isinstance(data, pd.DataFrame) else pd.DataFrame())
                if not part.empty:
                    ticker_frames.append(part)
                if page_req_key is None:
                    break
                if page == _FUTU_KLINE_MAX_PAGES - 1:
                    logger.warning("Hit %d-page cap fetching %s 10-min bars.", _FUTU_KLINE_MAX_PAGES, ticker)
            if ticker_frames:
                combined = pd.concat(ticker_frames, ignore_index=True)
                combined = combined.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
                frames.append(combined)
                logger.info(
                    "Futu 10-min %s: %d bars (%s → %s)",
                    ticker,
                    len(combined),
                    combined["datetime"].min(),
                    combined["datetime"].max(),
                )
    finally:
        if own_ctx and ctx is not None:
            try:
                ctx.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Error closing temporary OpenQuoteContext: %s", exc)

    if not frames:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    out = pd.concat(frames, ignore_index=True).sort_values(["ticker", "datetime"]).reset_index(drop=True)
    out = drop_incomplete_klines(out, now=end_ts)
    if out.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    logger.info(
        "Futu history overlay: %d rows, tickers=%s, span=%s → %s",
        len(out),
        sorted(out["ticker"].unique().tolist()),
        out["datetime"].min(),
        out["datetime"].max(),
    )
    return out


def overlay_live_ohlcv(
    panel: pd.DataFrame | None,
    live: pd.DataFrame | None,
    now: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Append live Futu bars onto a long panel. Live rows win on (ticker, datetime).

    ``news_score`` on brand-new bars is forward-filled from the last known
    sentiment so the env never sees NaNs. Missing news stays 0.0.

    Forming 10-min candles (bar still open) are dropped so the env only sees
    completed bars, matching training.
    """
    if panel is None or panel.empty:
        base = pd.DataFrame(columns=STANDARD_COLUMNS)
    else:
        base = panel.copy()
        base["datetime"] = pd.to_datetime(base["datetime"], errors="coerce")

    if live is None or live.empty:
        result = (
            base.reset_index(drop=True)
            if not base.empty
            else pd.DataFrame(columns=list(base.columns) or STANDARD_COLUMNS)
        )
        return drop_incomplete_klines(result, now=now)

    fresh = live.copy()
    fresh["datetime"] = pd.to_datetime(fresh["datetime"], errors="coerce")
    if "news_score" in base.columns and "news_score" not in fresh.columns:
        fresh["news_score"] = pd.NA
    if "sentiment_score" in base.columns and "sentiment_score" not in fresh.columns:
        fresh["sentiment_score"] = pd.NA

    parts = [frame for frame in (base, fresh) if frame is not None and not frame.empty]
    if not parts:
        return drop_incomplete_klines(pd.DataFrame(columns=STANDARD_COLUMNS), now=now)
    if len(parts) == 1:
        merged = parts[0].copy()
    else:
        aligned = []
        cols = list(dict.fromkeys(c for frame in parts for c in frame.columns))
        for frame in parts:
            extra = frame.copy()
            for col in cols:
                if col not in extra.columns:
                    extra[col] = pd.NA
            aligned.append(extra[cols])
        merged = pd.concat(aligned, ignore_index=True, sort=False)
    merged = merged.dropna(subset=["datetime", "ticker"])
    merged = merged.sort_values(["ticker", "datetime"])
    merged = merged.drop_duplicates(subset=["ticker", "datetime"], keep="last")
    for col in ("news_score", "sentiment_score"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
            merged[col] = merged.groupby("ticker", group_keys=False)[col].ffill().fillna(0.0)
    merged = merged.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    return drop_incomplete_klines(merged, now=now)


def persist_enhanced_panel(panel: pd.DataFrame, path: Path | None = None) -> Path | None:
    """Write the (possibly catch-up-extended) enhanced panel. Best-effort."""
    if panel is None or panel.empty:
        return None
    dest = path or ENHANCED_PARQUET
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(dest, index=False)
        logger.info("Updated %s (%d rows, last bar %s)", dest, len(panel), panel["datetime"].max())
        return dest
    except Exception as exc:  # noqa: BLE001 — never block inference on a parquet write
        logger.warning("Could not persist enhanced panel to %s (%s).", dest, exc)
        return None


def default_futu_fetch_start(
    panel: pd.DataFrame | None,
    *,
    now: datetime | pd.Timestamp | None = None,
    lookback_days: int = FUTU_KLINE_LOOKBACK_DAYS,
) -> pd.Timestamp:
    """Start date for an incremental Futu pull: last panel bar minus one day, capped at ``lookback_days``."""
    now_ts = _naive_hk_ts(now)
    floor = now_ts - pd.Timedelta(days=max(int(lookback_days), 1))
    if panel is None or panel.empty or "datetime" not in panel.columns:
        return floor
    last = pd.Timestamp(pd.to_datetime(panel["datetime"], errors="coerce").max())
    if pd.isna(last):
        return floor
    last = _naive_hk_ts(last)
    # One-day overlap so a mid-session restart re-fetches this morning's bars.
    start = last - pd.Timedelta(days=1)
    return max(start, floor)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    panel = load_unified()
    print(panel.head() if not panel.empty else "No data yet — drop Bloomberg CSVs into data/raw/bloomberg/")
    print(f"price rows={len(panel)} tickers={panel['ticker'].nunique() if not panel.empty else 0}")
    enhanced = load_enhanced_data()
    print(enhanced.head() if not enhanced.empty else "Enhanced panel empty.")
    print(f"enhanced rows={len(enhanced)} cols={list(enhanced.columns)}")
