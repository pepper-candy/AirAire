"""Load Bloomberg 2-year CSVs, overlay Futu incremental files, write unified parquet.

Bloomberg exports live in ``data/raw/bloomberg/``. Column names vary by HP export
(``date`` / ``Date`` / ``PX_LAST`` / ...). Everything is normalized to:

    datetime, ticker, open, high, low, close, volume
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.utils import (
    ALL_TICKERS,
    BLOOMBERG_FILES,
    BLOOMBERG_STALE_DAYS,
    DATA_RAW_BLOOMBERG,
    DATA_RAW_FUTU,
    FUTU_FILES,
    UNIFIED_PARQUET,
    setup_logging,
)

logger = setup_logging("airaire.data_loader")

STANDARD_COLUMNS = ["datetime", "ticker", "open", "high", "low", "close", "volume"]

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


def panel_to_wide(df: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Pivot long OHLCV to a datetime × ticker matrix for a single field."""
    if df.empty:
        return pd.DataFrame(columns=ALL_TICKERS)
    wide = df.pivot_table(index="datetime", columns="ticker", values=field, aggfunc="last")
    return wide.sort_index()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    panel = load_unified()
    print(panel.head() if not panel.empty else "No data yet — drop Bloomberg CSVs into data/raw/bloomberg/")
    print(f"rows={len(panel)} tickers={panel['ticker'].nunique() if not panel.empty else 0}")
