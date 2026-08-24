"""V4 panel: Bloomberg OHLC for 5 core + HSI + SPX, volume forced to 0, V2 news.

Does not write enhanced_data.parquet or enhanced_v3.parquet.
Does not overlay TradingView volume (that was V3).

    python -m src.data_loader_v4
    python -m src.data_loader_v4 --skip-futu
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
    overlay_live_ohlcv,
)
from src.futu_codes import FUTU_KLINE_ALIASES_V4
from src.utils import (
    ALL_TICKERS,
    CORE_TICKERS,
    DATA_ENHANCED,
    ENHANCED_PARQUET,
    HK_TZ,
    MODELS_DIR,
    OBSERVER_TICKERS,
    setup_logging,
)

logger = setup_logging("airaire.data_loader_v4")

ENHANCED_V4_PARQUET = DATA_ENHANCED / "enhanced_v4.parquet"
NEWS_GPU_V4_MODELS_DIR = MODELS_DIR / "news_gpu_v4"


def _zero_volume(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if "volume" in out.columns:
        out["volume"] = 0.0
    else:
        out["volume"] = 0.0
    return out


def _copy_v2_news(panel: pd.DataFrame) -> pd.DataFrame:
    """Observers have no AV ticker feed. Core names copy V2 parquet news_score."""
    out = panel.copy()
    out["news_score"] = 0.0
    if not ENHANCED_PARQUET.exists():
        logger.warning("No V2 enhanced parquet — news_score stays 0.")
        return out
    v2 = pd.read_parquet(ENHANCED_PARQUET)
    if "news_score" not in v2.columns:
        logger.warning("V2 parquet has no news_score — V4 news stays 0.")
        return out
    news = v2.loc[v2["ticker"].astype(str).isin(CORE_TICKERS), ["datetime", "ticker", "news_score"]].copy()
    news["datetime"] = pd.to_datetime(news["datetime"], errors="coerce")
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    merged = out.drop(columns=["news_score"]).merge(news, on=["datetime", "ticker"], how="left")
    merged["news_score"] = pd.to_numeric(merged["news_score"], errors="coerce")
    merged["news_score"] = merged.groupby("ticker", group_keys=False)["news_score"].ffill().fillna(0.0)
    logger.info(
        "Copied V2 news_score onto V4 core names. Observers stay 0. coverage=%.1f%%",
        100.0 * float((merged["news_score"].abs() > 1e-12).mean()) if len(merged) else 0.0,
    )
    return merged


def fetch_v4_futu_overlay(start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
    """OHLC from OpenD for 7 training tickers. Volume is discarded later."""
    frames: list[pd.DataFrame] = []
    end = end or datetime.now(tz=HK_TZ).replace(tzinfo=None)
    for train_code in ALL_TICKERS:
        aliases = FUTU_KLINE_ALIASES_V4.get(train_code, (train_code,))
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
            logger.warning("V4 Futu overlay: no klines for %s (tried %s).", train_code, aliases)
            continue
        if used != train_code:
            logger.info("V4 Futu klines %s remapped → %s (%d rows).", used, train_code, len(got))
        frames.append(got)
    if not frames:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def load_enhanced_v4(
    *,
    save: bool = True,
    fetch_futu: bool = True,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    if ENHANCED_V4_PARQUET.exists() and not force_rebuild:
        cached = pd.read_parquet(ENHANCED_V4_PARQUET)
        logger.info("Loading cached %s (%d rows).", ENHANCED_V4_PARQUET, len(cached))
        if fetch_futu:
            live = fetch_v4_futu_overlay()
            if live is not None and not live.empty:
                cached = overlay_live_ohlcv(cached, live, now=datetime.now(tz=HK_TZ).replace(tzinfo=None))
                cached = _zero_volume(cached)
                if save:
                    ENHANCED_V4_PARQUET.parent.mkdir(parents=True, exist_ok=True)
                    cached.to_parquet(ENHANCED_V4_PARQUET, index=False)
                    logger.info("Updated %s with Futu OHLC overlay; volume kept 0.", ENHANCED_V4_PARQUET)
        return cached

    prices = load_bloomberg()
    if prices.empty:
        raise FileNotFoundError("V4 Bloomberg backbone is empty. Need data/raw/bloomberg/*.csv")
    missing = [t for t in ALL_TICKERS if t not in set(prices["ticker"].astype(str))]
    if missing:
        logger.warning("Bloomberg missing %s — V4 cube will ffill/zero those names.", missing)

    prices = _zero_volume(prices)
    panel = _copy_v2_news(prices)

    if fetch_futu:
        live = fetch_v4_futu_overlay()
        if live is not None and not live.empty:
            panel = overlay_live_ohlcv(panel, live, now=datetime.now(tz=HK_TZ).replace(tzinfo=None))
            panel = _zero_volume(panel)

    panel = panel.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    tickers = sorted(panel["ticker"].astype(str).unique().tolist())
    logger.info(
        "V4 panel rows=%d tickers=%s span=%s → %s volume forced 0",
        len(panel),
        tickers,
        panel["datetime"].min() if not panel.empty else None,
        panel["datetime"].max() if not panel.empty else None,
    )
    missing_core = [t for t in CORE_TICKERS if t not in tickers]
    missing_obs = [t for t in OBSERVER_TICKERS if t not in tickers]
    if missing_core:
        logger.warning("V4 panel missing core: %s", missing_core)
    if missing_obs:
        logger.warning("V4 panel missing observers: %s", missing_obs)

    if save and not panel.empty:
        ENHANCED_V4_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(ENHANCED_V4_PARQUET, index=False)
        logger.info("Wrote %s. V2/V3 parquets were not touched.", ENHANCED_V4_PARQUET)
    return panel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build data/enhanced/enhanced_v4.parquet (Bloomberg, volume=0).")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--skip-futu", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    load_enhanced_v4(save=True, fetch_futu=not args.skip_futu, force_rebuild=args.force_rebuild)
