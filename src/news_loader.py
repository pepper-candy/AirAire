"""Historical Alpha Vantage NEWS_SENTIMENT loader (PLAN.md §4D / PHRASE-4).

Education / paid tier is 75 calls / 60s. Fine-tune always re-queries the latest
window before PPO so new bars are not trained on forward-filled stale scores.

* Fetch + cache article-level sentiment under ``data/raw/news/``
* Reduce to a 10-minute series: mean of the last 10 headlines
* Warn and continue (zeros) if the API is missing or fails
* Daily fine-tune uses ``force_fetch`` on the recent window; older cache stays

Alpha Vantage ``time_published`` is treated as a naive clock (same convention as
Bloomberg bars). A few hours of timezone skew still leaves the same-session
signal intact at 10-minute resolution.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils import (
    ALPHA_VANTAGE_URL,
    AV_MAX_REQUESTS,
    AV_TICKER_ALIASES,
    AV_TICKERS,
    AV_TOPIC_FALLBACK,
    AV_WINDOW_SECONDS,
    CORE_TICKERS,
    DATA_RAW_NEWS,
    NEWS_HEADLINE_WINDOW,
    NEWS_HISTORICAL_INTERVAL_SECONDS,
    NEWS_HISTORY_CHUNK_DAYS,
    RateLimiter,
    setup_logging,
)

load_dotenv()
logger = setup_logging("airaire.news_loader")

NEWS_COLUMNS = ["datetime", "ticker", "sentiment_score"]
ARTICLE_COLUMNS = [
    "datetime",
    "ticker",
    "av_symbol",
    "sentiment_score",
    "relevance_score",
    "overall_sentiment_score",
    "title",
    "url",
    "time_published",
    "source",
]
FETCH_META_PATH = DATA_RAW_NEWS / "fetch_meta.json"
NEWS_ALL_PARQUET = DATA_RAW_NEWS / "news_all.parquet"
_AV_SYMBOL_RE = re.compile(r"^[A-Za-z0-9:_-]+$")

if NEWS_HISTORICAL_INTERVAL_SECONDS > 0:
    _av_limiter = RateLimiter(max_requests=1, window_seconds=NEWS_HISTORICAL_INTERVAL_SECONDS)
else:
    _av_limiter = RateLimiter(max_requests=AV_MAX_REQUESTS, window_seconds=AV_WINDOW_SECONDS)

# Phrase-4 example filenames, accepted as extra cache locations.
_LEGACY_NEWS_CSV = {
    "HK.00700": "news_0700_HK.csv",
    "HK.03690": "news_3690_HK.csv",
    "HK.03750": "news_3750_HK.csv",
    "US.COST": "news_COST_US.csv",
    "US.KO": "news_KO_US.csv",
}


def _api_key() -> str:
    return (os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY") or "").strip()


def _naive(ts: pd.Series | pd.Timestamp | str) -> pd.Series | pd.Timestamp:
    if isinstance(ts, pd.Series):
        parsed = pd.to_datetime(ts, utc=True, errors="coerce")
        return parsed.dt.tz_localize(None)
    parsed = pd.Timestamp(ts)
    if parsed.tzinfo is not None:
        return parsed.tz_convert("UTC").tz_localize(None)
    return pd.Timestamp(parsed)


def parse_av_time(value: object) -> pd.Timestamp:
    """Parse Alpha Vantage ``YYYYMMDDTHHMM[SS]`` (or anything pandas understands)."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return pd.NaT
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return pd.NaT
    for fmt, n in (("%Y%m%dT%H%M%S", 15), ("%Y%m%dT%H%M", 13)):
        chunk = text[:n]
        try:
            return pd.Timestamp(datetime.strptime(chunk, fmt))
        except ValueError:
            continue
    return pd.to_datetime(text, errors="coerce")


def av_time_str(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.strftime("%Y%m%dT%H%M")


def news_csv_path(ticker: str) -> Path:
    return DATA_RAW_NEWS / f"news_{ticker.replace('.', '_')}.csv"


def articles_csv_path(ticker: str) -> Path:
    return DATA_RAW_NEWS / f"articles_{ticker.replace('.', '_')}.csv"


def candidate_news_csv_paths(ticker: str) -> list[Path]:
    paths = [news_csv_path(ticker)]
    legacy = _LEGACY_NEWS_CSV.get(ticker)
    if legacy:
        paths.append(DATA_RAW_NEWS / legacy)
    return paths


def av_symbols_for(ticker: str) -> tuple[str, ...]:
    aliases = AV_TICKER_ALIASES.get(ticker)
    raw = aliases if aliases else ((AV_TICKERS.get(ticker) or ticker),)
    valid = tuple(symbol for symbol in raw if _AV_SYMBOL_RE.fullmatch(symbol))
    skipped = [symbol for symbol in raw if symbol not in valid]
    if skipped:
        logger.warning("Skipping invalid Alpha Vantage tickers for %s: %s", ticker, skipped)
    return valid


def _empty_articles() -> pd.DataFrame:
    return pd.DataFrame(columns=ARTICLE_COLUMNS)


def _empty_news() -> pd.DataFrame:
    return pd.DataFrame(columns=NEWS_COLUMNS)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s (%s); starting with empty metadata.", path, exc)
        return {}


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _normalize_articles(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_articles()
    out = df.copy()
    for col in ARTICLE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["datetime"] = out["datetime"].apply(parse_av_time)
    missing_dt = out["datetime"].isna() & out["time_published"].notna()
    if missing_dt.any():
        out.loc[missing_dt, "datetime"] = out.loc[missing_dt, "time_published"].apply(parse_av_time)
    if ticker:
        out["ticker"] = ticker
    out["sentiment_score"] = pd.to_numeric(out["sentiment_score"], errors="coerce")
    out["relevance_score"] = pd.to_numeric(out["relevance_score"], errors="coerce")
    out["overall_sentiment_score"] = pd.to_numeric(out["overall_sentiment_score"], errors="coerce")
    out = out.dropna(subset=["datetime", "sentiment_score"])
    out["sentiment_score"] = out["sentiment_score"].clip(-1.0, 1.0)
    out = out.sort_values("datetime").drop_duplicates(subset=["ticker", "datetime", "url", "title"], keep="last")
    return out[ARTICLE_COLUMNS].reset_index(drop=True)


def _normalize_news(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_news()
    out = df.copy()
    colmap = {c.lower().strip(): c for c in out.columns}
    if "sentiment_score" not in out.columns:
        for alias in ("news_score", "score", "ticker_sentiment_score"):
            if alias in colmap:
                out["sentiment_score"] = out[colmap[alias]]
                break
    if "datetime" not in out.columns:
        for alias in ("date", "timestamp", "time"):
            if alias in colmap:
                out["datetime"] = out[colmap[alias]]
                break
    if "ticker" not in out.columns and ticker:
        out["ticker"] = ticker
    if "datetime" not in out.columns or "sentiment_score" not in out.columns:
        logger.warning("News CSV missing datetime/sentiment_score columns: %s", list(df.columns))
        return _empty_news()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    if getattr(out["datetime"].dt, "tz", None) is not None:
        out["datetime"] = _naive(out["datetime"])
    out["sentiment_score"] = pd.to_numeric(out["sentiment_score"], errors="coerce").clip(-1.0, 1.0)
    out = out.dropna(subset=["datetime", "ticker", "sentiment_score"])
    out = out.sort_values(["ticker", "datetime"]).drop_duplicates(subset=["ticker", "datetime"], keep="last")
    return out[NEWS_COLUMNS].reset_index(drop=True)


def load_news_csv(ticker: str, path: Path | None = None) -> pd.DataFrame:
    """Load a local 10-minute news CSV for one ticker. Empty frame if missing."""
    candidates = [path] if path is not None else candidate_news_csv_paths(ticker)
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        try:
            raw = pd.read_csv(candidate)
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Failed to read %s: %s", candidate, exc)
            continue
        frame = _normalize_news(raw, ticker=ticker)
        if frame.empty:
            continue
        logger.info("Loaded cached news %s: %d rows (%s → %s)", ticker, len(frame), frame["datetime"].min(), frame["datetime"].max())
        return frame
    return _empty_news()


def load_articles_csv(ticker: str) -> pd.DataFrame:
    path = articles_csv_path(ticker)
    if not path.exists():
        return _empty_articles()
    try:
        raw = pd.read_csv(path)
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to read articles cache %s: %s", path, exc)
        return _empty_articles()
    return _normalize_articles(raw, ticker=ticker)


def _save_articles(ticker: str, articles: pd.DataFrame) -> None:
    DATA_RAW_NEWS.mkdir(parents=True, exist_ok=True)
    path = articles_csv_path(ticker)
    _normalize_articles(articles, ticker=ticker).to_csv(path, index=False)


def _save_news_csv(ticker: str, news: pd.DataFrame) -> None:
    DATA_RAW_NEWS.mkdir(parents=True, exist_ok=True)
    path = news_csv_path(ticker)
    _normalize_news(news, ticker=ticker).to_csv(path, index=False)


def _payload_error(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return f"non-dict response ({type(payload).__name__})"
    if payload.get("feed") is not None:
        return None
    for key in ("Error Message", "Information", "Note"):
        msg = payload.get(key)
        if msg:
            return str(msg)
    return f"unexpected keys: {list(payload)[:8]}"


def _is_rate_limit(message: str) -> bool:
    lower = message.lower()
    return "frequency" in lower or "rate limit" in lower or "thank you for using alpha vantage" in lower or "premium" in lower


def _request_news_sentiment(params: dict[str, Any], retries: int = 4) -> dict[str, Any]:
    last_error = "unknown error"
    for attempt in range(retries):
        _av_limiter.acquire()
        try:
            resp = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            last_error = str(exc)
            logger.warning("NEWS_SENTIMENT HTTP error (attempt %d/%d): %s", attempt + 1, retries, exc)
            time.sleep(min(30 * (attempt + 1), 120))
            continue
        err = _payload_error(payload)
        if err is None:
            return payload
        last_error = err
        if _is_rate_limit(err):
            sleep_for = 60 * (attempt + 1)
            logger.warning("Alpha Vantage rate/limit note; sleeping %ds. %s", sleep_for, err[:160])
            time.sleep(sleep_for)
            continue
        logger.warning("NEWS_SENTIMENT payload error: %s", err[:200])
        return payload if isinstance(payload, dict) else {}
    logger.warning("NEWS_SENTIMENT giving up: %s", last_error[:200])
    return {}


def _extract_articles(payload: dict[str, Any], ticker: str, av_symbol: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in payload.get("feed") or []:
        published = parse_av_time(item.get("time_published"))
        overall = item.get("overall_sentiment_score", 0.0)
        try:
            overall_f = float(overall)
        except (TypeError, ValueError):
            overall_f = 0.0
        ticker_score = None
        relevance = None
        for ts in item.get("ticker_sentiment") or []:
            if str(ts.get("ticker", "")).upper() == av_symbol.upper():
                try:
                    ticker_score = float(ts.get("ticker_sentiment_score", overall_f))
                except (TypeError, ValueError):
                    ticker_score = overall_f
                try:
                    relevance = float(ts.get("relevance_score", 0.0))
                except (TypeError, ValueError):
                    relevance = 0.0
                break
        if ticker_score is None:
            ticker_score = overall_f
            relevance = 0.0
        rows.append(
            {
                "datetime": published,
                "ticker": ticker,
                "av_symbol": av_symbol,
                "sentiment_score": float(np.clip(ticker_score, -1.0, 1.0)),
                "relevance_score": relevance if relevance is not None else 0.0,
                "overall_sentiment_score": float(np.clip(overall_f, -1.0, 1.0)),
                "title": item.get("title") or "",
                "url": item.get("url") or "",
                "time_published": item.get("time_published") or "",
                "source": item.get("source") or item.get("source_domain") or "",
            }
        )
    return _normalize_articles(pd.DataFrame(rows), ticker=ticker)


def _chunk_range(start: pd.Timestamp, end: pd.Timestamp, days: int = 14) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end < start:
        start, end = end, start
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    step = pd.Timedelta(days=days)
    while cursor <= end:
        nxt = min(cursor + step, end)
        chunks.append((cursor, nxt))
        cursor = nxt + pd.Timedelta(minutes=1)
    return chunks or [(start, end)]


def _range_covered(meta: dict[str, Any], ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    entry = meta.get(ticker) or {}
    try:
        got_from = pd.Timestamp(entry.get("queried_from"))
        got_to = pd.Timestamp(entry.get("queried_to"))
    except (TypeError, ValueError):
        return False
    if pd.isna(got_from) or pd.isna(got_to):
        return False
    return got_from <= start and got_to >= end


def _update_meta(meta: dict[str, Any], ticker: str, start: pd.Timestamp, end: pd.Timestamp, n_articles: int) -> None:
    prev = meta.get(ticker) or {}
    try:
        prev_from = pd.Timestamp(prev["queried_from"]) if prev.get("queried_from") else start
        prev_to = pd.Timestamp(prev["queried_to"]) if prev.get("queried_to") else end
    except (TypeError, ValueError):
        prev_from, prev_to = start, end
    meta[ticker] = {
        "queried_from": str(min(prev_from, start)),
        "queried_to": str(max(prev_to, end)),
        "symbols": list(av_symbols_for(ticker)),
        "n_articles": int(n_articles),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def fetch_historical_news(
    ticker: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    api_key: str | None = None,
    *,
    force_fetch: bool = False,
) -> pd.DataFrame:
    """Fetch historical news sentiment for one core ticker.

    Returns article-level rows (datetime, ticker, sentiment_score, ...).
    Caches to ``data/raw/news/articles_*.csv``. On API failure, returns whatever
    is already cached (possibly empty).
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cached = load_articles_csv(ticker)
    meta = _load_json(FETCH_META_PATH)
    key = api_key or _api_key()

    if not force_fetch and _range_covered(meta, ticker, start, end) and not cached.empty:
        logger.info("%s news cache covers %s → %s (%d articles). Skipping API.", ticker, start.date(), end.date(), len(cached))
        return cached

    if not key:
        logger.warning("ALPHAVANTAGE_API_KEY unset — using cached/zero news for %s.", ticker)
        return cached

    frames = [cached] if not cached.empty else []
    symbols = av_symbols_for(ticker)
    fetched_any = False

    for symbol in symbols:
        for chunk_start, chunk_end in _chunk_range(start, end, days=NEWS_HISTORY_CHUNK_DAYS):
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "time_from": av_time_str(chunk_start),
                "time_to": av_time_str(chunk_end),
                "sort": "EARLIEST",
                "limit": 1000,
                "apikey": key,
            }
            logger.info("Fetching NEWS_SENTIMENT %s as %s  %s → %s", ticker, symbol, chunk_start, chunk_end)
            payload = _request_news_sentiment(params)
            if not isinstance(payload, dict) or "feed" not in payload:
                msg = ""
                if isinstance(payload, dict):
                    msg = str(payload.get("Error Message") or payload.get("Information") or payload.get("Note") or "")
                if "invalid ticker" in msg.lower():
                    logger.warning("Giving up on symbol %s for %s (invalid ticker format).", symbol, ticker)
                    break
                continue
            fetched_any = True
            feed = payload.get("feed") or []
            if not feed:
                continue
            chunk = _extract_articles(payload, ticker, symbol)
            if chunk.empty:
                continue
            frames.append(chunk)
            if len(feed) >= 1000:
                # API truncated this window — split it on the next pass by shrinking.
                logger.info("%s/%s hit limit=1000 in %s→%s; extra articles may exist in this window.", ticker, symbol, chunk_start.date(), chunk_end.date())

    have_articles = any(not f.empty for f in frames)
    if not have_articles:
        topic = AV_TOPIC_FALLBACK.get(ticker)
        if topic:
            logger.warning("%s: no ticker-specific articles. Falling back to topics=%s (overall sentiment).", ticker, topic)
            for chunk_start, chunk_end in _chunk_range(start, end, days=NEWS_HISTORY_CHUNK_DAYS):
                params = {
                    "function": "NEWS_SENTIMENT",
                    "topics": topic,
                    "time_from": av_time_str(chunk_start),
                    "time_to": av_time_str(chunk_end),
                    "sort": "EARLIEST",
                    "limit": 200,
                    "apikey": key,
                }
                payload = _request_news_sentiment(params)
                if not isinstance(payload, dict) or "feed" not in payload:
                    continue
                fetched_any = True
                feed = payload.get("feed") or []
                if not feed:
                    continue
                # Tag topic articles onto this ticker using overall_sentiment_score.
                topic_rows = []
                for item in feed:
                    published = parse_av_time(item.get("time_published"))
                    try:
                        score = float(item.get("overall_sentiment_score", 0.0))
                    except (TypeError, ValueError):
                        score = 0.0
                    topic_rows.append(
                        {
                            "datetime": published,
                            "ticker": ticker,
                            "av_symbol": f"TOPIC:{topic}",
                            "sentiment_score": float(np.clip(score, -1.0, 1.0)),
                            "relevance_score": 0.15,
                            "overall_sentiment_score": float(np.clip(score, -1.0, 1.0)),
                            "title": item.get("title") or "",
                            "url": item.get("url") or "",
                            "time_published": item.get("time_published") or "",
                            "source": item.get("source") or "",
                        }
                    )
                frames.append(_normalize_articles(pd.DataFrame(topic_rows), ticker=ticker))

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        logger.warning("No news articles for %s in %s → %s. Training will use 0.0 sentiment.", ticker, start.date(), end.date())
        if fetched_any:
            _update_meta(meta, ticker, start, end, 0)
            _save_json(FETCH_META_PATH, meta)
        return _empty_articles()

    combined = _normalize_articles(pd.concat(frames, ignore_index=True), ticker=ticker)
    _save_articles(ticker, combined)
    _update_meta(meta, ticker, start, end, len(combined))
    _save_json(FETCH_META_PATH, meta)
    logger.info("Cached %d articles for %s (%s → %s)", len(combined), ticker, combined["datetime"].min(), combined["datetime"].max())
    return combined


def articles_to_bar_scores(
    articles: pd.DataFrame,
    freq: str = "10min",
    headline_window: int = NEWS_HEADLINE_WINDOW,
) -> pd.DataFrame:
    """Collapse articles to a 10-minute series: rolling mean of the last N headlines.

    Matches live ``NewsPoller`` (average of up to 10 headlines).
    """
    if articles is None or articles.empty:
        return _empty_news()
    arts = _normalize_articles(articles)
    if arts.empty:
        return _empty_news()
    frames: list[pd.DataFrame] = []
    for ticker, group in arts.groupby("ticker"):
        g = group.sort_values("datetime").copy()
        # Relevance-weighted mean when scores exist; otherwise equal-weight last N.
        rel = g["relevance_score"].fillna(0.0).clip(lower=0.0)
        g["_w"] = np.where(rel > 0, rel, 1.0)
        g["_wx"] = g["_w"] * g["sentiment_score"]
        g["roll_w"] = g["_w"].rolling(headline_window, min_periods=1).sum()
        g["roll_wx"] = g["_wx"].rolling(headline_window, min_periods=1).sum()
        g["roll_mean"] = g["roll_wx"] / g["roll_w"].replace(0.0, np.nan)
        g["roll_mean"] = g["roll_mean"].fillna(g["sentiment_score"])
        series = g.set_index("datetime")["roll_mean"].sort_index()
        series = series[~series.index.duplicated(keep="last")]
        resampled = series.resample(freq).last().dropna()
        frame = resampled.rename("sentiment_score").reset_index()
        frame["ticker"] = ticker
        frames.append(frame)
    if not frames:
        return _empty_news()
    return _normalize_news(pd.concat(frames, ignore_index=True))


def _headline_rows(articles: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    """Newest-first article rows for the dashboard. One AV call already fetched these."""
    if articles is None or articles.empty:
        return []
    arts = articles.sort_values("datetime", ascending=False).head(int(limit))
    rows: list[dict[str, Any]] = []
    for rec in arts.to_dict(orient="records"):
        rows.append(
            {
                "title": str(rec.get("title") or ""),
                "source": str(rec.get("source") or ""),
                "url": str(rec.get("url") or ""),
                "time_published": str(rec.get("time_published") or ""),
                "sentiment_score": float(rec.get("sentiment_score") or 0.0),
            }
        )
    return rows


def latest_ticker_news(
    ticker: str,
    api_key: str | None = None,
    limit: int = NEWS_HEADLINE_WINDOW,
) -> tuple[float, list[dict[str, Any]]]:
    """Live score plus the headlines that produced it. One NEWS_SENTIMENT call."""
    if ticker not in CORE_TICKERS:
        return 0.0, []
    key = api_key or _api_key()
    if not key:
        logger.warning("NewsPoller: ALPHAVANTAGE_API_KEY unset — sentiment for %s stays 0.", ticker)
        return 0.0, []
    for symbol in av_symbols_for(ticker):
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "limit": int(limit),
            "apikey": key,
        }
        try:
            _av_limiter.acquire()
            resp = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            logger.warning("NewsPoller HTTP error for %s (%s): %s", ticker, symbol, exc)
            continue
        err = _payload_error(payload)
        if err is not None:
            logger.warning("NewsPoller unexpected payload for %s (%s): %s", ticker, symbol, err[:160])
            continue
        articles = _extract_articles(payload, ticker, symbol)
        if articles.empty:
            continue
        scores = articles["sentiment_score"].to_numpy(dtype=np.float64)
        avg = float(np.clip(np.mean(scores) if len(scores) else 0.0, -1.0, 1.0))
        headlines = _headline_rows(articles, limit)
        logger.info("NewsPoller %s (%s) score=%.3f n_headlines=%d", ticker, symbol, avg, len(scores))
        return avg, headlines
    logger.warning("NewsPoller: no headlines for %s across aliases %s.", ticker, av_symbols_for(ticker))
    return 0.0, []


def latest_ticker_score(ticker: str, api_key: str | None = None, limit: int = NEWS_HEADLINE_WINDOW) -> float:
    """One-shot live score used by ``NewsPoller``. Tries aliases; returns 0 on failure."""
    score, _ = latest_ticker_news(ticker, api_key=api_key, limit=limit)
    return score


def load_all_news(
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    force_fetch: bool = False,
) -> pd.DataFrame:
    """Load last-10-headline sentiment for all CORE_TICKERS, resampled to 10 minutes."""
    start = pd.Timestamp(start_date) if start_date is not None else pd.Timestamp("2026-02-24")
    end = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.now()
    DATA_RAW_NEWS.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for ticker in CORE_TICKERS:
        articles = fetch_historical_news(ticker, start, end, force_fetch=force_fetch)
        if articles.empty:
            local = load_news_csv(ticker)
            if not local.empty:
                frames.append(local)
                continue
            logger.warning("%s: no articles and no CSV cache — sentiment will be 0.0.", ticker)
            continue
        bars = articles_to_bar_scores(articles)
        _save_news_csv(ticker, bars)
        if not bars.empty:
            frames.append(bars)

    if not frames:
        logger.warning("load_all_news: empty result for %s → %s. Returning zeros schema.", start, end)
        return _empty_news()

    news = _normalize_news(pd.concat(frames, ignore_index=True))
    news = news[(news["datetime"] >= start - pd.Timedelta(days=2)) & (news["datetime"] <= end + pd.Timedelta(days=1))]
    DATA_RAW_NEWS.mkdir(parents=True, exist_ok=True)
    news.to_parquet(NEWS_ALL_PARQUET, index=False)
    logger.info(
        "News panel: %d 10-min rows, tickers=%s, span=%s → %s",
        len(news),
        sorted(news["ticker"].unique().tolist()),
        news["datetime"].min() if not news.empty else None,
        news["datetime"].max() if not news.empty else None,
    )
    return news.reset_index(drop=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch / cache Alpha Vantage news sentiment")
    p.add_argument("--start", default="2026-02-24")
    p.add_argument("--end", default=None, help="Inclusive end date (default: now)")
    p.add_argument("--force", action="store_true", help="Re-query Alpha Vantage even if cache covers the range.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    end = args.end or str(pd.Timestamp.now().date())
    panel = load_all_news(args.start, end, force_fetch=args.force)
    print(panel.head() if not panel.empty else "No news rows (API missing or empty feed).")
    print(f"rows={len(panel)} tickers={panel['ticker'].nunique() if not panel.empty else 0}")
    if not panel.empty:
        print(panel.groupby("ticker")["sentiment_score"].agg(["count", "mean", "min", "max"]).to_string())
