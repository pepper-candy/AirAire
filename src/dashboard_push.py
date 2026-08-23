"""One-way snapshot push from the GPU VM to Supabase.

Never raises into the trade path. If DASHBOARD_PUSH_URL / DASHBOARD_PUSH_KEY
are unset, this is a silent no-op so paper trading still runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import requests
from dotenv import load_dotenv

from src.utils import (
    CORE_TICKERS,
    DATA_LOGS,
    HK_TZ,
    INITIAL_CASH,
    TRADES_JSONL,
    setup_logging,
)

load_dotenv()
logger = setup_logging("airaire.dashboard_push")

SNAPSHOTS_TABLE = os.getenv("DASHBOARD_SNAPSHOTS_TABLE", "bot_snapshots").strip() or "bot_snapshots"
FILL_HISTORY = 50
STALE_AFTER_SECONDS = 180
_skip_logged = False


def _push_url() -> str:
    return (os.getenv("DASHBOARD_PUSH_URL") or "").strip().rstrip("/")


def _push_key() -> str:
    return (os.getenv("DASHBOARD_PUSH_KEY") or "").strip()


def configured() -> bool:
    return bool(_push_url() and _push_key())


def _rest_base() -> str:
    url = _push_url()
    if url.endswith("/rest/v1"):
        return url
    return f"{url}/rest/v1"


def _headers(*, prefer: str | None = None) -> dict[str, str]:
    key = _push_key()
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def hk_now_iso() -> str:
    return datetime.now(tz=HK_TZ).isoformat()


def append_fill(record: dict[str, Any]) -> None:
    """Append one blotter line. Failures are logged; they must not break orders."""
    try:
        DATA_LOGS.mkdir(parents=True, exist_ok=True)
        with TRADES_JSONL.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not append %s (%s).", TRADES_JSONL.name, exc)


def load_recent_fills(limit: int = FILL_HISTORY) -> list[dict[str, Any]]:
    if not TRADES_JSONL.exists():
        return []
    try:
        lines = TRADES_JSONL.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Could not read %s (%s).", TRADES_JSONL, exc)
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            rec = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows[-int(limit) :]


def build_snapshot(
    *,
    kind: str = "live",
    cash: float,
    equity: float,
    holdings: dict[str, float],
    last_action: dict[str, float],
    last_reason: str,
    last_bar_datetime: str,
    news_scores: dict[str, float],
    headlines: dict[str, list[dict[str, Any]]] | None = None,
    headline_baskets: list[dict[str, Any]] | None = None,
    fills: list[dict[str, Any]] | None = None,
    initial_cash: float = INITIAL_CASH,
    updated_at: str | None = None,
) -> dict[str, Any]:
    book_holdings = {t: float(holdings.get(t, 0.0)) for t in CORE_TICKERS}
    book_action = {t: float(last_action.get(t, 0.0)) for t in CORE_TICKERS}
    book_news = {t: float(news_scores.get(t, 0.0)) for t in CORE_TICKERS}
    book_headlines = {t: list((headlines or {}).get(t, [])) for t in CORE_TICKERS}
    return {
        "kind": kind,
        "updated_at": updated_at or hk_now_iso(),
        "cash": float(cash),
        "equity": float(equity),
        "holdings": book_holdings,
        "last_action": book_action,
        "last_reason": str(last_reason or ""),
        "last_bar_datetime": str(last_bar_datetime or ""),
        "news_scores": book_news,
        "headlines": book_headlines,
        "headline_baskets": list(headline_baskets or []),
        "fills": list(fills if fills is not None else load_recent_fills()),
        "initial_cash": float(initial_cash),
        "pnl": float(equity) - float(initial_cash),
    }


def snapshot_from_state(state: Any, headlines: dict[str, list[dict[str, Any]]] | None = None, kind: str = "live") -> dict[str, Any]:
    return build_snapshot(
        kind=kind,
        cash=float(getattr(state, "cash", INITIAL_CASH)),
        equity=float(getattr(state, "equity", getattr(state, "cash", INITIAL_CASH))),
        holdings=dict(getattr(state, "holdings", {}) or {}),
        last_action=dict(getattr(state, "last_action", {}) or {}),
        last_reason=str(getattr(state, "last_reason", "") or ""),
        last_bar_datetime=str(getattr(state, "last_bar_datetime", "") or ""),
        news_scores=dict(getattr(state, "news_scores", {}) or {}),
        headlines=headlines,
        fills=load_recent_fills(),
    )


def push_snapshot(payload: dict[str, Any]) -> bool:
    """POST one row. Returns False on skip/failure. Never raises."""
    global _skip_logged
    if not configured():
        if not _skip_logged:
            logger.info("Dashboard push skipped (DASHBOARD_PUSH_URL / DASHBOARD_PUSH_KEY unset).")
            _skip_logged = True
        return False
    body = {"kind": str(payload.get("kind") or "live"), "payload": payload}
    url = f"{_rest_base()}/{SNAPSHOTS_TABLE}"
    try:
        resp = requests.post(url, headers=_headers(prefer="return=minimal"), json=body, timeout=15)
        if resp.status_code >= 400:
            logger.warning(
                "Dashboard push HTTP %s: %s",
                resp.status_code,
                (resp.text or "")[:240],
            )
            return False
        logger.info("Dashboard snapshot pushed kind=%s equity=%.2f", body["kind"], float(payload.get("equity") or 0.0))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Dashboard push failed (%s). Trading continues.", exc)
        return False


def push_live_snapshot(
    state: Any,
    headlines: dict[str, list[dict[str, Any]]] | None = None,
    kind: str = "live",
) -> bool:
    try:
        return push_snapshot(snapshot_from_state(state, headlines=headlines, kind=kind))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Dashboard snapshot build failed (%s). Trading continues.", exc)
        return False


def ping() -> int:
    """GET the table so the operator knows SQL + Data API exposure worked."""
    if not configured():
        logger.error("Set DASHBOARD_PUSH_URL and DASHBOARD_PUSH_KEY in .env first.")
        return 2
    url = f"{_rest_base()}/{SNAPSHOTS_TABLE}?select=id,created_at,kind&order=created_at.desc&limit=1"
    try:
        resp = requests.get(url, headers=_headers(), timeout=15)
    except Exception as exc:  # noqa: BLE001
        logger.error("Ping failed: %s", exc)
        return 1
    if resp.status_code == 404:
        logger.error(
            "Table %s not reachable (HTTP 404). Run the SQL in guide/PHRASE-5.md, then expose "
            "public.%s in Supabase Data API (Automatically expose new tables is off).",
            SNAPSHOTS_TABLE,
            SNAPSHOTS_TABLE,
        )
        return 1
    if resp.status_code >= 400:
        logger.error("Ping HTTP %s: %s", resp.status_code, (resp.text or "")[:240])
        return 1
    logger.info("Dashboard ping ok HTTP %s body=%s", resp.status_code, (resp.text or "")[:200])
    return 0


def seed_updated_at(latest_datetime: Any) -> str:
    """Seed rows must look stale so Monday live pushes own the freshness pip."""
    now = datetime.now(tz=HK_TZ)
    stale_before = now - timedelta(seconds=STALE_AFTER_SECONDS + 30)
    stamp = None
    if latest_datetime is not None:
        try:
            ts = latest_datetime
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            if isinstance(ts, datetime):
                stamp = ts if ts.tzinfo is not None else ts.replace(tzinfo=HK_TZ)
                if stamp.tzinfo is not None and stamp.tzinfo != HK_TZ:
                    stamp = stamp.astimezone(HK_TZ)
        except (TypeError, ValueError):
            stamp = None
    if stamp is None or stamp > stale_before:
        stamp = stale_before
    return stamp.isoformat()


def snapshot_from_news_cache(seed: dict[str, Any] | None = None) -> dict[str, Any]:
    if seed is None:
        from src.news_loader import seed_news_from_cache

        seed = seed_news_from_cache()
    last_dt = seed.get("latest_datetime")
    last_bar = ""
    if last_dt is not None:
        last_bar = str(last_dt).split(".")[0][:19]
    return build_snapshot(
        kind="seed",
        cash=INITIAL_CASH,
        equity=INITIAL_CASH,
        holdings={ticker: 0.0 for ticker in CORE_TICKERS},
        last_action={ticker: 0.0 for ticker in CORE_TICKERS},
        last_reason="Seeded from training news cache. Paper book starts Monday.",
        last_bar_datetime=last_bar,
        news_scores=seed["news_scores"],
        headlines=seed["headlines"],
        headline_baskets=seed.get("headline_baskets") or [],
        fills=[],
        updated_at=seed_updated_at(last_dt),
    )


def seed_news(*, dry_run: bool = False) -> int:
    """INSERT one kind=seed row from local article CSVs. Does not call Alpha Vantage."""
    from src.news_loader import seed_news_from_cache

    seed = seed_news_from_cache()
    for ticker, info in seed["counts"].items():
        logger.info(
            "Seed cache %s articles=%d ticker_specific=%d headlines=%d score=%.3f",
            ticker,
            info["articles"],
            info["ticker_specific"],
            info["headlines"],
            float(seed["news_scores"].get(ticker, 0.0)),
        )
    for basket in seed.get("headline_baskets") or []:
        logger.info(
            "Seed basket %s members=%s headlines=%d",
            basket.get("title") or basket.get("id"),
            ",".join(basket.get("members") or []),
            len(basket.get("headlines") or []),
        )
    payload = snapshot_from_news_cache(seed)
    if dry_run:
        logger.info(
            "Seed dry-run kind=%s updated_at=%s headlines=%s",
            payload["kind"],
            payload["updated_at"],
            {ticker: len(rows) for ticker, rows in payload["headlines"].items()},
        )
        return 0
    if not configured():
        logger.error("Set DASHBOARD_PUSH_URL and DASHBOARD_PUSH_KEY in .env first.")
        return 2
    if not push_snapshot(payload):
        logger.error("Seed push failed.")
        return 1
    logger.info("Seed snapshot inserted. Refresh the blotter; the strip should stay stale.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AirAire dashboard snapshot push")
    p.add_argument("--ping", action="store_true", help="GET bot_snapshots; do not insert a row.")
    p.add_argument(
        "--seed-news",
        action="store_true",
        help="INSERT one kind=seed row from data/raw/news (no Alpha Vantage).",
    )
    p.add_argument("--dry-run", action="store_true", help="With --seed-news: print counts, do not POST.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.ping:
        sys.exit(ping())
    if args.seed_news:
        sys.exit(seed_news(dry_run=args.dry_run))
    logger.error("Use: python -m src.dashboard_push --ping | --seed-news [--dry-run]")
    sys.exit(2)
