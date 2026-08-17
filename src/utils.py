"""Shared helpers: universe constants, Futu rate limiting, logging, market hours, Telegram."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_BLOOMBERG = PROJECT_ROOT / "data" / "raw" / "bloomberg"
DATA_RAW_FUTU = PROJECT_ROOT / "data" / "raw" / "futu" / "latest"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_ENHANCED = PROJECT_ROOT / "data" / "enhanced"
UNIFIED_PARQUET = DATA_PROCESSED / "unified_data.parquet"
ENHANCED_PARQUET = DATA_ENHANCED / "enhanced_data.parquet"
STATE_PKL = PROJECT_ROOT / "state.pkl"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.zip"

# ---------------------------------------------------------------------------
# Asset universe (PLAN.md §3)
# ---------------------------------------------------------------------------
CORE_TICKERS = ["HK.00700", "HK.03690", "HK.03750", "US.COST", "US.KO"]
OBSERVER_TICKERS = ["HK.HSI", "US.SPX"]
ALL_TICKERS = CORE_TICKERS + OBSERVER_TICKERS

TICKER_NAMES = {
    "HK.00700": "Tencent",
    "HK.03690": "Meituan",
    "HK.03750": "CATL",
    "US.COST": "Costco",
    "US.KO": "Coca-Cola",
    "HK.HSI": "Hang Seng Index",
    "US.SPX": "S&P 500",
}

# Bloomberg HP export filenames vs Futu incremental filenames
BLOOMBERG_FILES = {
    "HK.00700": "0700_HK_1min.csv",
    "HK.03690": "3690_HK_1min.csv",
    "HK.03750": "3750_HK_1min.csv",
    "US.COST": "COST_US_1min.csv",
    "US.KO": "KO_US_1min.csv",
    "HK.HSI": "HSI_1min.csv",
    "US.SPX": "SPX_1min.csv",
}

FUTU_FILES = {
    "HK.00700": "00700_HK_1min.csv",
    "HK.03690": "03690_HK_1min.csv",
    "HK.03750": "03750_HK_1min.csv",
    "US.COST": "COST_US_1min.csv",
    "US.KO": "KO_US_1min.csv",
    "HK.HSI": "HSI_1min.csv",
    "US.SPX": "SPX_1min.csv",
}

# Alpha Vantage NEWS_SENTIMENT ticker symbols (Academic Full Tier)
AV_TICKERS = {
    "HK.00700": "TCEHY",
    "HK.03690": "MPNGY",
    "HK.03750": "3750.HKG",
    "US.COST": "COST",
    "US.KO": "KO",
}

LOT_SIZES = {
    "HK.00700": 100,
    "HK.03690": 100,
    "HK.03750": 100,
    "US.COST": 1,
    "US.KO": 1,
}

HK_TZ = ZoneInfo("Asia/Hong_Kong")
US_TZ = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Lunar New Year (first day) — used by calendar features
CNY_DATES = (
    date(2024, 2, 10),
    date(2025, 1, 29),
    date(2026, 2, 17),
    date(2027, 2, 6),
    date(2028, 1, 26),
)

FUTU_HOST = os.getenv("FUTU_HOST", "127.0.0.1")
FUTU_PORT = int(os.getenv("FUTU_PORT", "11111"))
INITIAL_CASH = float(os.getenv("INITIAL_CASH", "1000000"))

# Futu OpenD quote/trade: 60 requests / 30 seconds
FUTU_MAX_REQUESTS = 60
FUTU_WINDOW_SECONDS = 30

# Alpha Vantage Academic: ethical cap — once per ticker per 5 minutes
NEWS_MIN_INTERVAL_SECONDS = 5 * 60

BLOOMBERG_STALE_DAYS = 5


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(name: str = "airaire", level: int | None = None) -> logging.Logger:
    """Configure a consistent console logger. Safe to call from every module."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    resolved = level if level is not None else getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logger.setLevel(resolved)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = setup_logging("airaire.utils")


# ---------------------------------------------------------------------------
# Rate limiting (Futu)
# ---------------------------------------------------------------------------
class RateLimiter:
    """Sliding-window limiter. Default matches Futu OpenD: 60 requests / 30s."""

    def __init__(self, max_requests: int = FUTU_MAX_REQUESTS, window_seconds: float = FUTU_WINDOW_SECONDS) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request slot is free, then consume one slot."""
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - self.window_seconds
                while self._hits and self._hits[0] <= cutoff:
                    self._hits.popleft()
                if len(self._hits) < self.max_requests:
                    self._hits.append(now)
                    return
                sleep_for = self.window_seconds - (now - self._hits[0]) + 0.01
            logger.debug("Futu RateLimiter sleeping %.2fs (%d/%d in window)", sleep_for, self.max_requests, self.max_requests)
            time.sleep(max(sleep_for, 0.01))


# ---------------------------------------------------------------------------
# Market hours (HK UTC+8, US Eastern UTC-4/UTC-5 via zoneinfo DST)
# ---------------------------------------------------------------------------
def _is_weekday(local_dt: datetime) -> bool:
    return local_dt.weekday() < 5


def is_hk_market_open(now: datetime | None = None) -> bool:
    """HK cash session: 09:30–12:00 and 13:00–16:00 HKT, weekdays."""
    local = (now or datetime.now(tz=UTC)).astimezone(HK_TZ)
    if not _is_weekday(local):
        return False
    t = local.time()
    morning = t >= datetime.strptime("09:30", "%H:%M").time() and t < datetime.strptime("12:00", "%H:%M").time()
    afternoon = t >= datetime.strptime("13:00", "%H:%M").time() and t < datetime.strptime("16:00", "%H:%M").time()
    return morning or afternoon


def is_us_market_open(now: datetime | None = None) -> bool:
    """US cash session: 09:30–16:00 America/New_York (DST-aware), weekdays."""
    local = (now or datetime.now(tz=UTC)).astimezone(US_TZ)
    if not _is_weekday(local):
        return False
    t = local.time()
    return t >= datetime.strptime("09:30", "%H:%M").time() and t < datetime.strptime("16:00", "%H:%M").time()


def ticker_market(ticker: str) -> str:
    if ticker.startswith("HK."):
        return "HK"
    if ticker.startswith("US."):
        return "US"
    raise ValueError(f"Unknown market for ticker {ticker}")


def is_ticker_market_open(ticker: str, now: datetime | None = None) -> bool:
    market = ticker_market(ticker)
    if market == "HK":
        return is_hk_market_open(now)
    return is_us_market_open(now)


def any_core_market_open(now: datetime | None = None) -> bool:
    return is_hk_market_open(now) or is_us_market_open(now)


def seconds_until_next_open(now: datetime | None = None) -> int:
    """Sleep hint when both HK and US cash sessions are closed."""
    now_utc = (now or datetime.now(tz=UTC)).astimezone(UTC)
    candidates: list[datetime] = []
    for tz, open_hm in ((HK_TZ, (9, 30)), (US_TZ, (9, 30))):
        local = now_utc.astimezone(tz)
        open_dt = local.replace(hour=open_hm[0], minute=open_hm[1], second=0, microsecond=0)
        if open_dt <= local:
            open_dt += timedelta(days=1)
        while open_dt.weekday() >= 5:
            open_dt += timedelta(days=1)
        candidates.append(open_dt.astimezone(UTC))
    nxt = min(candidates)
    return max(int((nxt - now_utc).total_seconds()), 30)


def days_until_holiday(ref: date, holiday: date) -> int:
    """Non-negative days until the next occurrence of a fixed holiday."""
    this_year = holiday.replace(year=ref.year)
    if this_year < ref:
        this_year = holiday.replace(year=ref.year + 1)
    return (this_year - ref).days


def days_until_cny(ref: date) -> int:
    future = [d for d in CNY_DATES if d >= ref]
    if not future:
        return 365
    return (future[0] - ref).days


def calendar_feature_vector(ref: date | None = None) -> list[float]:
    """Normalized calendar effects: dow, month, days-to Christmas / CNY / National Day."""
    ref = ref or datetime.now(tz=HK_TZ).date()
    christmas = date(ref.year, 12, 25)
    national_day = date(ref.year, 10, 1)
    return [
        ref.weekday() / 6.0,
        ref.month / 12.0,
        days_until_holiday(ref, christmas) / 365.0,
        days_until_cny(ref) / 365.0,
        days_until_holiday(ref, national_day) / 365.0,
    ]


# ---------------------------------------------------------------------------
# Telegram alerts (placeholder that becomes live once env vars are set)
# ---------------------------------------------------------------------------
def send_telegram_alert(message: str) -> bool:
    """Send a Telegram message. No-ops with a log line if token/chat are missing."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        logger.info("Telegram alert skipped (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID unset): %s", message)
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
        resp.raise_for_status()
        logger.info("Telegram alert sent.")
        return True
    except requests.RequestException as exc:
        logger.warning("Telegram alert failed: %s", exc)
        return False
