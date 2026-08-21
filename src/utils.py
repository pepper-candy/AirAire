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
DATA_RAW_NEWS = PROJECT_ROOT / "data" / "raw" / "news"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_ENHANCED = PROJECT_ROOT / "data" / "enhanced"
UNIFIED_PARQUET = DATA_PROCESSED / "unified_data.parquet"
ENHANCED_PARQUET = DATA_ENHANCED / "enhanced_data.parquet"
STATE_PKL = PROJECT_ROOT / "state.pkl"
MODELS_DIR = PROJECT_ROOT / "models"
NEWS_MODELS_DIR = MODELS_DIR / "news"
NEWS_GPU_V2_MODELS_DIR = MODELS_DIR / "news_gpu_v2"
PRICE_ONLY_MODELS_DIR = MODELS_DIR / "price-only"
BEST_MODEL_PATH = MODELS_DIR / "best_model.zip"
# Paper-trading brain (Window 113, Calmar 2.05). Copied from checkpoint_2026-08-12.zip.
INFERENCE_MODEL_PATH = NEWS_GPU_V2_MODELS_DIR / "best_model.zip"
PRICE_ONLY_BEST_CHECKPOINT = PRICE_ONLY_MODELS_DIR / "checkpoint_2026-04-02.zip"
# Resurrection goldens in models/news_gpu_v2. Training must not clobber these;
# only an explicit Telegram Promote / --promote-zip may copy onto best_model.zip.
PROTECTED_INFERENCE_ZIPS = frozenset(
    {
        "best_model.zip",
        "checkpoint_2026-08-12.zip",
        "checkpoint_2026-08-18.zip",
    }
)


def is_protected_inference_artifact(path: Path) -> bool:
    """True if ``path`` is a Phase-4 golden inside ``models/news_gpu_v2``."""
    path = Path(path)
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")
    if path.name not in PROTECTED_INFERENCE_ZIPS:
        return False
    try:
        path.resolve().relative_to(NEWS_GPU_V2_MODELS_DIR.resolve())
        return True
    except ValueError:
        return False

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
    "HK.00700": "0700_HK_10min.csv",
    "HK.03690": "3690_HK_10min.csv",
    "HK.03750": "3750_HK_10min.csv",
    "US.COST": "COST_US_10min.csv",
    "US.KO": "KO_US_10min.csv",
    "HK.HSI": "HSI_10min.csv",
    "US.SPX": "SPX_10min.csv",
}

FUTU_FILES = {
    "HK.00700": "00700_HK_10min.csv",
    "HK.03690": "03690_HK_10min.csv",
    "HK.03750": "03750_HK_10min.csv",
    "US.COST": "COST_US_10min.csv",
    "US.KO": "KO_US_10min.csv",
    "HK.HSI": "HSI_10min.csv",
    "US.SPX": "SPX_10min.csv",
}

# Alpha Vantage NEWS_SENTIMENT ticker symbols (Academic Full Tier).
# Dots are illegal in this endpoint (alphanumeric / : / _ / - only).
AV_TICKERS = {
    "HK.00700": "TCEHY",
    "HK.03690": "MPNGY",
    "HK.03750": "300750",
    "US.COST": "COST",
    "US.KO": "KO",
}

# Extra symbols tried when the primary AV ticker returns an empty feed.
AV_TICKER_ALIASES: dict[str, tuple[str, ...]] = {
    "HK.00700": ("TCEHY",),
    "HK.03690": ("MPNGY",),
    "HK.03750": ("300750",),
    "US.COST": ("COST",),
    "US.KO": ("KO",),
}

# Topic fallback used only when every ticker alias returns zero articles.
AV_TOPIC_FALLBACK = {
    "HK.00700": "technology",
    "HK.03690": "technology",
    "HK.03750": "energy_transportation,manufacturing",
    "US.COST": "retail_wholesale",
    "US.KO": "retail_wholesale",
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

# Alpha Vantage education / paid NEWS_SENTIMENT: 75 calls / 60s.
# Set NEWS_HISTORICAL_INTERVAL (seconds per call) only if AV returns frequency notes.
# Live poller defaults to 60s so it matches --poll-seconds 60 (5 tickers/min << 75).
AV_MAX_REQUESTS = 75
AV_WINDOW_SECONDS = 60
NEWS_MIN_INTERVAL_SECONDS = int(os.getenv("NEWS_MIN_INTERVAL_SECONDS", "60"))
NEWS_HISTORICAL_INTERVAL_SECONDS = float(os.getenv("NEWS_HISTORICAL_INTERVAL", "0") or 0)
NEWS_HEADLINE_WINDOW = 10
NEWS_HISTORY_CHUNK_DAYS = 7  # smaller than 14 so liquid names do not hit the 1000-article cap
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

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
    # HK has a lunch break (12:00–13:00). Include 13:00 so we wake for the afternoon
    # session instead of aiming at the next 09:30 only.
    for tz, open_hm in ((HK_TZ, (9, 30)), (HK_TZ, (13, 0)), (US_TZ, (9, 30))):
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
TELEGRAM_CALLBACK_PROMOTE = "airaire_promote"
TELEGRAM_CALLBACK_KEEP = "airaire_keep"


def telegram_auth() -> tuple[str, str] | None:
    """Return (bot_token, chat_id) or None if .env is not configured."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return None
    return token, chat_id


def send_telegram_alert(message: str, reply_markup: dict | None = None) -> bool:
    """Send a Telegram message. No-ops with a log line if token/chat are missing."""
    return send_telegram_message(message, reply_markup=reply_markup) is not None


def send_telegram_message(message: str, reply_markup: dict | None = None) -> dict | None:
    """POST sendMessage. Returns Telegram's ``result`` object (includes message_id)."""
    auth = telegram_auth()
    if auth is None:
        logger.info("Telegram alert skipped (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID unset): %s", message)
        return None
    token, chat_id = auth
    payload: dict = {"chat_id": chat_id, "text": message}
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    data = _telegram_post(token, "sendMessage", payload, timeout=15)
    if data is None:
        return None
    logger.info("Telegram alert sent.")
    return data.get("result") if isinstance(data.get("result"), dict) else data


def _telegram_post(token: str, method: str, payload: dict, *, timeout: float) -> dict | None:
    url = f"https://api.telegram.org/bot{token}/{method}"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
    except requests.RequestException as exc:
        logger.warning("Telegram %s failed: %s", method, exc)
        return None
    except ValueError as exc:
        logger.warning("Telegram %s returned non-JSON: %s", method, exc)
        return None
    if not body.get("ok"):
        logger.warning("Telegram %s rejected: %s", method, body)
        return None
    return body


def drain_telegram_updates() -> int:
    """Discard pending updates so a stale button tap cannot promote today's zip."""
    auth = telegram_auth()
    if auth is None:
        return 0
    token, _ = auth
    body = _telegram_post(token, "getUpdates", {"timeout": 0}, timeout=15)
    if not body:
        return 0
    results = body.get("result") or []
    if not results:
        return 0
    last_id = int(results[-1]["update_id"])
    _telegram_post(token, "getUpdates", {"timeout": 0, "offset": last_id + 1}, timeout=15)
    return last_id + 1


def wait_telegram_callback(*, timeout_seconds: int, allowed: tuple[str, ...] = ()) -> str | None:
    """Long-poll ``getUpdates`` until a callback from our chat arrives or time runs out.

    Returns the ``callback_data`` string, or None on timeout / missing credentials.
    Only callbacks from ``TELEGRAM_CHAT_ID`` whose data is in ``allowed`` count.
    """
    auth = telegram_auth()
    if auth is None:
        return None
    token, chat_id = auth
    allowed_set = set(allowed) if allowed else {TELEGRAM_CALLBACK_PROMOTE, TELEGRAM_CALLBACK_KEEP}
    deadline = time.monotonic() + max(int(timeout_seconds), 1)
    offset = drain_telegram_updates()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        poll = int(min(25, max(1, remaining)))
        body = _telegram_post(
            token,
            "getUpdates",
            {"timeout": poll, "offset": offset, "allowed_updates": ["callback_query"]},
            timeout=poll + 10,
        )
        if not body:
            time.sleep(min(2.0, max(0.5, remaining)))
            continue
        for upd in body.get("result") or []:
            offset = int(upd.get("update_id", 0)) + 1
            query = upd.get("callback_query") or {}
            data = str(query.get("data") or "")
            from_chat = str((query.get("message") or {}).get("chat", {}).get("id") or query.get("from", {}).get("id") or "")
            if from_chat != str(chat_id):
                continue
            qid = query.get("id")
            if qid:
                _telegram_post(
                    token,
                    "answerCallbackQuery",
                    {"callback_query_id": qid},
                    timeout=10,
                )
            if data in allowed_set:
                msg = query.get("message") or {}
                mid = msg.get("message_id")
                if mid is not None:
                    _telegram_post(
                        token,
                        "editMessageReplyMarkup",
                        {"chat_id": chat_id, "message_id": mid, "reply_markup": {"inline_keyboard": []}},
                        timeout=10,
                    )
                return data
        # Empty poll or irrelevant updates — keep waiting until the deadline.
