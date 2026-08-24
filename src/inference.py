"""Local paper trader: load state.pkl first, poll news ethically, execute via Futu SIMULATE.

Resume contract (PLAN.md §8):
    1. Load ``state.pkl`` *before* any order.
    2. After every trade, persist holdings / cash / last action.
    3. On shutdown, flush state again.

Startup catch-up (Phase 4):
    If the operator logs in mid-session (e.g. 12:45), fetch missing 10-min bars
    from Futu, jump ``TradingEnv._bar_index`` to the latest completed bar, and
    sync ``state.pkl`` — no orders are placed during catch-up.

Live loop (Phase 4):
    Poll every 60s so we notice a new Futu 10-min close within a minute.
    Forming OpenD candles are dropped. HK and US names each wait out the
    first 10 minutes of their own cash session (HK 09:30 / 13:00, US 09:30
    ET) so the first fill is on a finished kline. Then place SIMULATE orders
    only on a *new* completed bar, a new session's first ready cycle, or
    when news jumps by ``NEWS_RETRADE_DELTA``. Same-bar 1-minute price
    drift does not rebalance.

    ``--predict-now``: one live score (quotes + news), no orders, no ``state.pkl``
    write. Safe while the continuous trader is already running.
"""

from __future__ import annotations

import argparse
import os
import pickle
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from dotenv import load_dotenv

from src.data_loader import (
    default_futu_fetch_start,
    fetch_futu_history,
    load_processed,
    overlay_live_ohlcv,
    persist_enhanced_panel,
)
from src.dashboard_push import append_fill, push_live_snapshot
from src.news_loader import latest_ticker_news
from src.trading_env import TradingEnv
from src.utils import (
    BEST_MODEL_PATH,
    CORE_TICKERS,
    ENHANCED_PARQUET,
    FUTU_HOST,
    FUTU_PORT,
    HK_TZ,
    INITIAL_CASH,
    INFERENCE_MODEL_PATH,
    LOT_SIZES,
    US_INITIAL_CASH,
    NEWS_GPU_V2_MODELS_DIR,
    NEWS_MIN_INTERVAL_SECONDS,
    STATE_PKL,
    TICKER_NAMES,
    RateLimiter,
    any_core_market_open,
    is_cash_open_bar_complete,
    is_kline_complete,
    is_ticker_market_open,
    live_session_clock,
    need_panel_refresh,
    panel_seek_now,
    ready_cash_sessions,
    seconds_until_next_open,
    session_bar_id,
    session_date_iso,
    round_to_tick,
    send_telegram_alert,
    setup_logging,
    ticker_market,
)

load_dotenv()
logger = setup_logging("airaire.inference")

futu_limiter = RateLimiter()


# ---------------------------------------------------------------------------
# Persistent bot state
# ---------------------------------------------------------------------------
@dataclass
class BotState:
    holdings: dict[str, float] = field(default_factory=lambda: {t: 0.0 for t in CORE_TICKERS})
    cash: float = INITIAL_CASH
    us_cash: float = US_INITIAL_CASH
    last_action: dict[str, float] = field(default_factory=lambda: {t: 0.0 for t in CORE_TICKERS})
    last_reason: str = "cold start"
    realized_pnl: float = 0.0
    equity: float = INITIAL_CASH
    news_scores: dict[str, float] = field(default_factory=lambda: {t: 0.0 for t in CORE_TICKERS})
    updated_at: str = ""
    last_bar_datetime: str = ""
    last_order_bar: str = ""
    last_session_ready: dict[str, str] = field(default_factory=dict)
    pending_orders: list[dict[str, Any]] = field(default_factory=list)
    placed_order_ids: list[str] = field(default_factory=list)
    settled_order_ids: list[str] = field(default_factory=list)
    last_buy_px: dict[str, float] = field(default_factory=dict)
    last_sell_px: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BotState":
        base = cls()
        holdings = dict(base.holdings)
        holdings.update({k: float(v) for k, v in (raw.get("holdings") or {}).items() if k in holdings})
        last_action = dict(base.last_action)
        last_action.update({k: float(v) for k, v in (raw.get("last_action") or {}).items() if k in last_action})
        news_scores = dict(base.news_scores)
        news_scores.update({k: float(v) for k, v in (raw.get("news_scores") or {}).items() if k in news_scores})
        last_buy_px = {
            str(k): float(v) for k, v in (raw.get("last_buy_px") or {}).items() if str(k) in holdings
        }
        last_sell_px = {
            str(k): float(v) for k, v in (raw.get("last_sell_px") or {}).items() if str(k) in holdings
        }
        pending = [dict(row) for row in (raw.get("pending_orders") or []) if isinstance(row, dict)]
        placed = [str(x) for x in (raw.get("placed_order_ids") or []) if str(x)][-400:]
        settled = [str(x) for x in (raw.get("settled_order_ids") or []) if str(x)][-400:]
        return cls(
            holdings=holdings,
            cash=float(raw.get("cash", INITIAL_CASH)),
            us_cash=float(raw.get("us_cash", US_INITIAL_CASH)),
            last_action=last_action,
            last_reason=str(raw.get("last_reason", "loaded from disk")),
            realized_pnl=float(raw.get("realized_pnl", 0.0)),
            equity=float(raw.get("equity", raw.get("cash", INITIAL_CASH))),
            news_scores=news_scores,
            updated_at=str(raw.get("updated_at", "")),
            last_bar_datetime=str(raw.get("last_bar_datetime", "")),
            last_order_bar=str(raw.get("last_order_bar", "")),
            last_session_ready={
                str(k): str(v) for k, v in (raw.get("last_session_ready") or {}).items()
            },
            pending_orders=pending,
            placed_order_ids=placed,
            settled_order_ids=settled,
            last_buy_px=last_buy_px,
            last_sell_px=last_sell_px,
        )


def load_state(path: Path = STATE_PKL) -> BotState:
    """MUST be called before any Futu order so we never double-buy after a restart."""
    if not path.exists():
        logger.info("No %s found — starting from a flat book (cash=%.2f).", path, INITIAL_CASH)
        state = BotState()
        state.updated_at = datetime.now(timezone.utc).isoformat()
        return state
    with path.open("rb") as fh:
        raw = pickle.load(fh)
    if isinstance(raw, BotState):
        # Re-hydrate through from_dict so older pickles missing new fields still load.
        payload = {}
        for key in (
            "holdings",
            "cash",
            "us_cash",
            "last_action",
            "last_reason",
            "realized_pnl",
            "equity",
            "news_scores",
            "updated_at",
            "last_bar_datetime",
            "last_order_bar",
            "last_session_ready",
            "pending_orders",
            "placed_order_ids",
            "settled_order_ids",
            "last_buy_px",
            "last_sell_px",
        ):
            if hasattr(raw, key):
                payload[key] = getattr(raw, key)
        state = BotState.from_dict(payload)
    elif isinstance(raw, dict):
        state = BotState.from_dict(raw)
    else:
        logger.warning("Unrecognized state.pkl payload (%s); using defaults.", type(raw))
        state = BotState()
    logger.info(
        "Loaded state.pkl updated_at=%s last_bar=%s cash=%.2f holdings=%s last_action=%s pnl=%.2f",
        state.updated_at,
        state.last_bar_datetime,
        state.cash,
        state.holdings,
        state.last_action,
        state.realized_pnl,
    )
    return state


def save_state(state: BotState, path: Path = STATE_PKL) -> None:
    state.updated_at = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(state.to_dict(), fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    logger.info("Wrote %s (cash=%.2f equity=%.2f holdings=%s)", path.name, state.cash, state.equity, state.holdings)


# ---------------------------------------------------------------------------
# Alpha Vantage news — 60s per ticker by default (education / paid 75/min)
# ---------------------------------------------------------------------------
class NewsPoller:
    """Live Alpha Vantage scores. Default 60s per ticker (education 75/min)."""

    def __init__(self, api_key: str | None = None, min_interval: int = NEWS_MIN_INTERVAL_SECONDS) -> None:
        self.api_key = (api_key or os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY") or "").strip()
        self.min_interval = min_interval
        self._last_call: dict[str, float] = {}
        self._cache: dict[str, float] = {t: 0.0 for t in CORE_TICKERS}
        self._headlines: dict[str, list[dict[str, Any]]] = {t: [] for t in CORE_TICKERS}

    def headlines_by_ticker(self) -> dict[str, list[dict[str, Any]]]:
        return {t: list(self._headlines.get(t, [])) for t in CORE_TICKERS}

    def fetch(self, ticker: str, now: datetime | None = None) -> float:
        if ticker not in CORE_TICKERS:
            return 0.0
        if not is_ticker_market_open(ticker, now):
            logger.debug("NewsPoller skip %s — market closed.", ticker)
            return self._cache.get(ticker, 0.0)
        last = self._last_call.get(ticker, 0.0)
        elapsed = time.monotonic() - last
        if last and elapsed < self.min_interval:
            logger.info(
                "NewsPoller skip %s — last call %.0fs ago (min interval %ds). Using cached score=%.3f",
                ticker,
                elapsed,
                self.min_interval,
                self._cache.get(ticker, 0.0),
            )
            return self._cache.get(ticker, 0.0)
        score = self._call_alpha_vantage(ticker)
        self._last_call[ticker] = time.monotonic()
        self._cache[ticker] = score
        return score

    def fetch_open_markets(self, now: datetime | None = None) -> dict[str, float]:
        scores = dict(self._cache)
        for ticker in CORE_TICKERS:
            if is_ticker_market_open(ticker, now):
                scores[ticker] = self.fetch(ticker, now)
        return scores

    def _call_alpha_vantage(self, ticker: str) -> float:
        score, headlines = latest_ticker_news(ticker, api_key=self.api_key)
        self._headlines[ticker] = headlines
        return score


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
def explain_action(
    ticker: str,
    action: float,
    qty: int,
    news_now: float,
    news_prev: float,
    corr_hint: float,
    *,
    current: float = 0.0,
    target_shares: float = 0.0,
) -> str:
    """Say what the policy wants and what the fill does to the book — not a fake headline."""
    name = TICKER_NAMES.get(ticker, ticker)
    if abs(action) < 0.05 and qty == 0:
        verb = "Hold"
    elif qty > 0:
        verb = f"Buy {qty} {name}"
    elif qty < 0:
        verb = f"Sell {abs(qty)} {name}"
    else:
        verb = f"Target {action:+.2f} {name}"

    reasons: list[str] = []
    tgt = float(action)
    if tgt >= 0.85:
        reasons.append(f"policy tgt {tgt:+.2f} (full long)")
    elif tgt <= -0.85:
        reasons.append(f"policy tgt {tgt:+.2f} (full short)")
    elif abs(tgt) < 0.05:
        reasons.append(f"policy tgt {tgt:+.2f} (flat)")
    elif tgt > 0:
        reasons.append(f"policy tgt {tgt:+.2f} (long)")
    else:
        reasons.append(f"policy tgt {tgt:+.2f} (short)")

    after = float(current) + float(qty)
    if qty != 0:
        reasons.append(f"book {current:.0f}→{after:.0f}")
        if float(target_shares) < -0.5 and after > 0.5:
            reasons.append("reduce-only (paper cannot short)")

    if corr_hint <= -0.4:
        reasons.append("HK×US corr deeply negative")
    elif corr_hint >= 0.6:
        reasons.append("HK×US corr high (names moving together)")
    delta = news_now - news_prev
    if delta <= -0.25:
        reasons.append(f"news {news_now:+.2f}, dropped {delta:.2f}")
    elif delta >= 0.25:
        reasons.append(f"news {news_now:+.2f}, jumped {delta:+.2f}")
    elif news_now <= -0.4:
        reasons.append(f"news {news_now:+.2f} (deeply negative)")
    elif news_now >= 0.4:
        reasons.append(f"news {news_now:+.2f} (strongly positive)")
    else:
        reasons.append(f"news {news_now:+.2f} (Δ{delta:+.2f}, no jump)")

    return f"Action: {verb}. {'; '.join(reasons)}."


# ---------------------------------------------------------------------------
# Futu OpenD (mirrors paper-trade-test.py / enquiry-test.py)
# ---------------------------------------------------------------------------
def _futu_available() -> bool:
    try:
        from futu import OpenQuoteContext  # noqa: F401
        return True
    except ImportError:
        return False


class FutuPaperBroker:
    """SIMULATE-only wrapper. Every OpenD call goes through the Futu RateLimiter."""

    def __init__(self, host: str = FUTU_HOST, port: int = FUTU_PORT, dry_run: bool = False) -> None:
        self.host = host
        self.port = port
        self.dry_run = dry_run or not _futu_available()
        self._quote = None
        self._trd_hk = None
        self._trd_us = None
        if self.dry_run:
            logger.warning("FutuPaperBroker running DRY-RUN (no OpenD orders).")

    def connect(self) -> None:
        if self.dry_run:
            return
        from futu import OpenQuoteContext, OpenSecTradeContext, TrdMarket

        futu_limiter.acquire()
        self._quote = OpenQuoteContext(host=self.host, port=self.port)
        futu_limiter.acquire()
        self._trd_hk = OpenSecTradeContext(filter_trdmarket=TrdMarket.HK, host=self.host, port=self.port)
        futu_limiter.acquire()
        self._trd_us = OpenSecTradeContext(filter_trdmarket=TrdMarket.US, host=self.host, port=self.port)
        logger.info("Connected to Futu OpenD at %s:%s", self.host, self.port)

    def close(self) -> None:
        for ctx in (self._quote, self._trd_hk, self._trd_us):
            if ctx is None:
                continue
            try:
                ctx.close()
            except Exception as exc:  # noqa: BLE001 — shutdown path
                logger.debug("Error closing Futu context: %s", exc)

    def _trade_ctx(self, ticker: str):
        return self._trd_hk if ticker_market(ticker) == "HK" else self._trd_us

    def snapshot_prices(self, tickers: list[str] | None = None) -> dict[str, float]:
        tickers = tickers or CORE_TICKERS
        if self.dry_run or self._quote is None:
            return {t: 0.0 for t in tickers}
        from futu import RET_OK

        futu_limiter.acquire()
        ret, data = self._quote.get_market_snapshot(tickers)
        if ret != RET_OK:
            logger.error("get_market_snapshot failed: %s", data)
            return {t: 0.0 for t in tickers}
        prices = {}
        for _, row in data.iterrows():
            prices[str(row["code"])] = float(row["last_price"])
        return prices

    def history_klines(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        tickers: list[str] | None = None,
    ):
        """10-min OHLCV for catch-up. Empty frame when dry-run / OpenD is down."""
        import pandas as pd

        if self.dry_run:
            logger.warning("Futu history kline fetch skipped (dry-run / no OpenD).")
            return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])
        return fetch_futu_history(
            tickers=tickers or CORE_TICKERS,
            start=start,
            end=end,
            quote_ctx=self._quote,
            limiter=futu_limiter,
            host=self.host,
            port=self.port,
        )

    def accinfo(self, ticker: str) -> dict[str, float]:
        if self.dry_run or self._trade_ctx(ticker) is None:
            return {}
        from futu import RET_OK, TrdEnv

        futu_limiter.acquire()
        ret, data = self._trade_ctx(ticker).accinfo_query(trd_env=TrdEnv.SIMULATE)
        if ret != RET_OK:
            logger.error("accinfo_query failed: %s", data)
            return {}
        row = data.iloc[0]
        return {
            "total_assets": float(row.get("total_assets", 0) or 0),
            "cash": float(row.get("cash", 0) or 0),
            "market_val": float(row.get("market_val", 0) or 0),
        }

    def positions(self, ticker: str) -> dict[str, float]:
        if self.dry_run or self._trade_ctx(ticker) is None:
            return {}
        from futu import RET_OK, TrdEnv

        futu_limiter.acquire()
        ret, data = self._trade_ctx(ticker).position_list_query(trd_env=TrdEnv.SIMULATE)
        if ret != RET_OK:
            logger.error("position_list_query failed: %s", data)
            return {}
        if data is None or len(data) == 0:
            return {}
        out = {}
        for _, row in data.iterrows():
            code = str(row.get("code", ""))
            qty = float(row.get("qty", 0) or 0)
            if code:
                out[code] = qty
        return out

    def place_order(self, ticker: str, qty: int, price: float, is_buy: bool) -> tuple[bool, str]:
        """Paper order. Same call shape as paper-trade-test.py. Returns (ok, order_id)."""
        if qty == 0:
            return False, ""
        side_name = "BUY" if is_buy else "SELL"
        px = round_to_tick(ticker, float(price))
        if px != float(price):
            logger.info("Rounded %s limit %s → %s (OpenD tick)", ticker, price, px)
        if self.dry_run or self._trade_ctx(ticker) is None:
            logger.info("[DRY-RUN] place_order %s %s qty=%s price=%s", side_name, ticker, qty, px)
            return True, "dry-run"
        from futu import RET_OK, TrdEnv, TrdSide

        futu_limiter.acquire()
        ctx = self._trade_ctx(ticker)
        ret, data = ctx.place_order(
            price=px,
            qty=int(qty),
            code=ticker,
            trd_side=TrdSide.BUY if is_buy else TrdSide.SELL,
            trd_env=TrdEnv.SIMULATE,
        )
        if ret == RET_OK:
            order_id = data["order_id"].iloc[0] if "order_id" in data.columns else data
            logger.info("SIMULATE order ok %s %s qty=%s price=%s order_id=%s", side_name, ticker, qty, px, order_id)
            return True, str(order_id)
        logger.error("place_order failed %s %s: %s", side_name, ticker, data)
        return False, ""

    def list_orders(self) -> list[dict[str, Any]] | None:
        """SIMULATE orders from both HK and US OpenD contexts (working + recent).

        Returns None when every OpenD query failed so callers can keep the last pending list.
        """
        from src.order_lifecycle import parse_order_row

        if self.dry_run:
            return []
        from futu import RET_OK, TrdEnv

        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        queried = 0
        failed = 0
        for ctx in (self._trd_hk, self._trd_us):
            if ctx is None:
                continue
            queried += 1
            try:
                futu_limiter.acquire()
                ret, data = ctx.order_list_query(trd_env=TrdEnv.SIMULATE, refresh_cache=True)
            except TypeError:
                futu_limiter.acquire()
                ret, data = ctx.order_list_query(trd_env=TrdEnv.SIMULATE)
            if ret != RET_OK:
                logger.warning("order_list_query failed: %s", data)
                failed += 1
                continue
            if data is None or len(data) == 0:
                continue
            for _, row in data.iterrows():
                parsed = parse_order_row(row)
                if parsed is None or parsed["order_id"] in seen:
                    continue
                seen.add(parsed["order_id"])
                out.append(parsed)
        if queried > 0 and failed == queried:
            return None
        return out

    def cancel_order(self, ticker: str, order_id: str) -> bool:
        if not order_id or self.dry_run or self._trade_ctx(ticker) is None:
            logger.info("[DRY-RUN] cancel_order %s %s", ticker, order_id)
            return bool(self.dry_run)
        from futu import RET_OK, ModifyOrderOp, TrdEnv

        futu_limiter.acquire()
        ret, data = self._trade_ctx(ticker).modify_order(
            ModifyOrderOp.CANCEL,
            str(order_id),
            0,
            0,
            trd_env=TrdEnv.SIMULATE,
        )
        if ret == RET_OK:
            logger.info("Cancelled SIMULATE order %s %s", ticker, order_id)
            return True
        logger.warning("cancel_order failed %s %s: %s", ticker, order_id, data)
        return False


def _sync_pending(state: BotState, broker: FutuPaperBroker) -> None:
    """Ask OpenD which limits are still working so the blotter can say PENDING."""
    if broker.dry_run:
        return
    from src.order_lifecycle import working_pending_rows

    try:
        live = broker.list_orders()
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenD order list failed (%s). Pending blotter unchanged.", exc)
        return
    if live is None:
        logger.warning("OpenD order list failed on every market. Pending blotter unchanged.")
        return
    from_open = working_pending_rows(live)
    have = {str(row.get("order_id") or "") for row in from_open if row.get("order_id")}
    live_ids = {str(row.get("order_id") or "") for row in live if row.get("order_id")}
    extras = [
        row
        for row in (state.pending_orders or [])
        if str(row.get("order_id") or "")
        and str(row.get("order_id") or "") not in have
        and str(row.get("order_id") or "") not in live_ids
    ]
    state.pending_orders = from_open + extras
    if state.pending_orders:
        logger.info(
            "OpenD pending: %s",
            ", ".join(
                f"{p.get('side')} {int(float(p.get('qty') or 0))} {p.get('ticker')} "
                f"@{float(p.get('price') or 0):.2f}"
                for p in state.pending_orders
            ),
        )


def round_to_lot(ticker: str, shares: float) -> int:
    lot = LOT_SIZES.get(ticker, 1)
    return int(shares // lot) * lot


# Match explain_action's "jumped / dropped sharply" threshold. Intra-bar
# rebalances are allowed only when news moves at least this much; otherwise
# 60-second polls would chase 1-minute price noise on the same 10-min bar.
NEWS_RETRADE_DELTA = 0.25


def _news_jumped(prev: dict[str, float], now: dict[str, float]) -> bool:
    for ticker in CORE_TICKERS:
        if abs(float(now.get(ticker, 0.0)) - float(prev.get(ticker, 0.0))) >= NEWS_RETRADE_DELTA:
            return True
    return False


def reconcile_with_futu(state: BotState, broker: FutuPaperBroker) -> BotState:
    """Log drift vs OpenD; state.pkl remains source of truth for last action / P&L."""
    hk_acc = broker.accinfo("HK.00700")
    us_acc = broker.accinfo("US.COST")
    hk_pos = broker.positions("HK.00700")
    us_pos = broker.positions("US.COST")
    live_pos = {**hk_pos, **us_pos}
    for ticker in CORE_TICKERS:
        live = float(live_pos.get(ticker, 0.0))
        saved = float(state.holdings.get(ticker, 0.0))
        if abs(live - saved) > 1e-6 and (live or saved):
            logger.warning(
                "Position drift %s: state.pkl=%.4f Futu=%.4f — keeping state.pkl to avoid double-buying.",
                ticker,
                saved,
                live,
            )
    if hk_acc:
        logger.info("Futu HK SIMULATE accinfo: %s", hk_acc)
    if us_acc:
        logger.info("Futu US SIMULATE accinfo: %s", us_acc)
    return state


def resolve_inference_model_path(explicit: Path | None = None) -> Path:
    """Paper-trading brain: ``models/news_gpu_v2/best_model.zip`` (Window 113 / Calmar 2.05)."""
    if explicit is not None:
        return Path(explicit)
    if INFERENCE_MODEL_PATH.exists():
        return INFERENCE_MODEL_PATH
    golden = NEWS_GPU_V2_MODELS_DIR / "checkpoint_2026-08-12.zip"
    if golden.exists():
        logger.warning(
            "%s is missing; falling back to %s (trading golden, Calmar 2.05).",
            INFERENCE_MODEL_PATH,
            golden,
        )
        return golden
    if BEST_MODEL_PATH.exists():
        logger.warning(
            "news_gpu_v2 best_model.zip is missing; falling back to legacy %s",
            BEST_MODEL_PATH,
        )
        return BEST_MODEL_PATH
    return INFERENCE_MODEL_PATH


def log_checkpoint_banner(model_path: Path) -> None:
    """First thing the operator should see: which zip the live policy is."""
    resolved = model_path.resolve()
    exists = model_path.exists()
    size = ""
    if exists:
        size = f"{model_path.stat().st_size / (1024 * 1024):.2f} MB"
    logger.info("============================================================")
    logger.info("AirAire inference — model checkpoint")
    logger.info("  path   : %s", resolved)
    logger.info("  exists : %s%s", exists, f"  ({size})" if size else "")
    logger.info("  role   : paper trading (models/news_gpu_v2/best_model.zip)")
    logger.info("============================================================")


def load_policy(model_path: Path | None = None):
    model_path = resolve_inference_model_path(model_path)
    if not model_path.exists():
        logger.warning("No trained model at %s — policy will HOLD (action=0).", model_path)
        return None
    try:
        from stable_baselines3 import PPO

        model = PPO.load(str(model_path))
        logger.info("Loaded PPO policy from %s", model_path.resolve())
        return model
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load model %s: %s", model_path, exc)
        return None


def _hk_now(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(tz=HK_TZ)
    if now.tzinfo is None:
        return now.replace(tzinfo=HK_TZ)
    return now.astimezone(HK_TZ)


def _maybe_push_dashboard(
    state: BotState,
    news: NewsPoller,
    enabled: bool,
    kind: str = "live",
    prices: dict[str, float] | None = None,
) -> None:
    if not enabled:
        return
    headlines = news.headlines_by_ticker()
    try:
        try:
            push_live_snapshot(state, headlines=headlines, kind=kind, prices=prices)
        except TypeError:
            # Older dashboard_push.py on the VM has no `kind=` / `prices=` yet.
            try:
                push_live_snapshot(state, headlines=headlines, kind=kind)
            except TypeError:
                push_live_snapshot(state, headlines=headlines)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Dashboard push skipped (%s). Trading continues.", exc)


def catch_up_env(
    env: TradingEnv,
    state: BotState,
    broker: FutuPaperBroker,
    panel,
    *,
    now: datetime | None = None,
    persist_panel: bool = True,
    persist_state: bool = True,
) -> tuple[TradingEnv, BotState, Any]:
    """Advance the env to the current bar WITHOUT placing any orders.

    Startup example: operator logs in at 12:45 after missing the morning
    session. We pull the missing 10-min bars from Futu, rebuild the price
    cubes, jump ``_bar_index`` to the latest completed bar, restore the
    ``state.pkl`` book, and persist ``updated_at`` / ``last_bar_datetime``.
    """
    import pandas as pd

    # Panel clocks (CSV / parquet) are tz-naive. _hk_now() is Asia/Hong_Kong-aware;
    # strip tz after conversion so fetch-start and Futu date bounds stay comparable.
    now_hk = _hk_now(now).replace(tzinfo=None)
    before_dt = env._current_dt() if len(getattr(env, "datetimes", [])) else None
    live = pd.DataFrame()
    try:
        start = default_futu_fetch_start(panel, now=now_hk)
        live = broker.history_klines(start=start.to_pydatetime(), end=now_hk)
    except Exception as exc:  # noqa: BLE001 — catch-up must never block a start
        logger.warning("Futu catch-up fetch failed (%s). Seeking on the existing panel only.", exc)

    rebuilt = False
    n_before = 0 if panel is None or getattr(panel, "empty", True) else len(panel)
    panel = overlay_live_ohlcv(panel, live, now=now_hk)
    n_after = 0 if panel is None or panel.empty else len(panel)
    if live is not None and not live.empty:
        logger.info(
            "Catch-up merged %d live Futu rows into the panel (%d → %d).",
            len(live),
            n_before,
            n_after,
        )
    if (live is not None and not live.empty) or n_after != n_before:
        env = TradingEnv(df=panel if panel is not None and not panel.empty else None, news_scores=state.news_scores)
        rebuilt = True
        if persist_panel and panel is not None and not panel.empty:
            persist_enhanced_panel(panel)

    env.reset()
    caught_dt = env.seek_to_datetime(panel_seek_now(now), completed_bars=True)
    env.restore_portfolio(state.cash, state.holdings)
    env.set_news_scores(state.news_scores)

    live_px = broker.snapshot_prices()
    if any(float(v or 0.0) > 0 for v in live_px.values()):
        state.equity = float(state.cash) + sum(
            float(state.holdings[t]) * float(live_px.get(t) or 0.0) for t in CORE_TICKERS if str(t).startswith("HK.")
        )
    else:
        state.equity = float(env._last_equity)

    state.last_bar_datetime = str(caught_dt)
    logger.info(
        "State catch-up complete (no orders). env_before=%s env_now=%s bar=%d/%d rebuilt=%s "
        "hk_cash=%.2f equity=%.2f seek=%s clock=%s",
        before_dt,
        caught_dt,
        env._bar_index,
        max(len(env.datetimes) - 1, 0),
        rebuilt,
        state.cash,
        state.equity,
        panel_seek_now(now),
        live_session_clock(now),
    )
    if persist_state:
        save_state(state)
    return env, state, panel


def predict_action(model, obs: np.ndarray) -> np.ndarray:
    if model is None:
        return np.zeros(len(CORE_TICKERS), dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    return np.asarray(action, dtype=np.float32).reshape(-1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
_shutdown = False


def _handle_stop(signum, _frame) -> None:  # noqa: ANN001
    global _shutdown
    logger.info("Received signal %s — will persist state and exit.", signum)
    _shutdown = True


def _mtm_prefix(holdings: dict[str, float], prices: dict[str, float], prefix: str) -> float:
    return sum(
        float(holdings.get(ticker, 0.0) or 0.0) * float(prices.get(ticker) or 0.0)
        for ticker in CORE_TICKERS
        if str(ticker).startswith(prefix)
    )


def _apply_trade_cash(state: BotState, ticker: str, delta: float, px: float) -> None:
    flow = -float(delta) * float(px)
    if str(ticker).startswith("US."):
        state.us_cash = float(getattr(state, "us_cash", US_INITIAL_CASH)) + flow
        return
    state.cash = float(state.cash) + flow


def run_loop(
    once: bool = False,
    dry_run: bool = False,
    poll_seconds: int = 60,
    model_path: Path | None = None,
    skip_catch_up: bool = False,
    predict_now: bool = False,
) -> None:
    # Log the trading brain before anything else so a wrong zip is obvious.
    resolved_model = resolve_inference_model_path(model_path)
    log_checkpoint_banner(resolved_model)
    if predict_now:
        once = True
        logger.info(
            "PREDICT NOW — one cycle, live quotes + news, no SIMULATE orders, state.pkl not written. "
            "Safe to run while run_trader.bat is already up."
        )
    persist_state = not predict_now and not dry_run
    push_dashboard = persist_state

    # §8: load state FIRST, before any order
    state = load_state()
    news = NewsPoller()
    # --dry-run alone skips OpenD. --predict-now still connects for live quotes;
    # orders are never sent (see the predict_now branch below).
    broker = FutuPaperBroker(dry_run=dry_run and not predict_now)
    panel = None
    try:
        if ENHANCED_PARQUET.exists():
            import pandas as pd

            panel = pd.read_parquet(ENHANCED_PARQUET)
            logger.info("Loaded enhanced panel with news_score for the local env.")
        else:
            panel = load_processed()
        env = TradingEnv(
            df=panel if panel is not None and not panel.empty else None,
            news_scores=state.news_scores,
        )
    except Exception as exc:  # noqa: BLE001 — parquet is optional in Phase 1
        logger.warning("Could not load unified/enhanced parquet (%s); TradingEnv will use synthetic bars.", exc)
        env = TradingEnv(news_scores=state.news_scores)
        panel = None
    model = load_policy(resolved_model)

    try:
        broker.connect()
        state = reconcile_with_futu(state, broker)
        _sync_pending(state, broker)
        if skip_catch_up:
            env.reset()
            env.seek_to_datetime(panel_seek_now(), completed_bars=True)
            env.restore_portfolio(state.cash, state.holdings)
            env.set_news_scores(state.news_scores)
            logger.warning("Catch-up skipped (--skip-catch-up). Env sought to now without Futu history.")
        else:
            env, state, panel = catch_up_env(
                env,
                state,
                broker,
                panel,
                persist_panel=persist_state,
                persist_state=persist_state,
            )

        while not _shutdown:
            if not any_core_market_open():
                if predict_now:
                    logger.info(
                        "Both HK and US cash sessions closed. Predict-now still scores the last completed bar (no orders)."
                    )
                else:
                    _sync_pending(state, broker)
                    _maybe_push_dashboard(state, news, push_dashboard, kind="heartbeat")
                    wait = min(seconds_until_next_open(), 300)
                    logger.info("Both HK and US cash sessions closed. HOLD. Sleeping %ss.", wait)
                    if once:
                        break
                    time.sleep(wait)
                    continue

            # Keep the observation on the live session's completed 10-min bar
            # (US Eastern while HK is shut — not HKT, which sticks on HK 16:00).
            if not skip_catch_up:
                try:
                    last = env._current_dt()
                    seek_now = panel_seek_now()
                    if need_panel_refresh(last, seek_now):
                        env, state, panel = catch_up_env(
                            env,
                            state,
                            broker,
                            panel,
                            persist_panel=False,
                            persist_state=persist_state,
                        )
                    else:
                        env.seek_to_datetime(seek_now, completed_bars=True)
                        env.restore_portfolio(state.cash, state.holdings)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("In-session catch-up failed (%s). Using current env bar.", exc)

            prices = broker.snapshot_prices()
            now_loop = _hk_now()
            ready_keys = ready_cash_sessions(now_loop)
            any_session_ready = bool(ready_keys)
            session_today = {key: session_date_iso(key, now_loop) for key in ready_keys}
            first_session = any(state.last_session_ready.get(key) != session_today[key] for key in ready_keys)

            news_prev = dict(state.news_scores)
            news_now = news.fetch_open_markets()
            env.set_news_scores(news_now)
            # Freeze book scores until a session's first 10-min bar has closed so
            # a 09:30 headline can still trip NEWS_RETRADE_DELTA at 09:40.
            if any_session_ready:
                news_jump = _news_jumped(news_prev, news_now)
                state.news_scores = news_now
            else:
                news_jump = False

            obs = env._get_obs()
            raw_action = predict_action(model, obs)
            long_term = env._long_term_features()
            # last 6 entries are HK×US correlations; use the mean as a reason hint
            corr_hint = float(np.mean(long_term[-6:])) if len(long_term) >= 6 else 0.0

            clock = live_session_clock(now_loop)
            clock_ticker = "US.COST" if clock == "US" else "HK.00700"
            bar_id = session_bar_id(env._current_dt(), now_loop)
            new_bar = bar_id != state.last_order_bar
            bar_ready = is_kline_complete(env._current_dt(), ticker=clock_ticker, now=now_loop)
            allow_orders = (
                (not predict_now)
                and bar_ready
                and any_session_ready
                and (new_bar or news_jump or first_session)
            )
            if predict_now:
                logger.info("Predict-now bar=%s (preview only).", bar_id)
            elif not any_session_ready:
                logger.info(
                    "Cash session in first 10 minutes — wait for a completed open bar "
                    "(HK 09:40 / 13:10, US 09:40 ET)."
                )
            elif not bar_ready:
                logger.info(
                    "Forming 10-min bar (%s) — wait for the close before orders (matches training).",
                    bar_id,
                )
            elif not allow_orders:
                logger.info(
                    "Same 10-min bar (%s) and news unchanged — skip orders this 60s cycle (no 1-min price chase).",
                    bar_id,
                )

            traded = False
            order_failed = False
            for i, ticker in enumerate(CORE_TICKERS):
                current = float(state.holdings.get(ticker, 0.0))
                # Closed market: KEEP the position. Gating the action to 0 would
                # flatten US names during the HK session (and HK names after 16:00).
                if not is_ticker_market_open(ticker):
                    logger.info(
                        "%s market closed — keep holdings=%.4f, no order.",
                        ticker,
                        current,
                    )
                    continue
                if not is_cash_open_bar_complete(ticker, now_loop):
                    logger.info(
                        "%s first 10-min cash bar still open — keep holdings=%.4f, no order.",
                        ticker,
                        current,
                    )
                    continue
                action_i = float(raw_action[i])
                px = float(prices.get(ticker) or 0.0)
                if px <= 0 and not dry_run:
                    logger.info("No live price for %s — HOLD.", ticker)
                    state.last_action[ticker] = 0.0
                    continue
                px = px or 1.0
                px = round_to_tick(ticker, px)
                if str(ticker).startswith("US."):
                    equity = max(float(getattr(state, "us_cash", US_INITIAL_CASH)) + _mtm_prefix(state.holdings, prices, "US."), 1.0)
                else:
                    equity = max(float(state.cash) + _mtm_prefix(state.holdings, prices, "HK."), 1.0)
                target_shares = (action_i * equity) / px
                delta = round_to_lot(ticker, target_shares - current)
                reason = explain_action(
                    ticker,
                    action_i,
                    delta,
                    news_now.get(ticker, 0.0),
                    news_prev.get(ticker, 0.0),
                    corr_hint,
                    current=current,
                    target_shares=target_shares,
                )
                prefix = "[predict-now] " if predict_now else ("" if allow_orders else "[preview, no order] ")
                logger.info("%s%s", prefix, reason)
                state.last_reason = reason
                state.last_action[ticker] = action_i
                if predict_now or not allow_orders or delta == 0:
                    continue
                is_buy = delta > 0
                ok, order_id = broker.place_order(ticker, abs(delta), px, is_buy=is_buy)
                if dry_run:
                    logger.info("[DRY-RUN] would %s %s qty=%s — book unchanged.", "BUY" if is_buy else "SELL", ticker, abs(delta))
                    continue
                if ok:
                    state.holdings[ticker] = current + delta
                    _apply_trade_cash(state, ticker, delta, px)
                    traded = True
                    append_fill(
                        {
                            "time": datetime.now(tz=HK_TZ).isoformat(),
                            "ticker": ticker,
                            "side": "BUY" if is_buy else "SELL",
                            "qty": int(abs(delta)),
                            "price": float(px),
                            "reason": reason,
                            "order_id": order_id,
                        }
                    )
                    name = TICKER_NAMES.get(ticker, ticker)
                    state.pending_orders = [
                        row
                        for row in (state.pending_orders or [])
                        if str(row.get("order_id") or "") != str(order_id)
                    ]
                    state.pending_orders.append(
                        {
                            "order_id": str(order_id),
                            "ticker": ticker,
                            "side": "BUY" if is_buy else "SELL",
                            "qty": int(abs(delta)),
                            "price": float(px),
                            "kind": "working",
                            "status": "SUBMITTED",
                            "time": datetime.now(tz=HK_TZ).isoformat(),
                            "reason": (
                                f"PENDING {'BUY' if is_buy else 'SELL'} {int(abs(delta))} {name} "
                                f"@ {px:.2f} (not a fill)"
                            ),
                        }
                    )
                    send_telegram_alert(reason)
                else:
                    order_failed = True

            if allow_orders and not order_failed:
                state.last_order_bar = bar_id
                state.last_session_ready.update(session_today)
            state.equity = float(state.cash) + _mtm_prefix(state.holdings, prices, "HK.")
            state.last_bar_datetime = bar_id
            env.restore_portfolio(state.cash, state.holdings)
            _sync_pending(state, broker)
            if persist_state and (traded or allow_orders or state.pending_orders):
                save_state(state)
            else:
                logger.info(
                    "No fills this cycle. clock=%s bar=%s HK equity≈%.2f cash=%.2f | US equity≈%.2f cash=%.2f",
                    clock,
                    bar_id,
                    state.equity,
                    state.cash,
                    float(getattr(state, "us_cash", US_INITIAL_CASH)) + _mtm_prefix(state.holdings, prices, "US."),
                    getattr(state, "us_cash", US_INITIAL_CASH),
                )
            _maybe_push_dashboard(state, news, push_dashboard, prices=prices)

            if predict_now:
                logger.info("============================================================")
                logger.info("PREDICT NOW summary  bar=%s  equity≈%.2f  cash=%.2f", state.last_bar_datetime, state.equity, state.cash)
                for ticker in CORE_TICKERS:
                    logger.info(
                        "  %s  action=%+.3f  holdings=%.4f  news=%.3f  %s",
                        ticker,
                        float(state.last_action.get(ticker, 0.0)),
                        float(state.holdings.get(ticker, 0.0)),
                        float(news_now.get(ticker, 0.0)),
                        "OPEN" if is_ticker_market_open(ticker) else "CLOSED",
                    )
                logger.info("============================================================")

            if once:
                if persist_state:
                    save_state(state)
                break
            time.sleep(max(poll_seconds, 5))
    finally:
        if persist_state:
            save_state(state)
            logger.info("Shutdown complete. state.pkl is current.")
        else:
            logger.info("Predict-now complete. state.pkl was not written.")
        broker.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AirAire paper trader")
    p.add_argument("--once", action="store_true", help="Run a single inference cycle then exit (writes state.pkl; do not use while the trader is already running).")
    p.add_argument("--dry-run", action="store_true", help="Test only: skip OpenD and do not send Futu orders. Do not use on the live trader.")
    p.add_argument(
        "--predict-now",
        action="store_true",
        help="One live predict cycle: quotes + news, no orders, do not write state.pkl. Safe while the trader is running.",
    )
    p.add_argument("--poll-seconds", type=int, default=60, help="Seconds between cycles while a market is open.")
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="PPO zip to load (default: models/news_gpu_v2/best_model.zip).",
    )
    p.add_argument(
        "--skip-catch-up",
        action="store_true",
        help="Do not pull missing Futu bars at startup (still seeks to now on the existing panel).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _handle_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_stop)
    run_loop(
        once=args.once,
        dry_run=args.dry_run,
        poll_seconds=args.poll_seconds,
        model_path=args.model,
        skip_catch_up=args.skip_catch_up,
        predict_now=args.predict_now,
    )
    sys.exit(0)
