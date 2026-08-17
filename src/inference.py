"""Local paper trader: load state.pkl first, poll news ethically, execute via Futu SIMULATE.

Resume contract (PLAN.md §8):
    1. Load ``state.pkl`` *before* any order.
    2. After every trade, persist holdings / cash / last action.
    3. On shutdown, flush state again.
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
import requests
from dotenv import load_dotenv

from src.data_loader import load_processed
from src.trading_env import TradingEnv
from src.utils import (
    AV_TICKERS,
    BEST_MODEL_PATH,
    CORE_TICKERS,
    FUTU_HOST,
    FUTU_PORT,
    INITIAL_CASH,
    LOT_SIZES,
    NEWS_MIN_INTERVAL_SECONDS,
    STATE_PKL,
    TICKER_NAMES,
    RateLimiter,
    any_core_market_open,
    is_ticker_market_open,
    seconds_until_next_open,
    send_telegram_alert,
    setup_logging,
    ticker_market,
)

load_dotenv()
logger = setup_logging("airaire.inference")

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
futu_limiter = RateLimiter()


# ---------------------------------------------------------------------------
# Persistent bot state
# ---------------------------------------------------------------------------
@dataclass
class BotState:
    holdings: dict[str, float] = field(default_factory=lambda: {t: 0.0 for t in CORE_TICKERS})
    cash: float = INITIAL_CASH
    last_action: dict[str, float] = field(default_factory=lambda: {t: 0.0 for t in CORE_TICKERS})
    last_reason: str = "cold start"
    realized_pnl: float = 0.0
    equity: float = INITIAL_CASH
    news_scores: dict[str, float] = field(default_factory=lambda: {t: 0.0 for t in CORE_TICKERS})
    updated_at: str = ""

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
        return cls(
            holdings=holdings,
            cash=float(raw.get("cash", INITIAL_CASH)),
            last_action=last_action,
            last_reason=str(raw.get("last_reason", "loaded from disk")),
            realized_pnl=float(raw.get("realized_pnl", 0.0)),
            equity=float(raw.get("equity", raw.get("cash", INITIAL_CASH))),
            news_scores=news_scores,
            updated_at=str(raw.get("updated_at", "")),
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
        state = raw
    elif isinstance(raw, dict):
        state = BotState.from_dict(raw)
    else:
        logger.warning("Unrecognized state.pkl payload (%s); using defaults.", type(raw))
        state = BotState()
    logger.info(
        "Loaded state.pkl updated_at=%s cash=%.2f holdings=%s last_action=%s pnl=%.2f",
        state.updated_at,
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
# Alpha Vantage news — ethical 5-minute cap per ticker, market hours only
# ---------------------------------------------------------------------------
class NewsPoller:
    """Academic-tier news fetcher. Never hits the same ticker more than once / 5 minutes."""

    def __init__(self, api_key: str | None = None, min_interval: int = NEWS_MIN_INTERVAL_SECONDS) -> None:
        self.api_key = (api_key or os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY") or "").strip()
        self.min_interval = min_interval
        self._last_call: dict[str, float] = {}
        self._cache: dict[str, float] = {t: 0.0 for t in CORE_TICKERS}

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
        if not self.api_key:
            logger.warning("NewsPoller: ALPHAVANTAGE_API_KEY unset — sentiment for %s stays at cache.", ticker)
            return self._cache.get(ticker, 0.0)
        av_symbol = AV_TICKERS.get(ticker, ticker)
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": av_symbol,
            "limit": 10,
            "apikey": self.api_key,
        }
        try:
            resp = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            logger.warning("NewsPoller HTTP error for %s: %s", ticker, exc)
            return self._cache.get(ticker, 0.0)
        if not isinstance(payload, dict) or "feed" not in payload:
            logger.warning("NewsPoller unexpected payload for %s: %s", ticker, list(payload)[:6] if isinstance(payload, dict) else type(payload))
            return self._cache.get(ticker, 0.0)
        scores = []
        for item in payload.get("feed") or []:
            for ts in item.get("ticker_sentiment") or []:
                if str(ts.get("ticker", "")).upper() == av_symbol.upper():
                    try:
                        scores.append(float(ts.get("ticker_sentiment_score", 0.0)))
                    except (TypeError, ValueError):
                        continue
            if not scores:
                try:
                    scores.append(float(item.get("overall_sentiment_score", 0.0)))
                except (TypeError, ValueError):
                    continue
        avg = float(np.clip(np.mean(scores) if scores else 0.0, -1.0, 1.0))
        logger.info("NewsPoller %s (%s) score=%.3f n_headlines=%d", ticker, av_symbol, avg, len(scores))
        return avg


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
) -> str:
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
    if corr_hint <= -0.4:
        reasons.append("High negative correlation detected")
    elif corr_hint >= 0.6:
        reasons.append("High positive correlation (pairs moving together)")
    delta = news_now - news_prev
    if delta <= -0.25:
        reasons.append("News Sentiment dropped sharply")
    elif delta >= 0.25:
        reasons.append("News Sentiment jumped")
    elif news_now <= -0.4:
        reasons.append("News Sentiment is deeply negative")
    elif news_now >= 0.4:
        reasons.append("News Sentiment is strongly positive")
    if abs(action) < 0.05:
        reasons.append("Policy near zero — stay flat")
    if not reasons:
        reasons.append("Policy network output (no strong news/corr overlay)")
    return f"Action: {verb}. Reason: {' + '.join(reasons)}."


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

    def place_order(self, ticker: str, qty: int, price: float, is_buy: bool) -> bool:
        """Paper order. Same call shape as paper-trade-test.py."""
        if qty == 0:
            return False
        side_name = "BUY" if is_buy else "SELL"
        if self.dry_run or self._trade_ctx(ticker) is None:
            logger.info("[DRY-RUN] place_order %s %s qty=%s price=%s", side_name, ticker, qty, price)
            return True
        from futu import RET_OK, TrdEnv, TrdSide

        futu_limiter.acquire()
        ret, data = self._trade_ctx(ticker).place_order(
            price=float(price),
            qty=int(qty),
            code=ticker,
            trd_side=TrdSide.BUY if is_buy else TrdSide.SELL,
            trd_env=TrdEnv.SIMULATE,
        )
        if ret == RET_OK:
            order_id = data["order_id"].iloc[0] if "order_id" in data.columns else data
            logger.info("SIMULATE order ok %s %s qty=%s price=%s order_id=%s", side_name, ticker, qty, price, order_id)
            return True
        logger.error("place_order failed %s %s: %s", side_name, ticker, data)
        return False


def round_to_lot(ticker: str, shares: float) -> int:
    lot = LOT_SIZES.get(ticker, 1)
    return int(shares // lot) * lot


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


def load_policy(model_path: Path = BEST_MODEL_PATH):
    if not model_path.exists():
        logger.warning("No trained model at %s — policy will HOLD (action=0).", model_path)
        return None
    try:
        from stable_baselines3 import PPO

        model = PPO.load(str(model_path))
        logger.info("Loaded PPO policy from %s", model_path)
        return model
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load model %s: %s", model_path, exc)
        return None


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


def run_loop(once: bool = False, dry_run: bool = False, poll_seconds: int = 60) -> None:
    # §8: load state FIRST, before any order
    state = load_state()
    news = NewsPoller()
    broker = FutuPaperBroker(dry_run=dry_run)
    try:
        panel = load_processed()
        env = TradingEnv(df=panel if panel is not None and not panel.empty else None)
    except Exception as exc:  # noqa: BLE001 — parquet is optional in Phase 1
        logger.warning("Could not load unified parquet (%s); TradingEnv will use synthetic bars.", exc)
        env = TradingEnv()
    model = load_policy()

    try:
        broker.connect()
        state = reconcile_with_futu(state, broker)
        env._cash = state.cash
        env._holdings = np.asarray([state.holdings[t] for t in CORE_TICKERS], dtype=np.float64)
        env.set_news_scores(state.news_scores)
        env.reset()
        env._cash = state.cash
        env._holdings = np.asarray([state.holdings[t] for t in CORE_TICKERS], dtype=np.float64)

        while not _shutdown:
            if not any_core_market_open():
                wait = min(seconds_until_next_open(), 300)
                logger.info("Both HK and US cash sessions closed. HOLD. Sleeping %ss.", wait)
                if once:
                    break
                time.sleep(wait)
                continue

            prices = broker.snapshot_prices()
            news_prev = dict(state.news_scores)
            news_now = news.fetch_open_markets()
            env.set_news_scores(news_now)
            state.news_scores = news_now

            obs = env._get_obs()
            raw_action = predict_action(model, obs)
            # Do not trade names whose market is shut (e.g. US names during HK morning).
            gated = raw_action.copy()
            for i, ticker in enumerate(CORE_TICKERS):
                if not is_ticker_market_open(ticker):
                    gated[i] = 0.0

            long_term = env._long_term_features()
            # last 6 entries are HK×US correlations; use the mean as a reason hint
            corr_hint = float(np.mean(long_term[-6:])) if len(long_term) >= 6 else 0.0

            traded = False
            for i, ticker in enumerate(CORE_TICKERS):
                action_i = float(gated[i])
                px = float(prices.get(ticker) or 0.0)
                if px <= 0 and not dry_run:
                    logger.info("No live price for %s — HOLD.", ticker)
                    state.last_action[ticker] = 0.0
                    continue
                px = px or 1.0
                equity = max(state.cash + sum(state.holdings[t] * float(prices.get(t) or 0.0) for t in CORE_TICKERS), 1.0)
                target_shares = (action_i * equity) / px
                current = float(state.holdings.get(ticker, 0.0))
                delta = round_to_lot(ticker, target_shares - current)
                reason = explain_action(ticker, action_i, delta, news_now.get(ticker, 0.0), news_prev.get(ticker, 0.0), corr_hint)
                logger.info(reason)
                state.last_reason = reason
                state.last_action[ticker] = action_i
                if delta == 0:
                    continue
                is_buy = delta > 0
                ok = broker.place_order(ticker, abs(delta), px, is_buy=is_buy)
                if ok:
                    state.holdings[ticker] = current + delta
                    state.cash -= delta * px
                    traded = True
                    send_telegram_alert(reason)

            state.equity = state.cash + sum(state.holdings[t] * float(prices.get(t) or 0.0) for t in CORE_TICKERS)
            if traded:
                save_state(state)
            else:
                logger.info("No fills this cycle. Equity≈%.2f cash=%.2f", state.equity, state.cash)

            if once:
                # Persist even a hold cycle so a restart sees the latest news/action snapshot.
                save_state(state)
                break
            time.sleep(max(poll_seconds, 5))
    finally:
        save_state(state)
        broker.close()
        logger.info("Shutdown complete. state.pkl is current.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AirAire paper trader")
    p.add_argument("--once", action="store_true", help="Run a single inference cycle then exit.")
    p.add_argument("--dry-run", action="store_true", help="Skip Futu OpenD orders.")
    p.add_argument("--poll-seconds", type=int, default=60, help="Seconds between cycles while a market is open.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _handle_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_stop)
    run_loop(once=args.once, dry_run=args.dry_run, poll_seconds=args.poll_seconds)
    sys.exit(0)
