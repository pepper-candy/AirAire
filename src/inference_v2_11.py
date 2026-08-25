"""V2.11 paper inference — 782-dim hybrid, book on fill, isolated from live V2.

HK actions are clipped to ``[0, 1]`` (Futu has no HK short). US stay ``[-1, 1]``.
``decide_order`` skips underwater round-trips. Stale working limits are cancelled
after one completed bar / 12 minutes, then the book is read from OpenD.
Orders are **not** booked on submit. Min notional / min weight-step skip COST/KO flicker.

Does not write ``state.pkl``, ``models/news_gpu_v2/``, or ``enhanced_data.parquet``.
Own book: ``state_v2_11.pkl``. Do not run while ``run_trader.bat`` (V2) is live
on the same SIMULATE account.

    python -m src.inference_v2_11 --predict-now
    python -m src.inference_v2_11 --poll-seconds 60
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.dashboard_push import append_fill
from src.inference import (
    BotState,
    FutuPaperBroker,
    NewsPoller,
    _hk_now,
    _maybe_push_dashboard,
    _news_jumped,
    catch_up_env,
    load_policy,
    load_state,
    predict_action,
    round_to_lot,
    save_state,
)
from src.inference_v3 import (
    _apply_cash,
    _cap_ids,
    _hk_equity,
    _mark_books,
    _us_equity,
)
from src.order_lifecycle import (
    classify_place_error,
    decide_order,
    skip_tiny_rebalance,
    stale_working_orders,
)
from src.trading_env_v2_11 import TradingEnv, observation_dim
from src.utils import (
    CORE_TICKERS,
    ENHANCED_PARQUET,
    HK_TZ,
    INITIAL_CASH,
    NEWS_GPU_V2_11_MODELS_DIR,
    TICKER_NAMES,
    US_INITIAL_CASH,
    any_core_market_open,
    attach_session_file_log,
    is_cash_open_bar_complete,
    is_kline_complete,
    is_ticker_market_open,
    live_session_clock,
    need_panel_refresh,
    panel_seek_now,
    ready_cash_sessions,
    round_to_tick,
    seconds_until_next_open,
    send_telegram_alert,
    session_bar_id,
    session_date_iso,
    setup_logging,
)
from src.v2_11 import (
    STATE_V2_11_PKL,
    V2_11_OBS_DIM,
    clip_hybrid_action,
    install_seed_into_v2_11,
    log_seed_banner,
    refuse_wrong_inference_zip,
    resolve_v2_paper_seed_zip,
)

logger = setup_logging("airaire.inference_v2_11")


def _explain(
    ticker: str,
    action: float,
    qty: int,
    news_now: float,
    news_prev: float,
    corr_hint: float,
    *,
    current: float = 0.0,
    target_shares: float = 0.0,
    raw_action: float | None = None,
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
    tgt = float(action)
    if raw_action is not None and abs(float(raw_action) - tgt) > 1e-9:
        reasons.append(f"raw {float(raw_action):+.2f} → clipped {tgt:+.2f} (HK long-only)")
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
        reasons.append(f"book {current:.0f}→{after:.0f} (on fill, not submit)")
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


def resolve_v2_11_model(explicit: Path | None = None) -> Path:
    if explicit is not None:
        refuse_wrong_inference_zip(explicit)
        return Path(explicit)
    local = NEWS_GPU_V2_11_MODELS_DIR / "best_model.zip"
    if local.exists():
        return local
    return resolve_v2_paper_seed_zip()


def load_v2_11_state(*, predict_now: bool) -> BotState:
    if STATE_V2_11_PKL.exists():
        return load_state(STATE_V2_11_PKL)
    from src.utils import STATE_PKL

    if predict_now and STATE_PKL.exists():
        state = load_state(STATE_PKL)
        state.last_order_bar = ""
        state.last_reason = "predict-now scoring against V2 leftover (not written)"
        logger.info(
            "No %s — predict-now scores V2 leftover from %s (cash=%.2f). File not written.",
            STATE_V2_11_PKL.name,
            STATE_PKL.name,
            state.cash,
        )
        return state
    logger.info("No %s — starting a flat V2.11 book (cash=%.2f).", STATE_V2_11_PKL.name, INITIAL_CASH)
    state = BotState()
    state.updated_at = datetime.now(tz=HK_TZ).isoformat()
    return state


def _clamp_hk_delta(ticker: str, delta: int, current: float) -> int:
    """HK cannot short. US SIMULATE can."""
    if str(ticker).startswith("US."):
        return int(delta)
    if delta >= 0:
        return int(delta)
    sellable = round_to_lot(ticker, current)
    if sellable <= 0:
        return 0
    return max(int(delta), -int(sellable))


def _remember_placed(state: BotState, order_id: str) -> None:
    oid = str(order_id or "")
    if not oid:
        return
    if oid not in state.placed_order_ids:
        state.placed_order_ids.append(oid)
    state.placed_order_ids = _cap_ids(state.placed_order_ids)


def _mark_settled(state: BotState, order_id: str) -> None:
    oid = str(order_id or "")
    if not oid:
        return
    if oid not in state.settled_order_ids:
        state.settled_order_ids.append(oid)
    state.settled_order_ids = _cap_ids(state.settled_order_ids)


def _apply_futu_fill(state: BotState, order: dict, *, persist_fills: bool, reason: str) -> None:
    ticker = str(order.get("ticker") or "")
    if ticker not in CORE_TICKERS:
        return
    qty = int(float(order.get("dealt_qty") or 0) or float(order.get("qty") or 0))
    px = float(order.get("dealt_avg_price") or order.get("price") or 0.0)
    if qty <= 0 or px <= 0:
        return
    side = str(order.get("side") or "BUY").upper()
    if side == "BUY":
        state.holdings[ticker] = float(state.holdings.get(ticker, 0.0)) + qty
        _apply_cash(state, ticker, -qty * px)
        state.last_buy_px[ticker] = px
    else:
        state.holdings[ticker] = float(state.holdings.get(ticker, 0.0)) - qty
        _apply_cash(state, ticker, qty * px)
        state.last_sell_px[ticker] = px
    if str(ticker).startswith("HK."):
        state.holdings[ticker] = max(0.0, float(state.holdings[ticker]))
    name = TICKER_NAMES.get(ticker, ticker)
    line = reason or f"FILLED {side} {qty} {name} @ {px:.4f}"
    logger.info(
        "Booked fill (not submit): %s hk_cash=%.2f us_cash=%.2f holdings=%s",
        line,
        state.cash,
        getattr(state, "us_cash", US_INITIAL_CASH),
        state.holdings,
    )
    if persist_fills:
        append_fill(
            {
                "time": datetime.now(tz=HK_TZ).isoformat(),
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "price": px,
                "reason": f"[V2.11] {line}",
                "order_id": str(order.get("order_id") or ""),
            }
        )
        send_telegram_alert(f"[V2.11] {line}")


def settle_orders(state: BotState, broker: FutuPaperBroker, *, persist_fills: bool) -> bool:
    """Book pickle only on fill or cancel, never on submit."""
    try:
        live = broker.list_orders()
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenD order list failed (%s). Leaving pending book unchanged.", exc)
        return False
    if live is None:
        logger.warning("OpenD order list failed on every market. Leaving pending book unchanged.")
        return False
    placed = set(state.placed_order_ids)
    settled = set(state.settled_order_ids)
    pending: list[dict] = []
    changed = False
    for order in live:
        ticker = str(order.get("ticker") or "")
        if ticker not in CORE_TICKERS:
            continue
        oid = str(order.get("order_id") or "")
        kind = str(order.get("kind") or "")
        if kind == "working":
            name = TICKER_NAMES.get(ticker, ticker)
            extra = {}
            for row in state.pending_orders or []:
                if str(row.get("order_id") or "") == oid:
                    extra = {"bar_id": row.get("bar_id"), "submitted_at": row.get("submitted_at")}
                    break
            pending.append(
                {
                    **order,
                    **{k: v for k, v in extra.items() if v},
                    "reason": (
                        f"PENDING {order.get('side')} {int(float(order.get('qty') or 0))} {name} "
                        f"@ {float(order.get('price') or 0):.2f} (not a fill)"
                    ),
                }
            )
            continue
        if oid not in placed or oid in settled:
            continue
        if kind == "filled":
            _apply_futu_fill(state, order, persist_fills=persist_fills, reason="")
            _mark_settled(state, oid)
            changed = True
        elif kind == "cancelled":
            dealt = float(order.get("dealt_qty") or 0.0)
            if dealt > 0:
                part = dict(order)
                part["qty"] = dealt
                _apply_futu_fill(state, part, persist_fills=persist_fills, reason="partial fill before cancel")
            if persist_fills:
                append_fill(
                    {
                        "time": datetime.now(tz=HK_TZ).isoformat(),
                        "ticker": ticker,
                        "side": "CANCEL",
                        "qty": int(float(order.get("qty") or 0) - dealt),
                        "price": float(order.get("price") or 0.0),
                        "reason": "CANCEL unfilled SIMULATE order — book unchanged.",
                        "order_id": oid,
                    }
                )
            logger.info("OpenD cancelled %s %s — pickle not reduced as a fill.", ticker, oid)
            _mark_settled(state, oid)
            changed = True
    state.pending_orders = pending
    if pending:
        logger.info(
            "OpenD pending: %s",
            ", ".join(
                f"{p.get('side')} {int(float(p.get('qty') or 0))} {p.get('ticker')} "
                f"@{float(p.get('price') or 0):.2f}"
                for p in pending
            ),
        )
    return changed


def sync_book_from_opend(state: BotState, broker: FutuPaperBroker) -> None:
    """After stale cancel: trust OpenD positions/cash, not pickle hopes. HK cash ≠ US cash."""
    hk_pos = broker.positions("HK.00700")
    us_pos = broker.positions("US.COST")
    live = {**hk_pos, **us_pos}
    for ticker in CORE_TICKERS:
        if ticker not in live:
            continue
        qty = float(live[ticker])
        if str(ticker).startswith("HK."):
            qty = max(0.0, qty)
        saved = float(state.holdings.get(ticker, 0.0))
        if abs(qty - saved) > 1e-6:
            logger.info("V2.11 sync OpenD position %s: pickle=%.4f -> OpenD=%.4f", ticker, saved, qty)
        state.holdings[ticker] = qty
    hk_acc = broker.accinfo("HK.00700")
    us_acc = broker.accinfo("US.COST")
    if hk_acc and hk_acc.get("cash") is not None:
        state.cash = float(hk_acc["cash"])
        logger.info("V2.11 sync OpenD HK cash=%.2f accinfo=%s", state.cash, hk_acc)
    if us_acc and us_acc.get("cash") is not None:
        state.us_cash = float(us_acc["cash"])
        logger.info("V2.11 sync OpenD US cash=%.2f accinfo=%s", state.us_cash, us_acc)


def cancel_stale(
    state: BotState,
    broker: FutuPaperBroker,
    *,
    now: datetime,
    bar_id: str,
    persist_fills: bool,
) -> bool:
    stale = stale_working_orders(list(state.pending_orders or []), now=now, current_bar_id=bar_id)
    if not stale:
        return False
    cancelled_any = False
    for row in stale:
        ticker = str(row.get("ticker") or "")
        oid = str(row.get("order_id") or "")
        if not ticker or not oid:
            continue
        logger.warning(
            "Stale working %s %s %s @ %s (bar_id=%s age) — cancelling, then re-decide from OpenD.",
            row.get("side"),
            ticker,
            row.get("qty"),
            row.get("price"),
            row.get("bar_id"),
        )
        if broker.cancel_order(ticker, oid):
            _mark_settled(state, oid)
            cancelled_any = True
            if persist_fills:
                append_fill(
                    {
                        "time": datetime.now(tz=HK_TZ).isoformat(),
                        "ticker": ticker,
                        "side": "CANCEL",
                        "qty": int(float(row.get("qty") or 0)),
                        "price": float(row.get("price") or 0.0),
                        "reason": f"[V2.11] stale cancel after bar {row.get('bar_id') or '?'} / {bar_id}",
                        "order_id": oid,
                    }
                )
    if cancelled_any:
        settle_orders(state, broker, persist_fills=persist_fills)
        sync_book_from_opend(state, broker)
    return cancelled_any


def _oldest_working(pending: list[dict[str, Any]], ticker: str) -> str:
    same = [row for row in pending if str(row.get("ticker") or "") == ticker and row.get("order_id")]
    if same:
        return str(same[0].get("order_id") or "")
    any_live = [row for row in pending if row.get("order_id")]
    if not any_live:
        return ""
    return str(any_live[0].get("order_id") or "")


def place_with_compensation(
    broker: FutuPaperBroker,
    state: BotState,
    ticker: str,
    qty: int,
    px: float,
    is_buy: bool,
) -> tuple[bool, str]:
    """Classify OpenD rejects, retry once when legal, never book on failure."""
    if str(ticker).startswith("HK.") and not is_buy:
        if float(state.holdings.get(ticker, 0.0)) <= 0:
            logger.error("Compensation: refuse HK short %s qty=%s (no long to reduce).", ticker, qty)
            send_telegram_alert(f"[V2.11] blocked HK short {ticker} qty={qty}")
            return False, "hk_short"
    ok, extra = broker.place_order(ticker, int(qty), px, is_buy=is_buy)
    if ok:
        return True, extra
    kind = classify_place_error(extra, ticker=ticker)
    send_telegram_alert(f"[V2.11] place_order failed {ticker} classified={kind} raw={extra}")
    if kind == "hk_short":
        logger.error("Compensation: HK short rejected — force qty 0, book unchanged.")
        return False, extra
    if kind in {"price_precision", "lot_qty"}:
        snapped_px = round_to_tick(ticker, px)
        snapped_qty = abs(round_to_lot(ticker, qty))
        if snapped_qty <= 0:
            return False, extra
        logger.warning(
            "Compensation retry once %s qty %s→%s px %s→%s (classified=%s)",
            ticker,
            qty,
            snapped_qty,
            px,
            snapped_px,
            kind,
        )
        return broker.place_order(ticker, snapped_qty, snapped_px, is_buy=is_buy)
    if kind == "buying_power":
        oid = _oldest_working(list(state.pending_orders or []), ticker)
        if oid and broker.cancel_order(ticker, oid):
            _mark_settled(state, oid)
            logger.warning("Compensation: cancelled oldest working %s %s, retry place.", ticker, oid)
            return broker.place_order(ticker, int(qty), px, is_buy=is_buy)
    logger.error("Compensation: no retry for classified=%s. Book unchanged.", kind)
    return False, extra


_shutdown = False


def _handle_stop(signum, _frame) -> None:  # noqa: ANN001
    global _shutdown
    logger.info("Received signal %s — will persist state_v2_11.pkl and exit.", signum)
    _shutdown = True


def _patch_v2_env() -> None:
    import src.inference as inf

    inf.TradingEnv = TradingEnv


def run_loop(
    *,
    once: bool = False,
    dry_run: bool = False,
    poll_seconds: int = 60,
    model_path: Path | None = None,
    skip_catch_up: bool = False,
    predict_now: bool = False,
    push_dashboard: bool = False,
) -> None:
    session_log = attach_session_file_log(prefix="trader_v2_11")
    install_seed_into_v2_11()
    resolved_model = resolve_v2_11_model(model_path)
    refuse_wrong_inference_zip(resolved_model)
    log_seed_banner(resolved_model, role="V2.11 inference")
    logger.info("Session log file (survives closing this window): %s", session_log)
    logger.info(
        "V2.11 paper: obs_dim=%d expected=%d state=%s (does not write state.pkl / news_gpu_v2)",
        V2_11_OBS_DIM,
        observation_dim(),
        STATE_V2_11_PKL,
    )
    if predict_now:
        once = True
        logger.info(
            "PREDICT NOW — one cycle, live quotes + news, no SIMULATE orders, state_v2_11.pkl not written. "
            "Safe while run_trader.bat (V2) is already up."
        )
    persist_state = not predict_now and not dry_run
    if push_dashboard and persist_state:
        logger.warning("V2.11 dashboard push is ON — this shares the V2 blotter. Prefer --predict-now first.")

    _patch_v2_env()
    state = load_v2_11_state(predict_now=predict_now)
    news = NewsPoller()
    broker = FutuPaperBroker(dry_run=dry_run and not predict_now)
    panel = None
    try:
        if ENHANCED_PARQUET.exists():
            import pandas as pd

            panel = pd.read_parquet(ENHANCED_PARQUET)
            logger.info("Loaded V2 panel %s in memory (%d rows). Will not persist it.", ENHANCED_PARQUET, len(panel))
        env = TradingEnv(
            df=panel if panel is not None and not getattr(panel, "empty", True) else None,
            news_scores=state.news_scores,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load enhanced parquet (%s); env may be synthetic.", exc)
        env = TradingEnv(news_scores=state.news_scores)
        panel = None

    model = load_policy(resolved_model)
    probe = env._get_obs()
    logger.info("V2.11 env obs_dim=%d expected=%d", int(probe.shape[0]), V2_11_OBS_DIM)
    if int(probe.shape[0]) != V2_11_OBS_DIM:
        raise RuntimeError(f"V2.11 env obs_dim={probe.shape[0]} expected {V2_11_OBS_DIM}")

    try:
        broker.connect()
        if predict_now:
            logger.info("Predict-now skips Futu position reconcile so live V2 SIMULATE is left alone.")
        else:
            sync_book_from_opend(state, broker)
        if skip_catch_up:
            env.reset()
            env.seek_to_datetime(panel_seek_now(), completed_bars=True)
            env.restore_portfolio(state.cash, state.holdings)
            env.set_news_scores(state.news_scores)
            logger.warning("Catch-up skipped (--skip-catch-up).")
        else:
            env, state, panel = catch_up_env(
                env,
                state,
                broker,
                panel,
                persist_panel=False,
                persist_state=False,
            )
            if persist_state:
                save_state(state, STATE_V2_11_PKL)

        while not _shutdown:
            if not any_core_market_open():
                settled = settle_orders(state, broker, persist_fills=persist_state)
                if persist_state and settled:
                    save_state(state, STATE_V2_11_PKL)
                if predict_now:
                    logger.info(
                        "Both HK and US cash sessions closed. Predict-now still scores the last completed bar (no orders)."
                    )
                else:
                    _maybe_push_dashboard(state, news, push_dashboard and persist_state, kind="heartbeat")
                    wait = min(seconds_until_next_open(), 300)
                    logger.info("Both HK and US cash sessions closed. HOLD. Sleeping %ss.", wait)
                    if once:
                        break
                    time.sleep(wait)
                    continue

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
                            persist_state=False,
                        )
                    else:
                        env.seek_to_datetime(seek_now, completed_bars=True)
                        env.restore_portfolio(state.cash, state.holdings)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("In-session V2.11 catch-up failed (%s). Using current env bar.", exc)

            prices = broker.snapshot_prices(CORE_TICKERS)
            settled = settle_orders(state, broker, persist_fills=persist_state)
            now_loop = _hk_now()
            clock = live_session_clock(now_loop)
            clock_ticker = "US.COST" if clock == "US" else "HK.00700"
            bar_id = session_bar_id(env._current_dt(), now_loop)
            if not predict_now:
                if cancel_stale(state, broker, now=now_loop, bar_id=bar_id, persist_fills=persist_state):
                    settled = True
                    env.restore_portfolio(state.cash, state.holdings)
            if settled:
                env.restore_portfolio(state.cash, state.holdings)

            ready_keys = ready_cash_sessions(now_loop)
            any_session_ready = bool(ready_keys)
            session_today = {key: session_date_iso(key, now_loop) for key in ready_keys}
            first_session = any(state.last_session_ready.get(key) != session_today[key] for key in ready_keys)

            news_prev = dict(state.news_scores)
            news_now = news.fetch_open_markets()
            env.set_news_scores(news_now)
            if any_session_ready:
                news_jump = _news_jumped(news_prev, news_now)
                state.news_scores = news_now
            else:
                news_jump = False

            obs = env._get_obs()
            if not np.all(np.isfinite(obs)) or int(obs.shape[0]) != V2_11_OBS_DIM:
                logger.error("Invalid V2.11 observation (dim=%s). HOLD this cycle.", getattr(obs, "shape", None))
                raw_action = np.zeros(len(CORE_TICKERS), dtype=np.float32)
            else:
                raw_action = np.asarray(predict_action(model, obs), dtype=np.float32).reshape(-1)
            clipped_action = clip_hybrid_action(raw_action).astype(np.float32)
            long_term = env._long_term_features()
            corr_hint = float(np.mean(long_term[-6:])) if len(long_term) >= 6 else 0.0

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

            order_failed = False
            for i, ticker in enumerate(CORE_TICKERS):
                current = float(state.holdings.get(ticker, 0.0))
                raw_i = float(raw_action[i]) if i < len(raw_action) else 0.0
                action_i = float(clipped_action[i]) if i < len(clipped_action) else 0.0
                state.last_action[ticker] = action_i
                if str(ticker).startswith("HK.") and action_i < -1e-9:
                    logger.error("HK action still negative after clip (%s=%+.4f). Forcing 0.", ticker, action_i)
                    action_i = 0.0
                    state.last_action[ticker] = 0.0
                if not is_ticker_market_open(ticker):
                    logger.info(
                        "%s market closed — policy tgt %+.3f (raw %+.3f), keep holdings=%.4f, no order.",
                        ticker,
                        action_i,
                        raw_i,
                        current,
                    )
                    continue
                if not is_cash_open_bar_complete(ticker, now_loop):
                    logger.info(
                        "%s first 10-min cash bar still open — policy tgt %+.3f, keep holdings=%.4f, no order.",
                        ticker,
                        action_i,
                        current,
                    )
                    continue
                px = float(prices.get(ticker) or 0.0)
                if px <= 0 and not dry_run:
                    logger.info("No live price for %s — HOLD.", ticker)
                    continue
                px = px or 1.0
                px = round_to_tick(ticker, px)
                if ticker.startswith("US."):
                    equity = max(_us_equity(state, prices), 1.0)
                else:
                    equity = max(_hk_equity(state, prices), 1.0)
                target_shares = (action_i * equity) / px
                delta = round_to_lot(ticker, target_shares - current)
                delta = _clamp_hk_delta(ticker, delta, current)
                tiny = skip_tiny_rebalance(
                    ticker=ticker,
                    delta=int(delta),
                    px=px,
                    current=current,
                    equity=equity,
                    target_weight=action_i,
                )
                if tiny and delta != 0:
                    logger.info("%s", tiny)
                    delta = 0
                reason = _explain(
                    ticker,
                    action_i,
                    delta,
                    news_now.get(ticker, 0.0),
                    news_prev.get(ticker, 0.0),
                    corr_hint,
                    current=current,
                    target_shares=target_shares,
                    raw_action=raw_i,
                )
                prefix = "[predict-now] " if predict_now else ("" if allow_orders else "[preview, no order] ")
                logger.info("%s%s", prefix, reason)
                state.last_reason = reason
                if predict_now or not allow_orders or delta == 0:
                    continue
                is_buy = delta > 0
                last_buy = state.last_buy_px.get(ticker)
                last_sell = state.last_sell_px.get(ticker)
                decision = decide_order(
                    ticker=ticker,
                    is_buy=is_buy,
                    qty=int(abs(delta)),
                    px=px,
                    pending=list(state.pending_orders or []),
                    last_buy_px=float(last_buy) if last_buy else None,
                    last_sell_px=float(last_sell) if last_sell else None,
                )
                logger.info("%s%s", prefix, decision.reason)
                if decision.action == "skip":
                    continue
                if dry_run:
                    logger.info(
                        "[DRY-RUN] would %s %s qty=%s — book unchanged until fill.",
                        "BUY" if is_buy else "SELL",
                        ticker,
                        abs(delta),
                    )
                    continue
                if decision.action == "replace":
                    cancelled_ok = True
                    for oid in decision.cancel_ids:
                        if not oid:
                            continue
                        if broker.cancel_order(ticker, oid):
                            _mark_settled(state, oid)
                            state.pending_orders = [
                                row for row in state.pending_orders if str(row.get("order_id") or "") != oid
                            ]
                        else:
                            cancelled_ok = False
                    if not cancelled_ok:
                        logger.warning("Could not cancel working %s order(s) — skip new order this cycle.", ticker)
                        continue
                ok, order_id = place_with_compensation(
                    broker, state, ticker, int(abs(delta)), px, is_buy=is_buy
                )
                if ok:
                    _remember_placed(state, order_id)
                    state.pending_orders.append(
                        {
                            "order_id": order_id,
                            "ticker": ticker,
                            "side": "BUY" if is_buy else "SELL",
                            "qty": float(abs(delta)),
                            "price": float(px),
                            "kind": "working",
                            "status": "SUBMITTED",
                            "time": datetime.now(tz=HK_TZ).isoformat(),
                            "submitted_at": datetime.now(tz=HK_TZ).isoformat(),
                            "bar_id": bar_id,
                            "reason": (
                                f"PENDING {'BUY' if is_buy else 'SELL'} {int(abs(delta))} "
                                f"{TICKER_NAMES.get(ticker, ticker)} @ {px:.2f} (not a fill)"
                            ),
                        }
                    )
                    logger.info(
                        "Submitted %s %s qty=%s @ %s order_id=%s — pickle unchanged until fill.",
                        "BUY" if is_buy else "SELL",
                        ticker,
                        abs(delta),
                        px,
                        order_id,
                    )
                else:
                    order_failed = True

            if allow_orders and not order_failed:
                state.last_order_bar = bar_id
                state.last_session_ready.update(session_today)
            _mark_books(state, prices)
            state.last_bar_datetime = bar_id
            env.restore_portfolio(state.cash, state.holdings)
            if persist_state and (settled or allow_orders or state.pending_orders):
                save_state(state, STATE_V2_11_PKL)
            else:
                logger.info(
                    "No fills this cycle. clock=%s bar=%s HK equity≈%.2f cash=%.2f | US equity≈%.2f cash=%.2f",
                    clock,
                    bar_id,
                    state.equity,
                    state.cash,
                    _us_equity(state, prices),
                    getattr(state, "us_cash", US_INITIAL_CASH),
                )
            _maybe_push_dashboard(state, news, push_dashboard and persist_state, prices=prices)

            if predict_now:
                logger.info("============================================================")
                logger.info(
                    "PREDICT NOW V2.11  bar=%s  equity≈%.2f  cash=%.2f  (HK actions must be >= 0)",
                    state.last_bar_datetime,
                    state.equity,
                    state.cash,
                )
                for ticker in CORE_TICKERS:
                    act = float(state.last_action.get(ticker, 0.0))
                    flag = ""
                    if str(ticker).startswith("HK.") and act < -1e-9:
                        flag = "  *** HK NEGATIVE — BUG ***"
                    logger.info(
                        "  %s  action=%+.3f  holdings=%.4f  news=%.3f  %s%s",
                        ticker,
                        act,
                        float(state.holdings.get(ticker, 0.0)),
                        float(news_now.get(ticker, 0.0)),
                        "OPEN" if is_ticker_market_open(ticker) else "CLOSED",
                        flag,
                    )
                logger.info("============================================================")

            if once:
                if persist_state:
                    save_state(state, STATE_V2_11_PKL)
                break
            time.sleep(max(poll_seconds, 5))
    finally:
        if persist_state:
            save_state(state, STATE_V2_11_PKL)
            logger.info("Shutdown complete. state_v2_11.pkl is current. V2 state.pkl was not touched.")
        else:
            logger.info("Predict-now complete. state_v2_11.pkl was not written. V2 state.pkl was not touched.")
        broker.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AirAire V2.11 hybrid paper trader (does not touch live V2).")
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--predict-now",
        action="store_true",
        help="One live predict: quotes + news, no orders, do not write state_v2_11.pkl. Safe while V2 is running.",
    )
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="V2 or V2.11 zip (782-dim). Default: models/news_gpu_v2_11/best_model.zip.",
    )
    p.add_argument("--skip-catch-up", action="store_true")
    p.add_argument(
        "--push-dashboard",
        action="store_true",
        help="Push snapshots to the V2 blotter (off by default).",
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
        push_dashboard=args.push_dashboard,
    )
    sys.exit(0)
