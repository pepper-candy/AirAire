"""V3 paper inference — shared by V3, V3.1, and V3.2.

Same 1082-dim observation (5 tradable + HSI + SPX OHLCV). Actions stay 5 names.
V3 / V3.1: weights in [-1, +1] (Futu SIMULATE still cannot short).
V3.2: weights in [0, 1] (long-only). Detected from the zip path or action_space.

Does **not** write V2 files: ``inference.py``, ``run_trader.bat``, ``state.pkl``,
``models/news_gpu_v2/``, ``data/enhanced/enhanced_data.parquet``.

Own book: ``state_v3.pkl``. If that file is missing, the V2 leftover in
``state.pkl`` (cash, holdings, equity, news) is the starting book so V3 does
not pretend the paper trade is flat. Own panel writes: ``enhanced_v3.parquet``.
``--predict-now`` first (no orders, no state write). Safe while V2 is running.
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
import pandas as pd

from src.data_loader import overlay_live_ohlcv, persist_enhanced_panel
from src.data_loader_v3 import ENHANCED_V3_PARQUET
from src.inference import (
    BotState,
    FutuPaperBroker,
    NewsPoller,
    _hk_now,
    _maybe_push_dashboard,
    _news_jumped,
    predict_action,
    reconcile_with_futu,
    round_to_lot,
)
from src.inference import load_state as _load_state
from src.inference import save_state as _save_state
from src.trading_env_v3 import observation_dim as v3_observation_dim
from src.utils import (
    ALL_TICKERS,
    CORE_TICKERS,
    HK_TZ,
    INITIAL_CASH,
    MODELS_DIR,
    OBSERVER_TICKERS,
    PROJECT_ROOT,
    STATE_PKL,
    any_core_market_open,
    is_cash_open_bar_complete,
    is_kline_complete,
    is_ticker_market_open,
    ready_cash_sessions,
    seconds_until_next_open,
    session_date_iso,
    send_telegram_alert,
    setup_logging,
    TICKER_NAMES,
)
from src.dashboard_push import append_fill

logger = setup_logging("airaire.inference_v3")


def _explain_action(
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
    """Local copy so we do not depend on the VM's older inference.explain_action."""
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

STATE_V3_PKL = PROJECT_ROOT / "state_v3.pkl"
V3_OBS_DIM = v3_observation_dim()


def load_v3_state() -> BotState:
    """Prefer ``state_v3.pkl``. If V3 has never traded, continue V2 leftover."""
    if STATE_V3_PKL.exists():
        return _load_state(STATE_V3_PKL)
    if STATE_PKL.exists():
        state = _load_state(STATE_PKL)
        # V3 has not placed an order yet — allow the first live cycle to rebalance
        # this bar. Cash / holdings / equity / news stay the V2 leftover.
        state.last_order_bar = ""
        state.last_reason = "inherited from V2 state.pkl"
        logger.info(
            "No %s — continuing V2 leftover from %s (cash=%.2f equity=%.2f holdings=%s). "
            "%s is not written. First live V3 cycle may rebalance this bar.",
            STATE_V3_PKL.name,
            STATE_PKL.name,
            state.cash,
            state.equity,
            state.holdings,
            STATE_PKL.name,
        )
        return state
    logger.info(
        "No %s and no %s — starting from a flat book (cash=%.2f).",
        STATE_V3_PKL.name,
        STATE_PKL.name,
        INITIAL_CASH,
    )
    state = BotState()
    state.updated_at = datetime.now(tz=HK_TZ).isoformat()
    return state


# Training tickers → OpenD quote codes to try (first hit wins, then remapped).
# HSI is HK.800000. US cash indices are documented as limited; we try SPX anyway.
FUTU_KLINE_ALIASES: dict[str, tuple[str, ...]] = {
    "HK.00700": ("HK.00700",),
    "HK.03690": ("HK.03690",),
    "HK.03750": ("HK.03750",),
    "US.COST": ("US.COST",),
    "US.KO": ("US.KO",),
    "HK.HSI": ("HK.800000", "HK.HSI"),
    "US.SPX": ("US.SPX", "US..SPX"),
}


def _refuse_v2_zip(path: Path) -> None:
    parts = {str(p) for p in Path(path).resolve().parts}
    if "news_gpu_v2" in parts:
        raise ValueError(
            f"inference_v3 refuses {path}. That is the V2 paper brain (782-dim). "
            "Use python -m src.inference for V2."
        )


def _is_long_only(model_path: Path, flag: bool, model: Any = None) -> bool:
    if flag:
        return True
    parts = {str(p) for p in Path(model_path).resolve().parts}
    if "news_gpu_v3_2" in parts:
        return True
    if model is not None:
        space = getattr(model, "action_space", None)
        low = getattr(space, "low", None)
        if low is not None and float(np.min(np.asarray(low))) >= -1e-8:
            return True
    return False


def resolve_v3_model_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        _refuse_v2_zip(path)
        return path
    for cand in (
        MODELS_DIR / "news_gpu_v3_1" / "best_model.zip",
        MODELS_DIR / "news_gpu_v3" / "best_model.zip",
        MODELS_DIR / "news_gpu_v3_2" / "best_model.zip",
    ):
        if cand.exists():
            _refuse_v2_zip(cand)
            return cand
    return MODELS_DIR / "news_gpu_v3_1" / "best_model.zip"


def log_v3_banner(model_path: Path, *, long_only: bool) -> None:
    resolved = model_path.resolve()
    exists = model_path.exists()
    size = f"{model_path.stat().st_size / (1024 * 1024):.2f} MB" if exists else ""
    logger.info("============================================================")
    logger.info("AirAire inference V3 — model checkpoint")
    logger.info("  path      : %s", resolved)
    logger.info("  exists    : %s%s", exists, f"  ({size})" if size else "")
    logger.info("  obs_dim   : %d (5 core + HSI + SPX)", V3_OBS_DIM)
    logger.info("  actions   : %s", "[0, 1] long-only" if long_only else "[-1, +1] (SIMULATE still cannot short)")
    logger.info("  state     : %s (seed from V2 state.pkl if missing)", STATE_V3_PKL.name)
    logger.info("  panel     : %s", ENHANCED_V3_PARQUET)
    logger.info("============================================================")


def load_v3_policy(model_path: Path):
    _refuse_v2_zip(model_path)
    if not model_path.exists():
        logger.warning("No V3 zip at %s — policy will HOLD (action=0).", model_path)
        return None
    try:
        from stable_baselines3 import PPO

        model = PPO.load(str(model_path))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load model %s: %s", model_path, exc)
        return None
    shape = getattr(getattr(model, "observation_space", None), "shape", None)
    dim = int(shape[0]) if shape else -1
    if dim != V3_OBS_DIM:
        logger.error(
            "Zip obs_dim=%s but V3 env is %d. Refusing this checkpoint (V2 zips are 782).",
            dim,
            V3_OBS_DIM,
        )
        return None
    logger.info("Loaded V3 PPO from %s  obs_dim=%d", model_path.resolve(), dim)
    return model


def make_v3_env(df, news_scores, *, long_only: bool):
    if long_only:
        from src.trading_env_v3_2 import TradingEnv
    else:
        from src.trading_env_v3 import TradingEnv
    return TradingEnv(df=df, news_scores=news_scores)


def _history_klines_v3(broker: FutuPaperBroker, start: datetime | None, end: datetime | None) -> pd.DataFrame:
    """Fetch 7 names. Remap OpenD aliases (HK.800000 → HK.HSI) to training tickers."""
    frames: list[pd.DataFrame] = []
    for train_code in ALL_TICKERS:
        aliases = FUTU_KLINE_ALIASES.get(train_code, (train_code,))
        got = None
        used = None
        for code in aliases:
            part = broker.history_klines(start=start, end=end, tickers=[code])
            if part is not None and not part.empty:
                got = part.copy()
                got["ticker"] = train_code
                used = code
                break
        if got is None:
            logger.warning(
                "No Futu 10-min bars for %s (tried %s). Env will keep parquet/last close.",
                train_code,
                ",".join(aliases),
            )
            continue
        if used != train_code:
            logger.info("Futu klines %s remapped → %s (%d rows).", used, train_code, len(got))
        frames.append(got)
    if not frames:
        return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])
    out = pd.concat(frames, ignore_index=True)
    present = sorted(out["ticker"].astype(str).unique().tolist())
    missing_obs = [t for t in OBSERVER_TICKERS if t not in present]
    if missing_obs:
        logger.warning(
            "Observers missing from live klines: %s. Observation cube will ffill the V3 parquet (may be stale).",
            missing_obs,
        )
    return out


def catch_up_env_v3(
    env,
    state,
    broker: FutuPaperBroker,
    panel,
    *,
    long_only: bool,
    now: datetime | None = None,
    persist_panel: bool = True,
    persist_state: bool = True,
):
    """Same catch-up as V2, but 7 klines and V3 parquet / V3 env / state_v3.pkl."""
    from src.data_loader import default_futu_fetch_start

    now_hk = _hk_now(now).replace(tzinfo=None)
    before_dt = env._current_dt() if len(getattr(env, "datetimes", [])) else None
    live = pd.DataFrame()
    try:
        start = default_futu_fetch_start(panel, now=now_hk)
        live = _history_klines_v3(broker, start.to_pydatetime(), now_hk)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Futu V3 catch-up fetch failed (%s). Seeking on the existing panel only.", exc)

    rebuilt = False
    n_before = 0 if panel is None or getattr(panel, "empty", True) else len(panel)
    panel = overlay_live_ohlcv(panel, live, now=now_hk)
    n_after = 0 if panel is None or panel.empty else len(panel)
    if live is not None and not live.empty:
        logger.info(
            "V3 catch-up merged %d live Futu rows into the panel (%d → %d).",
            len(live),
            n_before,
            n_after,
        )
    if (live is not None and not live.empty) or n_after != n_before:
        env = make_v3_env(
            panel if panel is not None and not panel.empty else None,
            state.news_scores,
            long_only=long_only,
        )
        rebuilt = True
        if persist_panel and panel is not None and not panel.empty:
            persist_enhanced_panel(panel, path=ENHANCED_V3_PARQUET)

    env.reset()
    caught_dt = env.seek_to_datetime(now_hk, completed_bars=True)
    env.restore_portfolio(state.cash, state.holdings)
    env.set_news_scores(state.news_scores)

    live_px = broker.snapshot_prices(CORE_TICKERS)
    if any(float(v or 0.0) > 0 for v in live_px.values()):
        state.equity = state.cash + sum(state.holdings[t] * float(live_px.get(t) or 0.0) for t in CORE_TICKERS)
    else:
        state.equity = float(env._last_equity)

    state.last_bar_datetime = str(caught_dt)
    logger.info(
        "V3 catch-up complete (no orders). env_before=%s env_now=%s bar=%d/%d rebuilt=%s cash=%.2f equity=%.2f",
        before_dt,
        caught_dt,
        env._bar_index,
        max(len(env.datetimes) - 1, 0),
        rebuilt,
        state.cash,
        state.equity,
    )
    if persist_state:
        _save_state(state, STATE_V3_PKL)
    return env, state, panel


def _clip_action(raw: np.ndarray, *, long_only: bool) -> np.ndarray:
    action = np.nan_to_num(np.asarray(raw, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if action.shape[0] != len(CORE_TICKERS):
        logger.error("Action dim %s != 5. Holding.", action.shape)
        return np.zeros(len(CORE_TICKERS), dtype=np.float32)
    if long_only:
        return np.clip(action, 0.0, 1.0)
    return np.clip(action, -1.0, 1.0)


def _clamp_delta(ticker: str, delta: int, current: float, *, reduce_only: bool) -> int:
    if not reduce_only or delta >= 0:
        return delta
    sellable = round_to_lot(ticker, current)
    if sellable <= 0:
        return 0
    return max(int(delta), -int(sellable))


_shutdown = False


def _handle_stop(signum, _frame) -> None:  # noqa: ANN001
    global _shutdown
    logger.info("Received signal %s — will persist state_v3.pkl and exit.", signum)
    _shutdown = True


def run_loop(
    *,
    once: bool = False,
    dry_run: bool = False,
    poll_seconds: int = 60,
    model_path: Path | None = None,
    skip_catch_up: bool = False,
    predict_now: bool = False,
    long_only_flag: bool = False,
    reduce_only: bool = True,
    push_dashboard: bool = False,
) -> None:
    resolved_model = resolve_v3_model_path(model_path)
    long_only = _is_long_only(resolved_model, long_only_flag)
    log_v3_banner(resolved_model, long_only=long_only)
    if predict_now:
        once = True
        logger.info(
            "PREDICT NOW — one cycle, live quotes + news, no SIMULATE orders, state_v3.pkl not written. "
            "Safe while run_trader.bat (V2) is already up."
        )
    persist_state = not predict_now and not dry_run
    if push_dashboard and persist_state:
        logger.warning("V3 dashboard push is ON — this shares the V2 blotter. Prefer --predict-now first.")

    state = load_v3_state()
    news = NewsPoller()
    broker = FutuPaperBroker(dry_run=dry_run and not predict_now)
    panel = None
    try:
        if ENHANCED_V3_PARQUET.exists():
            panel = pd.read_parquet(ENHANCED_V3_PARQUET)
            logger.info("Loaded V3 panel %s (%d rows).", ENHANCED_V3_PARQUET, len(panel))
        else:
            logger.error("%s is missing. Build it with python -m src.data_loader_v3 before live V3.", ENHANCED_V3_PARQUET)
            panel = None
        env = make_v3_env(
            panel if panel is not None and not panel.empty else None,
            state.news_scores,
            long_only=long_only,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load V3 parquet (%s); env may be synthetic.", exc)
        env = make_v3_env(None, state.news_scores, long_only=long_only)
        panel = None

    model = load_v3_policy(resolved_model)
    long_only = _is_long_only(resolved_model, long_only_flag, model)

    probe = env._get_obs()
    logger.info("V3 env obs_dim=%d expected=%d", int(probe.shape[0]), V3_OBS_DIM)
    if int(probe.shape[0]) != V3_OBS_DIM:
        raise RuntimeError(f"V3 env obs_dim={probe.shape[0]} expected {V3_OBS_DIM}")

    try:
        broker.connect()
        if predict_now:
            logger.info(
                "Predict-now skips Futu position reconcile. Scoring against the in-memory book "
                "(V3 file, else V2 leftover); SIMULATE positions stay with run_trader."
            )
        else:
            state = reconcile_with_futu(state, broker)
        if skip_catch_up:
            env.reset()
            env.seek_to_datetime(_hk_now(), completed_bars=True)
            env.restore_portfolio(state.cash, state.holdings)
            env.set_news_scores(state.news_scores)
            logger.warning("Catch-up skipped (--skip-catch-up). Env sought to now without Futu history.")
        else:
            env, state, panel = catch_up_env_v3(
                env,
                state,
                broker,
                panel,
                long_only=long_only,
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
                    _maybe_push_dashboard(state, news, push_dashboard and persist_state, kind="heartbeat")
                    wait = min(seconds_until_next_open(), 300)
                    logger.info("Both HK and US cash sessions closed. HOLD. Sleeping %ss.", wait)
                    if once:
                        break
                    time.sleep(wait)
                    continue

            if not skip_catch_up:
                try:
                    last = pd.Timestamp(env._current_dt())
                    now_naive = pd.Timestamp(_hk_now().replace(tzinfo=None))
                    if pd.notna(last) and (now_naive - last) >= pd.Timedelta(minutes=10):
                        env, state, panel = catch_up_env_v3(
                            env,
                            state,
                            broker,
                            panel,
                            long_only=long_only,
                            persist_panel=False,
                            persist_state=persist_state,
                        )
                    else:
                        env.seek_to_datetime(_hk_now(), completed_bars=True)
                        env.restore_portfolio(state.cash, state.holdings)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("In-session V3 catch-up failed (%s). Using current env bar.", exc)

            prices = broker.snapshot_prices(CORE_TICKERS)
            now_loop = _hk_now()
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
            if not np.all(np.isfinite(obs)) or int(obs.shape[0]) != V3_OBS_DIM:
                logger.error("Invalid V3 observation (dim=%s). HOLD this cycle.", getattr(obs, "shape", None))
                raw_action = np.zeros(len(CORE_TICKERS), dtype=np.float32)
            else:
                raw_action = _clip_action(predict_action(model, obs), long_only=long_only)
            long_term = env._long_term_features()
            corr_hint = float(np.mean(long_term[-6:])) if len(long_term) >= 6 else 0.0

            bar_id = str(env._current_dt())
            new_bar = bar_id != state.last_order_bar
            bar_ready = is_kline_complete(env._current_dt(), now=now_loop)
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
            for i, ticker in enumerate(CORE_TICKERS):
                current = float(state.holdings.get(ticker, 0.0))
                action_i = float(raw_action[i]) if i < len(raw_action) else 0.0
                state.last_action[ticker] = action_i
                if not is_ticker_market_open(ticker):
                    logger.info(
                        "%s market closed — policy tgt %+.3f, keep holdings=%.4f, no order.",
                        ticker,
                        action_i,
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
                    state.last_action[ticker] = 0.0
                    continue
                px = px or 1.0
                equity = max(state.cash + sum(state.holdings[t] * float(prices.get(t) or 0.0) for t in CORE_TICKERS), 1.0)
                target_shares = (action_i * equity) / px
                delta = round_to_lot(ticker, target_shares - current)
                delta = _clamp_delta(ticker, delta, current, reduce_only=reduce_only)
                reason = _explain_action(
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
                    logger.info(
                        "[DRY-RUN] would %s %s qty=%s — book unchanged.",
                        "BUY" if is_buy else "SELL",
                        ticker,
                        abs(delta),
                    )
                    continue
                if ok:
                    state.holdings[ticker] = current + delta
                    state.cash -= delta * px
                    traded = True
                    append_fill(
                        {
                            "time": datetime.now(tz=HK_TZ).isoformat(),
                            "ticker": ticker,
                            "side": "BUY" if is_buy else "SELL",
                            "qty": int(abs(delta)),
                            "price": float(px),
                            "reason": f"[V3] {reason}",
                            "order_id": order_id,
                        }
                    )
                    send_telegram_alert(f"[V3] {reason}")

            if allow_orders:
                state.last_order_bar = bar_id
                state.last_session_ready.update(session_today)
            state.equity = state.cash + sum(state.holdings[t] * float(prices.get(t) or 0.0) for t in CORE_TICKERS)
            state.last_bar_datetime = str(env._current_dt())
            env.restore_portfolio(state.cash, state.holdings)
            if persist_state and (traded or allow_orders):
                _save_state(state, STATE_V3_PKL)
            else:
                logger.info("No fills this cycle. Equity≈%.2f cash=%.2f", state.equity, state.cash)
            _maybe_push_dashboard(state, news, push_dashboard and persist_state)

            if predict_now:
                logger.info("============================================================")
                logger.info(
                    "PREDICT NOW V3  bar=%s  equity≈%.2f  cash=%.2f  long_only=%s",
                    state.last_bar_datetime,
                    state.equity,
                    state.cash,
                    long_only,
                )
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
                    _save_state(state, STATE_V3_PKL)
                break
            time.sleep(max(poll_seconds, 5))
    finally:
        if persist_state:
            _save_state(state, STATE_V3_PKL)
            logger.info("Shutdown complete. state_v3.pkl is current. V2 state.pkl was not touched.")
        else:
            logger.info("Predict-now complete. state_v3.pkl was not written. V2 state.pkl was not touched.")
        broker.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AirAire V3 paper trader (1082-dim; does not touch V2).")
    p.add_argument("--once", action="store_true", help="One cycle then exit (writes state_v3.pkl).")
    p.add_argument("--dry-run", action="store_true", help="Skip OpenD / orders.")
    p.add_argument(
        "--predict-now",
        action="store_true",
        help="One live predict: quotes + news, no orders, do not write state_v3.pkl. Safe while V2 is running.",
    )
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="V3/V3.1/V3.2 zip. Default: models/news_gpu_v3_1/best_model.zip if present.",
    )
    p.add_argument("--skip-catch-up", action="store_true")
    p.add_argument(
        "--long-only",
        action="store_true",
        help="Force [0, 1] actions (V3.2). Auto-on when the zip path contains news_gpu_v3_2.",
    )
    p.add_argument(
        "--allow-shorts",
        action="store_true",
        help="Do not clamp sells to current holdings. Futu SIMULATE will still reject true shorts.",
    )
    p.add_argument(
        "--push-dashboard",
        action="store_true",
        help="Push snapshots to the V2 blotter (off by default so V2 paper is not overwritten).",
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
        long_only_flag=args.long_only,
        reduce_only=not args.allow_shorts,
        push_dashboard=args.push_dashboard,
    )
    sys.exit(0)
