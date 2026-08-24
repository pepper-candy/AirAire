"""Reverse unfilled Futu SIMULATE CATL sells that were booked on submit.

Orders 8899494 (400 @ 638) and 8899530 (700 @ 637.5) stayed pending.
The live book treated RET_OK as a fill. This restores shares + cash and
marks those blotter rows CANCEL.

Stop the V3 trader first, run this, then restart — otherwise the in-memory
loop will push CATL=200 again.
"""

from __future__ import annotations

import json
from typing import Any

from src.dashboard_push import (
    configured,
    fetch_latest_payload,
    hk_now_iso,
    push_snapshot,
    rewrite_trades_jsonl,
)
from src.utils import CORE_TICKERS, INITIAL_CASH, setup_logging

logger = setup_logging("airaire.correct_unfilled")

# Pending CATL sells from 2026-08-24 15:07 / 15:20 HKT.
UNFILLED_SELLS: tuple[dict[str, Any], ...] = (
    {"order_id": "8899494", "ticker": "HK.03750", "qty": 400, "price": 638.0},
    {"order_id": "8899530", "ticker": "HK.03750", "qty": 700, "price": 637.5},
)


def _mark_price(payload: dict[str, Any], ticker: str) -> float:
    holdings = payload.get("holdings") or {}
    qty = float(holdings.get(ticker, 0.0) or 0.0)
    cash = float(payload.get("cash") or 0.0)
    equity = float(payload.get("equity") or 0.0)
    if qty > 1e-9:
        return max((equity - cash) / qty, 0.01)
    return 637.0


def reverse_unfilled(payload: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(payload))
    holdings = {t: float((out.get("holdings") or {}).get(t, 0.0) or 0.0) for t in CORE_TICKERS}
    cash = float(out.get("cash") or INITIAL_CASH)
    ids = {str(row["order_id"]) for row in UNFILLED_SELLS}
    marks = {t: _mark_price(out, t) for t in CORE_TICKERS}

    restored = []
    for row in UNFILLED_SELLS:
        ticker = str(row["ticker"])
        qty = int(row["qty"])
        px = float(row["price"])
        holdings[ticker] = float(holdings.get(ticker, 0.0)) + qty
        cash -= qty * px
        restored.append(f"{qty} {ticker}")

    equity = cash + sum(holdings[t] * float(marks.get(t) or 0.0) for t in CORE_TICKERS)
    fills = []
    for fill in list(out.get("fills") or []):
        item = dict(fill)
        if str(item.get("order_id") or "") in ids:
            item["side"] = "CANCEL"
            prev = str(item.get("reason") or "")
            item["reason"] = "CANCEL unfilled limit — shares restored. " + prev
        fills.append(item)

    out["cash"] = float(cash)
    out["holdings"] = holdings
    out["equity"] = float(equity)
    out["pnl"] = float(equity) - float(out.get("initial_cash") or INITIAL_CASH)
    out["fills"] = fills
    out["kind"] = "live"
    out["updated_at"] = hk_now_iso()
    out["last_reason"] = (
        "Unfilled CATL sells cancelled (400 @ 638 + 700 @ 637.50). "
        f"Book restored; holdings CATL={holdings['HK.03750']:.0f}."
    )
    logger.info(
        "Reversed unfilled sells %s cash=%.2f equity=%.2f CATL=%.0f",
        restored,
        cash,
        equity,
        holdings["HK.03750"],
    )
    return out


def patch_state_v3(payload: dict[str, Any]) -> bool:
    from src.inference import save_state
    from src.inference_v3 import STATE_V3_PKL, load_v3_state

    if not STATE_V3_PKL.exists():
        logger.warning("No %s in this folder — blotter pushed; copy this script onto the GPU box to patch the pickle.", STATE_V3_PKL)
        return False
    state = load_v3_state()
    state.cash = float(payload["cash"])
    state.equity = float(payload["equity"])
    state.holdings = {t: float((payload.get("holdings") or {}).get(t, 0.0) or 0.0) for t in CORE_TICKERS}
    state.last_reason = str(payload.get("last_reason") or state.last_reason)
    save_state(state, STATE_V3_PKL)
    logger.info("Patched %s CATL=%.0f cash=%.2f", STATE_V3_PKL.name, state.holdings["HK.03750"], state.cash)
    return True


def patch_trades_jsonl() -> None:
    rewrite_trades_jsonl({str(row["order_id"]) for row in UNFILLED_SELLS})


def run(*, dry_run: bool = False) -> int:
    payload = fetch_latest_payload()
    if not payload:
        logger.error("No latest snapshot to correct.")
        return 1
    catl = float((payload.get("holdings") or {}).get("HK.03750") or 0.0)
    if catl >= 1299:
        logger.info("CATL already restored (%.0f). Not reversing twice.", catl)
        return 0
    corrected = reverse_unfilled(payload)
    if dry_run:
        logger.info("Dry-run cash=%.2f equity=%.2f CATL=%.0f", corrected["cash"], corrected["equity"], corrected["holdings"]["HK.03750"])
        return 0
    if not configured():
        logger.error("DASHBOARD_PUSH_URL / DASHBOARD_PUSH_KEY unset.")
        return 2
    if not push_snapshot(corrected):
        logger.error("Dashboard push failed.")
        return 1
    patch_trades_jsonl()
    patch_state_v3(corrected)
    logger.info("Blotter updated. Stop/restart V3 trader so it loads the restored book.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
