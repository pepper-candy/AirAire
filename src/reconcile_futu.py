"""Align state_v3.pkl + dashboard to Futu SIMULATE fills.

OpenD position qty is the book. Working limits stay PENDING (not holdings).
HK SIMULATE cash is pickle cash. Stop V3 before --apply or the loop will
overwrite the pickle on the next cycle.

    python -m src.reconcile_futu
    python -m src.reconcile_futu --apply
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any

from src.dashboard_push import configured, fetch_latest_payload, hk_now_iso, push_snapshot, snapshot_from_state
from src.inference import FutuPaperBroker, save_state
from src.inference_v3 import STATE_V3_PKL, load_v3_state
from src.utils import CORE_TICKERS, HK_TZ, TICKER_NAMES, setup_logging

logger = setup_logging("airaire.reconcile_futu")


def _qty_map(raw: dict[str, float]) -> dict[str, float]:
    return {t: float(raw.get(t, 0.0) or 0.0) for t in CORE_TICKERS}


def _pending_from_live(live: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = datetime.now(tz=HK_TZ).isoformat()
    rows: list[dict[str, Any]] = []
    for order in live:
        if str(order.get("kind") or "") != "working":
            continue
        ticker = str(order.get("ticker") or "")
        if ticker not in CORE_TICKERS:
            continue
        name = TICKER_NAMES.get(ticker, ticker)
        qty = int(float(order.get("qty") or 0.0))
        px = float(order.get("price") or 0.0)
        side = str(order.get("side") or "BUY")
        rows.append(
            {
                **order,
                "time": str(order.get("time") or now),
                "reason": f"PENDING {side} {qty} {name} @ {px:.2f} (not a fill)",
            }
        )
    return rows


def _print_table(
    *,
    pickle_h: dict[str, float],
    futu_h: dict[str, float],
    dash_h: dict[str, float],
    pickle_cash: float,
    futu_cash: float,
    dash_cash: float,
    pickle_eq: float,
    futu_eq: float,
    dash_eq: float,
    hk_acc: dict[str, float],
    us_acc: dict[str, float],
    pending: list[dict[str, Any]],
) -> None:
    print("")
    print(f"{'ticker':<10} {'pickle':>10} {'futu':>10} {'dashboard':>12} {'match':>8}")
    drifted = False
    for ticker in CORE_TICKERS:
        a = pickle_h.get(ticker, 0.0)
        b = futu_h.get(ticker, 0.0)
        c = dash_h.get(ticker, 0.0)
        ok = abs(a - b) < 0.5 and abs(c - b) < 0.5
        drifted = drifted or not ok
        print(f"{ticker:<10} {a:10.0f} {b:10.0f} {c:12.0f} {'ok' if ok else 'DRIFT':>8}")
    cash_ok = abs(pickle_cash - futu_cash) < 1.0 and abs(dash_cash - futu_cash) < 1.0
    drifted = drifted or not cash_ok
    print(f"{'cash':<10} {pickle_cash:10.2f} {futu_cash:10.2f} {dash_cash:12.2f} {'ok' if cash_ok else 'DRIFT':>8}")
    print(f"{'equity':<10} {pickle_eq:10.2f} {futu_eq:10.2f} {dash_eq:12.2f}")
    print("")
    print(f"Futu HK accinfo  cash={hk_acc.get('cash', 0):.2f}  market_val={hk_acc.get('market_val', 0):.2f}  total_assets={hk_acc.get('total_assets', 0):.2f}")
    if us_acc:
        print(f"Futu US accinfo  cash={us_acc.get('cash', 0):.2f}  market_val={us_acc.get('market_val', 0):.2f}  total_assets={us_acc.get('total_assets', 0):.2f}  (USD, not mixed into HKD pickle cash)")
    if pending:
        print("Working Futu orders (PENDING, not holdings):")
        for row in pending:
            print(
                f"  {row.get('side')} {int(float(row.get('qty') or 0))} {row.get('ticker')} "
                f"@ {float(row.get('price') or 0):.2f}  id={row.get('order_id')}"
            )
    else:
        print("No working Futu orders.")
    print("")
    if not drifted:
        print("Book already matches Futu fills. --apply only refreshes the dashboard.")
    else:
        print("Futu filled qty + HK cash are the source of truth. CANCEL blotter rows do not move this.")
        print("Stop V3, then re-run with --apply, then restart V3.")


def run(*, apply: bool = False) -> int:
    if not STATE_V3_PKL.exists():
        logger.error("No %s — run this on the GPU box that has the live pickle.", STATE_V3_PKL)
        return 1
    state = load_v3_state()
    dash = fetch_latest_payload() or {}
    broker = FutuPaperBroker()
    broker.connect()
    try:
        if broker.dry_run:
            logger.error("OpenD not connected. Start FutuOpenD and log in first.")
            return 1
        hk_pos = broker.positions("HK.00700")
        us_pos = broker.positions("US.COST")
        live_pos = {**hk_pos, **us_pos}
        hk_acc = broker.accinfo("HK.00700")
        us_acc = broker.accinfo("US.COST")
        prices = broker.snapshot_prices(CORE_TICKERS)
        live_orders = broker.list_orders()
    finally:
        broker.close()

    futu_h = _qty_map(live_pos)
    pickle_h = _qty_map(state.holdings)
    dash_h = _qty_map(dash.get("holdings") or {})
    futu_cash = float(hk_acc.get("cash") or 0.0)
    pickle_cash = float(state.cash or 0.0)
    dash_cash = float(dash.get("cash") or 0.0)
    pending = _pending_from_live(live_orders)
    marks = {t: float(prices.get(t) or 0.0) for t in CORE_TICKERS}
    if any(v <= 0 for t, v in marks.items() if futu_h.get(t, 0.0) > 0.5):
        logger.warning("Missing live marks %s — equity will use pickle prices where needed.", marks)
    for ticker in CORE_TICKERS:
        if marks[ticker] <= 0 and pickle_h.get(ticker, 0.0) > 0.5:
            leftover = float(state.equity or 0.0) - pickle_cash
            qty = pickle_h.get(ticker, 0.0)
            if qty > 1e-9:
                marks[ticker] = max(leftover / qty, 0.01)
    futu_eq = futu_cash + sum(futu_h[t] * marks[t] for t in CORE_TICKERS)
    pickle_eq = float(state.equity or 0.0)
    dash_eq = float(dash.get("equity") or 0.0)

    _print_table(
        pickle_h=pickle_h,
        futu_h=futu_h,
        dash_h=dash_h,
        pickle_cash=pickle_cash,
        futu_cash=futu_cash,
        dash_cash=dash_cash,
        pickle_eq=pickle_eq,
        futu_eq=futu_eq,
        dash_eq=dash_eq,
        hk_acc=hk_acc,
        us_acc=us_acc,
        pending=pending,
    )

    if not apply:
        return 0
    if not hk_acc:
        logger.error("HK accinfo empty — not writing the pickle.")
        return 1

    state.holdings = dict(futu_h)
    state.cash = futu_cash
    state.equity = float(futu_eq)
    state.pending_orders = pending
    placed = {str(x) for x in (state.placed_order_ids or [])}
    placed.update(str(row.get("order_id") or "") for row in pending if row.get("order_id"))
    state.placed_order_ids = [x for x in placed if x]
    catl = futu_h.get("HK.03750", 0.0)
    state.last_reason = (
        f"Reconciled to Futu SIMULATE fills. CATL={catl:.0f} cash={futu_cash:.2f} "
        f"equity={futu_eq:.2f}. Working orders={len(pending)}."
    )
    save_state(state, STATE_V3_PKL)
    payload = snapshot_from_state(state)
    payload["kind"] = "live"
    payload["updated_at"] = hk_now_iso()
    payload["last_reason"] = state.last_reason
    if not configured():
        logger.warning("Dashboard env unset — pickle written, blotter not pushed.")
        return 0
    if not push_snapshot(payload):
        logger.error("Pickle written; dashboard push failed.")
        return 1
    logger.info("Dashboard updated from Futu. Restart V3 so it loads the restored pickle.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare / apply Futu SIMULATE positions onto state_v3.pkl.")
    p.add_argument("--apply", action="store_true", help="Write pickle + dashboard. Default is print-only.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run(apply=args.apply))
