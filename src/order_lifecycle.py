"""Live order common sense: pending vs fill, and do not round-trip at a worse price.

Training still assumes instant fills. This layer is OpenD-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils import CORE_TICKERS, TICKER_NAMES

WORKING = frozenset(
    {
        "NONE",
        "WAITING_SUBMIT",
        "SUBMITTING",
        "SUBMITTED",
        "FILLED_PART",
        "WAITING",
    }
)
FILLED = frozenset({"FILLED_ALL"})
CANCELLED = frozenset(
    {
        "CANCELLED_PART",
        "CANCELLED_ALL",
        "CANCELLED",
        "FAILED",
        "DISABLED",
        "DELETED",
    }
)


def _enum_name(raw: Any) -> str:
    text = str(getattr(raw, "name", raw) or "")
    text = text.replace("OrderStatus.", "").replace("TrdSide.", "").replace("orderstatus.", "")
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.strip().upper()


def status_kind(raw: Any) -> str:
    name = _enum_name(raw)
    if name in FILLED or (name.startswith("FILLED") and "ALL" in name and "PART" not in name):
        return "filled"
    if name in CANCELLED or name.startswith("CANCEL"):
        return "cancelled"
    if name in WORKING or "SUBMIT" in name or name.endswith("_PART"):
        return "working"
    return "other"


def side_name(raw: Any) -> str:
    name = _enum_name(raw)
    if "SELL" in name or name in {"SHORT", "SELL_SHORT"}:
        return "SELL"
    return "BUY"


@dataclass
class OrderDecision:
    action: str  # place | skip | replace
    reason: str
    cancel_ids: tuple[str, ...] = ()


def working_on_ticker(pending: list[dict[str, Any]], ticker: str) -> list[dict[str, Any]]:
    return [row for row in pending if str(row.get("ticker") or "") == ticker]


def decide_order(
    *,
    ticker: str,
    is_buy: bool,
    qty: int,
    px: float,
    pending: list[dict[str, Any]],
    last_buy_px: float | None,
    last_sell_px: float | None,
) -> OrderDecision:
    """Skip round-trips that pay up; otherwise cancel working lots and place the new size."""
    if qty <= 0:
        return OrderDecision("skip", "qty=0")
    name = TICKER_NAMES.get(ticker, ticker)
    new_side = "BUY" if is_buy else "SELL"

    if not is_buy and last_buy_px is not None and px <= float(last_buy_px) + 1e-9:
        return OrderDecision(
            "skip",
            f"skip sell {name} @ {px:.4f} — not above last buy {float(last_buy_px):.4f} (spread+fee)",
        )
    if is_buy and last_sell_px is not None and px >= float(last_sell_px) - 1e-9:
        return OrderDecision(
            "skip",
            f"skip buy {name} @ {px:.4f} — not below last sell {float(last_sell_px):.4f} (spread+fee)",
        )

    live = working_on_ticker(pending, ticker)
    if not live:
        return OrderDecision("place", f"place {new_side} {qty} {name} @ {px:.4f}")

    ids = tuple(str(row.get("order_id") or "") for row in live if row.get("order_id"))
    sells = [row for row in live if str(row.get("side") or "").upper() == "SELL"]
    buys = [row for row in live if str(row.get("side") or "").upper() == "BUY"]

    if is_buy and sells:
        best_sell = min(float(row.get("price") or 0.0) for row in sells)
        if px < best_sell - 1e-9:
            return OrderDecision(
                "replace",
                f"cancel pending sell @ {best_sell:.4f}, buy {qty} {name} cheaper @ {px:.4f}",
                ids,
            )
        return OrderDecision(
            "skip",
            f"skip buy {name} @ {px:.4f} — pending sell @ {best_sell:.4f} is not worse",
        )
    if (not is_buy) and buys:
        best_buy = max(float(row.get("price") or 0.0) for row in buys)
        if px > best_buy + 1e-9:
            return OrderDecision(
                "replace",
                f"cancel pending buy @ {best_buy:.4f}, sell {qty} {name} higher @ {px:.4f}",
                ids,
            )
        return OrderDecision(
            "skip",
            f"skip sell {name} @ {px:.4f} — pending buy @ {best_buy:.4f} is not worse",
        )

    # Same side: only replace if the new price is better (or equal) — then use the new amount.
    if is_buy and buys:
        best_buy = min(float(row.get("price") or 0.0) for row in buys)
        if px <= best_buy + 1e-9:
            return OrderDecision(
                "replace",
                f"replace pending buy @ {best_buy:.4f} with {qty} {name} @ {px:.4f}",
                ids,
            )
        return OrderDecision(
            "skip",
            f"keep pending buy @ {best_buy:.4f} — new buy @ {px:.4f} is more expensive",
        )
    if (not is_buy) and sells:
        best_sell = max(float(row.get("price") or 0.0) for row in sells)
        if px >= best_sell - 1e-9:
            return OrderDecision(
                "replace",
                f"replace pending sell @ {best_sell:.4f} with {qty} {name} @ {px:.4f}",
                ids,
            )
        return OrderDecision(
            "skip",
            f"keep pending sell @ {best_sell:.4f} — new sell @ {px:.4f} is cheaper (worse)",
        )
    return OrderDecision("replace", f"replace working {name} with {new_side} {qty} @ {px:.4f}", ids)


def pending_fill_rows(pending: list[dict[str, Any]], *, now_iso: str) -> list[dict[str, Any]]:
    rows = []
    for item in pending:
        ticker = str(item.get("ticker") or "")
        if ticker not in CORE_TICKERS:
            continue
        rows.append(
            {
                "time": str(item.get("time") or now_iso),
                "ticker": ticker,
                "side": "PENDING",
                "qty": int(float(item.get("qty") or 0)),
                "price": float(item.get("price") or 0.0),
                "reason": str(item.get("reason") or "Working SIMULATE order (not a fill)."),
                "order_id": str(item.get("order_id") or ""),
            }
        )
    return rows


def parse_order_row(row: Any) -> dict[str, Any] | None:
    try:
        ticker = str(row.get("code") or "")
        oid = str(row.get("order_id") or "")
        if not ticker or not oid:
            return None
        qty = float(row.get("qty") or 0.0)
        dealt = float(row.get("dealt_qty") or 0.0)
        px = float(row.get("price") or 0.0)
        dealt_px = float(row.get("dealt_avg_price") or 0.0) or px
        return {
            "order_id": oid,
            "ticker": ticker,
            "side": side_name(row.get("trd_side")),
            "qty": qty,
            "price": px,
            "dealt_qty": dealt,
            "dealt_avg_price": dealt_px,
            "status": _enum_name(row.get("order_status")),
            "kind": status_kind(row.get("order_status")),
            "time": str(row.get("create_time") or row.get("updated_time") or ""),
        }
    except Exception:  # noqa: BLE001
        return None
