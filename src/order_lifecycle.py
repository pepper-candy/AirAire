"""Live order common sense: pending vs fill, and do not round-trip at a worse price.

Training still assumes instant fills. This layer is OpenD-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.utils import CORE_TICKERS, HK_TZ, TICKER_NAMES, ticker_market

# Skip COST/KO 7-vs-14 flicker. HK lot=100 already filters most noise.
MIN_NOTIONAL_HKD = 10_000.0
MIN_NOTIONAL_USD = 8_000.0
MIN_WEIGHT_STEP = 0.01
# One completed 10-min bar + slack. OpenD does not cancel leftovers for us.
STALE_ORDER_MINUTES = 12.0

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


def working_pending_rows(live: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenD working limits for the dashboard (not fills)."""
    rows: list[dict[str, Any]] = []
    for order in live:
        ticker = str(order.get("ticker") or "")
        if ticker not in CORE_TICKERS:
            continue
        if str(order.get("kind") or "") != "working":
            continue
        name = TICKER_NAMES.get(ticker, ticker)
        side = str(order.get("side") or "BUY").upper()
        if side not in {"BUY", "SELL"}:
            side = "BUY"
        qty = int(float(order.get("qty") or 0))
        px = float(order.get("price") or 0.0)
        rows.append(
            {
                **order,
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "price": px,
                "reason": f"PENDING {side} {qty} {name} @ {px:.2f} (not a fill)",
            }
        )
    return rows


def pending_fill_rows(pending: list[dict[str, Any]], *, now_iso: str) -> list[dict[str, Any]]:
    rows = []
    for item in pending:
        ticker = str(item.get("ticker") or "")
        if ticker not in CORE_TICKERS:
            continue
        working_side = str(item.get("working_side") or item.get("side") or "BUY").upper()
        if working_side == "PENDING":
            working_side = "BUY"
        if working_side not in {"BUY", "SELL"}:
            working_side = "BUY"
        qty = int(float(item.get("qty") or 0))
        px = float(item.get("price") or 0.0)
        name = TICKER_NAMES.get(ticker, ticker)
        reason = str(item.get("reason") or "").strip()
        if not reason or reason.upper().startswith("ACTION:"):
            reason = f"PENDING {working_side} {qty} {name} @ {px:.2f} (not a fill)"
        rows.append(
            {
                "time": str(item.get("time") or now_iso),
                "ticker": ticker,
                "side": "PENDING",
                "working_side": working_side,
                "qty": qty,
                "price": px,
                "reason": reason,
                "order_id": str(item.get("order_id") or ""),
            }
        )
    return rows


def overlay_pending_fills(
    fills: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    *,
    now_iso: str,
) -> list[dict[str, Any]]:
    """Rewrite optimistic BUY/SELL blotter lines while OpenD still has them working."""
    pending_rows = pending_fill_rows(pending, now_iso=now_iso)
    by_id = {str(row.get("order_id") or ""): row for row in pending_rows if row.get("order_id")}
    out: list[dict[str, Any]] = []
    used: set[str] = set()
    for fill in fills:
        oid = str(fill.get("order_id") or "")
        if oid and oid in by_id:
            if oid in used:
                continue
            overlay = dict(fill)
            overlay.update(by_id[oid])
            overlay["time"] = str(fill.get("time") or by_id[oid].get("time") or now_iso)
            overlay["side"] = "PENDING"
            out.append(overlay)
            used.add(oid)
            continue
        out.append(fill)
    for oid, row in by_id.items():
        if oid not in used:
            out.append(row)
    return out


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


# ---------------------------------------------------------------------------
# Futu place_order reject taxonomy
# ---------------------------------------------------------------------------
# First matching rule wins. Add new OpenD strings here — never swallow them.
_PLACE_ERROR_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "hk_short",
        (
            "卖空",
            "close only",
            "平仓",
            "not allow short",
            "cannot short",
            "can't short",
            "no short",
            "short selling",
            "short sell",
            "short not",
        ),
    ),
    (
        "price_precision",
        ("价格精度", "price precision", "tick size", "invalid price", "decimal"),
    ),
    (
        "lot_qty",
        ("手数", "lot size", "quantity", "qty invalid", "invalid qty", "not multiple"),
    ),
    (
        "buying_power",
        ("购买力", "buying power", "insufficient", "not enough cash", "max buying"),
    ),
    (
        "price_deviation",
        ("偏离", "max price", "price limit", "too far", "spread"),
    ),
    (
        "session_halt",
        ("停牌", "not tradable", "market closed", "halt", "suspend", "非交易"),
    ),
    (
        "rate_limit",
        ("too frequent", "频率", "rate limit", "throttle"),
    ),
    (
        "duplicate",
        ("重复", "processing", "duplicate", "locked"),
    ),
    (
        "position",
        ("持仓不足", "position not enough", "not enough position", "qty exceed"),
    ),
    (
        "account",
        ("trd_env", "simulate", "unlock", "account", "权限"),
    ),
)

# Standalone "short" after the phrases above so "shortage" / "shortcut" do not match.
_HK_SHORT_WORD = re.compile(r"(?<![a-z])short(?![a-z])", re.IGNORECASE)

KNOWN_PLACE_ERROR_KINDS = frozenset(kind for kind, _ in _PLACE_ERROR_RULES) | {"unknown", "hk_short"}


def classify_place_error(data: Any, *, ticker: str = "") -> str:
    """Map OpenD ``place_order`` ``data`` to a stable kind for logs and compensation."""
    text = str(data or "").strip()
    blob = text.lower()
    for kind, needles in _PLACE_ERROR_RULES:
        for needle in needles:
            if needle.lower() in blob:
                if kind == "hk_short" and ticker and str(ticker).startswith("US."):
                    continue
                return kind
    if ticker and str(ticker).startswith("HK.") and _HK_SHORT_WORD.search(text):
        return "hk_short"
    if not ticker and _HK_SHORT_WORD.search(text):
        return "hk_short"
    return "unknown"


def min_notional_for(ticker: str) -> float:
    try:
        return MIN_NOTIONAL_USD if ticker_market(ticker) == "US" else MIN_NOTIONAL_HKD
    except ValueError:
        return MIN_NOTIONAL_HKD


def skip_tiny_rebalance(
    *,
    ticker: str,
    delta: int,
    px: float,
    current: float,
    equity: float,
    target_weight: float,
) -> str | None:
    """Skip noisy US 7-share round-trips. Flattening to zero is always allowed."""
    if int(delta) == 0:
        return "qty=0"
    after = float(current) + float(delta)
    if abs(after) < 1e-6:
        return None
    notional = abs(float(delta)) * abs(float(px))
    eq = max(abs(float(equity)), 1.0)
    current_w = (float(current) * float(px)) / eq
    gap = abs(float(target_weight) - current_w)
    floor = min_notional_for(ticker)
    if gap < MIN_WEIGHT_STEP:
        return (
            f"skip tiny |Δw|={gap:.4f} < {MIN_WEIGHT_STEP} "
            f"({ticker} Δ{int(delta)} @ {px:.4f} notional={notional:.0f})"
        )
    if notional < floor:
        market = "USD" if str(ticker).startswith("US.") else "HKD"
        return f"skip min-notional {notional:.0f} < {floor:.0f} {market} ({ticker} Δ{int(delta)})"
    return None


def _parse_order_time(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
    ):
        try:
            dt = datetime.strptime(text.replace("Z", "+00:00") if fmt.endswith("%z") else text, fmt)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=HK_TZ)
            return dt
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=HK_TZ)
        return dt
    except ValueError:
        return None


def order_age_minutes(row: dict[str, Any], now: datetime | None = None) -> float | None:
    submitted = _parse_order_time(row.get("submitted_at") or row.get("time"))
    if submitted is None:
        return None
    if now is None:
        now = datetime.now(tz=HK_TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=HK_TZ)
    if submitted.tzinfo is None:
        submitted = submitted.replace(tzinfo=HK_TZ)
    return max((now - submitted.astimezone(now.tzinfo)).total_seconds() / 60.0, 0.0)


def is_stale_working(
    row: dict[str, Any],
    *,
    now: datetime | None = None,
    current_bar_id: str = "",
    stale_minutes: float = STALE_ORDER_MINUTES,
) -> bool:
    """True if this working limit is from a previous completed bar or older than T minutes."""
    bar = str(row.get("bar_id") or "")
    if bar and current_bar_id and bar != str(current_bar_id):
        return True
    age = order_age_minutes(row, now)
    if age is not None and age >= float(stale_minutes):
        return True
    return False


def stale_working_orders(
    pending: list[dict[str, Any]],
    *,
    now: datetime | None = None,
    current_bar_id: str = "",
    stale_minutes: float = STALE_ORDER_MINUTES,
) -> list[dict[str, Any]]:
    return [
        row
        for row in pending
        if is_stale_working(row, now=now, current_bar_id=current_bar_id, stale_minutes=stale_minutes)
    ]
