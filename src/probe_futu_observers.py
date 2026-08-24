"""Quote-only OpenD probe: can we get HSI and SPX (or a listed ETF proxy)?

Does not place orders. Does not write pickle. Safe while the V3.2 trader is running
(OpenD allows a second quote connection).

    python -m src.probe_futu_observers

Paste the table back. We already know from today's replay:
  HK.800000 → HSI bars work
  US.SPX / US..SPX → \"US stock indices are not supported\"
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.futu_codes import HSI_CODES, OBSERVER_PROBE_CODES, SPX_INDEX_CODES, SPX_PROXY_ETFS
from src.utils import FUTU_HOST, FUTU_PORT, HK_TZ, RateLimiter, setup_logging

logger = setup_logging("airaire.probe_futu_observers")
_limiter = RateLimiter()


def _quote_ctx():
    from futu import OpenQuoteContext

    _limiter.acquire()
    ctx = OpenQuoteContext(host=FUTU_HOST, port=FUTU_PORT)
    return ctx


def _search_names(ctx, market, stock_type, needles: tuple[str, ...]) -> list[tuple[str, str]]:
    from futu import RET_OK

    _limiter.acquire()
    ret, data = ctx.get_stock_basicinfo(market=market, stock_type=stock_type)
    if ret != RET_OK or data is None or len(data) == 0:
        logger.info("basicinfo %s %s: %s", market, stock_type, data if ret != RET_OK else "empty")
        return []
    hits: list[tuple[str, str]] = []
    name_col = "name" if "name" in data.columns else None
    code_col = "code" if "code" in data.columns else None
    if code_col is None:
        return []
    lower = tuple(n.lower() for n in needles)
    for _, row in data.iterrows():
        code = str(row[code_col])
        name = str(row[name_col] or "") if name_col else ""
        blob = f"{code} {name}".lower()
        if any(n in blob for n in lower):
            hits.append((code, name))
    return hits


def _snapshot(ctx, code: str) -> str:
    from futu import RET_OK

    _limiter.acquire()
    ret, data = ctx.get_market_snapshot([code])
    if ret != RET_OK:
        return f"FAIL {data}"
    if data is None or len(data) == 0:
        return "empty"
    row = data.iloc[0]
    last = row["last_price"] if "last_price" in data.columns else "?"
    status = row["suspension"] if "suspension" in data.columns else ""
    return f"last={last} suspension={status}"


def _klines(ctx, code: str, start: str, end: str) -> str:
    from futu import AuType, KLType, RET_OK

    _limiter.acquire()
    try:
        ret, data, _page = ctx.request_history_kline(
            code,
            start=start,
            end=end,
            ktype=KLType.K_10M,
            autype=AuType.QFQ,
            max_count=100,
        )
    except Exception as exc:  # noqa: BLE001
        return f"raised {type(exc).__name__}: {exc}"
    if ret != RET_OK:
        return f"FAIL {data}"
    if data is None or len(data) == 0:
        return "0 rows"
    tcol = "time_key" if "time_key" in data.columns else data.columns[0]
    last_t = data[tcol].iloc[-1]
    last_c = data["close"].iloc[-1] if "close" in data.columns else "?"
    return f"{len(data)} rows  last={last_t} close={last_c}"


def main() -> int:
    print("Futu observer probe (quote only, no orders).")
    print(f"OpenD {FUTU_HOST}:{FUTU_PORT}")
    print("")
    try:
        from futu import KLType, Market, SecurityType  # noqa: F401
    except ImportError:
        print("futu-api is not installed in this venv.")
        return 2

    end = datetime.now(tz=HK_TZ).date()
    start = end - timedelta(days=5)
    start_s, end_s = start.isoformat(), end.isoformat()

    ctx = _quote_ctx()
    try:
        from futu import Market, SecurityType

        print("--- index / ETF name search (needles: HSI, Hang Seng, SPX, S&P, SPY) ---")
        searches = [
            (Market.HK, SecurityType.IDX, ("hsi", "hang seng", "恒生")),
            (Market.US, SecurityType.IDX, ("spx", "s&p", "s & p", "500")),
            (Market.US, SecurityType.ETF, ("spy", "voo", "ivv", "s&p")),
        ]
        extra: list[str] = []
        for market, stype, needles in searches:
            try:
                hits = _search_names(ctx, market, stype, needles)
            except Exception as exc:  # noqa: BLE001
                print(f"  search {market} {stype}: {exc}")
                continue
            print(f"  {market} {stype}: {len(hits)} hit(s)")
            for code, name in hits[:15]:
                print(f"    {code:<16} {name}")
                extra.append(code)

        codes: list[str] = []
        seen: set[str] = set()
        for code in list(OBSERVER_PROBE_CODES) + extra:
            if code in seen:
                continue
            seen.add(code)
            codes.append(code)

        print("")
        print(f"--- snapshot + 10-min kline {start_s} → {end_s} ---")
        print(f"{'code':<16} {'kind':<8} snapshot                              kline")
        for code in codes:
            if code in HSI_CODES:
                kind = "HSI"
            elif code in SPX_INDEX_CODES:
                kind = "SPXidx"
            elif code in SPX_PROXY_ETFS:
                kind = "SPXetf"
            else:
                kind = "search"
            snap = _snapshot(ctx, code)
            kl = _klines(ctx, code, start_s, end_s)
            print(f"{code:<16} {kind:<8} {snap:<36} {kl}")

        print("")
        print("How to read this:")
        print("  HSI: HK.800000 kline rows > 0 means V4 can overlay Hang Seng from OpenD.")
        print("  SPXidx FAIL + SPXetf rows > 0 means cash SPX is blocked; SPY is a different")
        print("  instrument. Do not train on Bloomberg SPX and live-feed SPY into that slot.")
        print("  If both SPXidx and SPXetf fail, V4 still trains on Bloomberg SPX; live US")
        print("  session will ffill the last Bloomberg bar until you export a fresh CSV.")
    finally:
        ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
