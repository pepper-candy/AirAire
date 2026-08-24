import { isHkCashOpen, isUsCashOpen } from "@/lib/marketHours";
import {
  CORE_TICKERS,
  HK_INITIAL_CASH,
  HK_TICKERS,
  PHANTOM_CATL_SELLS,
  US_INITIAL_CASH,
  US_TICKERS,
  type Fill,
  type Snapshot,
} from "@/lib/types";

/** Unfilled CATL limit sells that were booked as fills (qty×price, no Futu fee). */
export const PHANTOM_PROCEEDS = PHANTOM_CATL_SELLS.reduce((sum, row) => sum + row.qty * row.price, 0);

const PHANTOM_IDS: Set<string> = new Set(PHANTOM_CATL_SELLS.map((row) => row.order_id));

export function isHkTicker(ticker: string): boolean {
  return ticker.startsWith("HK.");
}

export function pendingIntent(fill: Fill): "BUY" | "SELL" | "" {
  const working = (fill.working_side || "").toUpperCase();
  if (working === "BUY" || working === "SELL") {
    return working;
  }
  const reason = fill.reason || "";
  const tagged = reason.match(/\bPENDING\s+(BUY|SELL)\b/i);
  if (tagged) {
    return tagged[1].toUpperCase() as "BUY" | "SELL";
  }
  const action = reason.match(/\bAction:\s*(Buy|Sell)\b/i);
  if (action) {
    return action[1].toUpperCase() === "SELL" ? "SELL" : "BUY";
  }
  return "";
}

export function isRealFill(fill: Fill): boolean {
  const side = (fill.side || "").toUpperCase();
  if (side === "CANCEL" || side === "CANCELLED" || side === "PENDING") {
    return false;
  }
  if (PHANTOM_IDS.has(String(fill.order_id || "")) && side === "SELL") {
    return false;
  }
  return side === "BUY" || side === "SELL";
}

export function normalizeFills(fills: Fill[], pendingIds: Iterable<string> = []): Fill[] {
  const pending = new Set(
    [...pendingIds].map((id) => String(id || "")).filter(Boolean),
  );
  for (const fill of fills) {
    if ((fill.side || "").toUpperCase() === "PENDING" && fill.order_id) {
      pending.add(String(fill.order_id));
    }
  }

  const out: Fill[] = [];
  const seenPending = new Set<string>();
  for (const fill of fills) {
    const oid = String(fill.order_id || "");
    let next = fill;
    const side = (fill.side || "").toUpperCase();
    if (PHANTOM_IDS.has(oid) && side !== "CANCEL" && side !== "CANCELLED") {
      next = {
        ...next,
        side: "CANCEL",
        reason: `CANCEL unfilled limit — shares restored. ${fill.reason || ""}`.trim(),
      };
    } else if (oid && pending.has(oid) && (side === "BUY" || side === "SELL" || side === "PENDING")) {
      if (seenPending.has(oid)) {
        continue;
      }
      const intent = pendingIntent(next) || (side === "SELL" ? "SELL" : "BUY");
      next = {
        ...next,
        side: "PENDING",
        working_side: next.working_side || intent,
        reason:
          next.reason && /\bnot a fill\b/i.test(next.reason)
            ? next.reason
            : `PENDING ${intent} ${next.qty} (not a fill). ${next.reason || ""}`.trim(),
      };
      seenPending.add(oid);
    }
    out.push(next);
  }
  return out;
}

function lastMarks(fills: Fill[], prices?: Record<string, number>): Record<string, number> {
  const marks: Record<string, number> = {};
  const chrono = [...fills].filter(isRealFill).sort((a, b) => Date.parse(a.time) - Date.parse(b.time));
  for (const fill of chrono) {
    if (fill.price > 0) {
      marks[fill.ticker] = fill.price;
    }
  }
  // Live / last-close from OpenD wins. Fill prices are cash only — after HK close
  // the blotter still says 638 while Futu marks the 16:00 close (~638.50 → ~−7,300).
  for (const [ticker, px] of Object.entries(prices || {})) {
    const n = Number(px);
    if (n > 0) {
      marks[ticker] = n;
    }
  }
  return marks;
}

function mtm(tickers: readonly string[], qty: Record<string, number>, marks: Record<string, number>): number {
  return tickers.reduce((sum, ticker) => sum + (qty[ticker] || 0) * (marks[ticker] || 0), 0);
}

export type MarketBook = {
  market: "HK" | "US";
  open: boolean;
  currency: "HKD" | "USD";
  cash: number;
  equity: number;
  pnl: number;
  names: string[];
};

export type SplitBooks = {
  fills: Fill[];
  holdings: Record<string, number>;
  marks: Record<string, number>;
  hk: MarketBook;
  us: MarketBook;
};

/**
 * Two Futu SIMULATE accounts: HK HKD and US USD. Cancels are not fills.
 * Does not apply OpenD handling fees (those are still unknown).
 */
export function splitBooks(snap: Snapshot | null, now: Date = new Date()): SplitBooks {
  const raw = snap?.fills || [];
  const fills = normalizeFills(raw, snap?.pending_order_ids);
  const marks = lastMarks(raw, snap?.prices);
  const qty: Record<string, number> = {};
  for (const ticker of CORE_TICKERS) {
    qty[ticker] = 0;
  }

  const real = [...fills].filter(isRealFill).sort((a, b) => Date.parse(a.time) - Date.parse(b.time));
  const hkStart = snap?.initial_cash ?? HK_INITIAL_CASH;
  let hkCash = hkStart;
  let usCash = snap?.us_cash ?? US_INITIAL_CASH;
  const touched = new Set<string>();

  if (real.length > 0) {
    usCash = US_INITIAL_CASH;
    for (const fill of real) {
      const buy = (fill.side || "").toUpperCase() === "BUY";
      const signedQty = buy ? fill.qty : -fill.qty;
      qty[fill.ticker] = (qty[fill.ticker] || 0) + signedQty;
      touched.add(fill.ticker);
      const flow = buy ? -fill.qty * fill.price : fill.qty * fill.price;
      if (isHkTicker(fill.ticker)) {
        hkCash += flow;
      } else {
        usCash += flow;
      }
    }
  }

  if (snap) {
    for (const ticker of CORE_TICKERS) {
      if (!touched.has(ticker)) {
        qty[ticker] = Number(snap.holdings?.[ticker] || 0);
      }
    }
    if (real.length === 0) {
      hkCash = Number(snap.cash || hkStart);
    }
    for (const fill of fills) {
      if ((fill.side || "").toUpperCase() !== "PENDING") {
        continue;
      }
      if (touched.has(fill.ticker)) {
        continue;
      }
      const intent = pendingIntent(fill);
      if (intent === "BUY" && (qty[fill.ticker] || 0) >= fill.qty - 1e-9) {
        qty[fill.ticker] = (qty[fill.ticker] || 0) - fill.qty;
      } else if (intent === "SELL") {
        qty[fill.ticker] = (qty[fill.ticker] || 0) + fill.qty;
      }
    }
    for (const row of PHANTOM_CATL_SELLS) {
      const found = raw.find((fill) => String(fill.order_id) === row.order_id);
      const side = (found?.side || "").toUpperCase();
      if (side === "SELL") {
        if (!touched.has(row.ticker)) {
          qty[row.ticker] = (qty[row.ticker] || 0) + row.qty;
        }
        if (real.length === 0) {
          hkCash -= row.qty * row.price;
        }
      }
    }
  }

  const catl = qty["HK.03750"] || 0;
  const phantomMentioned = raw.some((fill) => PHANTOM_IDS.has(String(fill.order_id || "")));
  if (phantomMentioned && catl < 1299) {
    qty["HK.03750"] = catl + PHANTOM_CATL_SELLS.reduce((sum, row) => sum + row.qty, 0);
  }
  if ((qty["HK.03750"] || 0) >= 1299 && hkCash > 200_000) {
    hkCash -= PHANTOM_PROCEEDS;
  }

  const usFromFills = [...touched].some((ticker) => ticker.startsWith("US."));
  if (!usFromFills) {
    const usMtmNow = mtm(US_TICKERS, qty, marks);
    const savedUs = snap?.us_cash;
    if (savedUs != null && Math.abs(savedUs - US_INITIAL_CASH) > 1) {
      usCash = savedUs;
    } else if (US_TICKERS.some((ticker) => Math.abs(qty[ticker] || 0) > 1e-9)) {
      // Pickle short/long with no US fill trail: keep USD equity near the 1M start at current marks.
      usCash = US_INITIAL_CASH - usMtmNow;
    }
  }

  const hkEquity = hkCash + mtm(HK_TICKERS, qty, marks);
  const usEquity = usCash + mtm(US_TICKERS, qty, marks);

  return {
    fills,
    holdings: qty,
    marks,
    hk: {
      market: "HK",
      open: isHkCashOpen(now),
      currency: "HKD",
      cash: hkCash,
      equity: hkEquity,
      pnl: hkEquity - hkStart,
      names: [...HK_TICKERS],
    },
    us: {
      market: "US",
      open: isUsCashOpen(now),
      currency: "USD",
      cash: usCash,
      equity: usEquity,
      pnl: usEquity - US_INITIAL_CASH,
      names: [...US_TICKERS],
    },
  };
}
