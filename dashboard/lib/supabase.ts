import { splitBooks } from "./bookSplit";
import { hkYmd, metaFor, parseDayParam, rangeBounds, shapeBookSeries, shiftHkYmd } from "./equityRange";
import { isHkCashOpen, isUsCashOpen, snapshotAt } from "./marketHours";
import {
  STALE_AFTER_SECONDS,
  type EquityMeta,
  type EquityPoint,
  type RangeMode,
  type Snapshot,
  type SnapshotResponse,
  type SnapshotRow,
} from "./types";

const SKIP_EQUITY_KINDS = new Set(["heartbeat", "seed"]);
const PAGE_SIZE = 1000;

function supabaseUrl(): string {
  return (process.env.NEXT_PUBLIC_SUPABASE_URL || "").trim().replace(/\/$/, "");
}

function anonKey(): string {
  return (process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "").trim();
}

function restBase(): string {
  const url = supabaseUrl();
  if (url.endsWith("/rest/v1")) {
    return url;
  }
  return `${url}/rest/v1`;
}

function isStale(updatedAt: string | undefined): boolean {
  if (!updatedAt) {
    return true;
  }
  const ms = Date.parse(updatedAt);
  if (Number.isNaN(ms)) {
    return true;
  }
  return Date.now() - ms > STALE_AFTER_SECONDS * 1000;
}

function missingEnvError(): string {
  return "NEXT_PUBLIC_SUPABASE_URL or NEXT_PUBLIC_SUPABASE_ANON_KEY is unset.";
}

function emptyLatest(error?: string): SnapshotResponse {
  return {
    latest: null,
    equitySeries: [],
    hkEquitySeries: [],
    usEquitySeries: [],
    stale: true,
    staleAfterSeconds: STALE_AFTER_SECONDS,
    error,
  };
}

async function restGet(pathAndQuery: string): Promise<{ ok: true; rows: SnapshotRow[] } | { ok: false; error: string }> {
  const url = supabaseUrl();
  const key = anonKey();
  if (!url || !key) {
    return { ok: false, error: missingEnvError() };
  }
  const endpoint = `${restBase()}/${pathAndQuery}`;
  const resp = await fetch(endpoint, {
    headers: {
      apikey: key,
      Authorization: `Bearer ${key}`,
    },
    cache: "no-store",
  });
  if (!resp.ok) {
    const text = await resp.text();
    return { ok: false, error: `Supabase HTTP ${resp.status}: ${text.slice(0, 180)}` };
  }
  const rows = (await resp.json()) as SnapshotRow[];
  return { ok: true, rows };
}

function toBookPoints(rows: SnapshotRow[]): { hk: EquityPoint[]; us: EquityPoint[] } {
  const hk: EquityPoint[] = [];
  const us: EquityPoint[] = [];
  for (const row of rows) {
    const kind = String(row.kind || row.payload?.kind || "live").toLowerCase();
    if (SKIP_EQUITY_KINDS.has(kind)) {
      continue;
    }
    const payload = row.payload;
    if (!payload) {
      continue;
    }
    const at = snapshotAt(payload.updated_at) || snapshotAt(row.created_at);
    if (!at) {
      continue;
    }
    const books = splitBooks(payload, at);
    const t = row.created_at;
    if (isHkCashOpen(at) && Number.isFinite(books.hk.equity)) {
      hk.push({ t, equity: books.hk.equity });
    }
    if (isUsCashOpen(at) && Number.isFinite(books.us.equity)) {
      us.push({ t, equity: books.us.equity });
    }
  }
  return { hk, us };
}

export async function fetchLatest(): Promise<SnapshotResponse> {
  const got = await restGet("bot_snapshots?select=id,created_at,kind,payload&order=created_at.desc&limit=1");
  if (!got.ok) {
    return emptyLatest(got.error);
  }
  const latest = got.rows[0]?.payload ?? null;
  return {
    latest,
    equitySeries: [],
    hkEquitySeries: [],
    usEquitySeries: [],
    stale: isStale(latest?.updated_at),
    staleAfterSeconds: STALE_AFTER_SECONDS,
  };
}

type EquitySeriesResult = {
  hkEquitySeries: EquityPoint[];
  usEquitySeries: EquityPoint[];
  equitySeries: EquityPoint[];
  equityMeta: EquityMeta;
  error?: string;
};

async function fetchEquityWindow(mode: RangeMode, day?: string): Promise<EquitySeriesResult> {
  const { start, end, ymd } = rangeBounds(mode, day);
  const startIso = start.toISOString();
  const endIso = end.toISOString();
  const collected: SnapshotRow[] = [];
  let cursor = startIso;
  let useGte = true;

  for (let page = 0; page < 8; page += 1) {
    const fromOp = useGte ? "gte" : "gt";
    const query =
      `bot_snapshots?select=id,created_at,kind,payload` +
      `&created_at=${fromOp}.${encodeURIComponent(cursor)}` +
      `&created_at=lt.${encodeURIComponent(endIso)}` +
      `&order=created_at.asc&limit=${PAGE_SIZE}`;
    const got = await restGet(query);
    if (!got.ok) {
      const shaped = shapeBookSeries([], [], mode);
      return {
        hkEquitySeries: [],
        usEquitySeries: [],
        equitySeries: [],
        equityMeta: {
          ...metaFor(mode, ymd, 0, 0, shaped.bucketMinutes),
          hkRawCount: 0,
          usRawCount: 0,
          hkShownCount: 0,
          usShownCount: 0,
        },
        error: got.error,
      };
    }
    if (got.rows.length === 0) {
      break;
    }
    collected.push(...got.rows);
    if (got.rows.length < PAGE_SIZE) {
      break;
    }
    const last = got.rows[got.rows.length - 1]?.created_at;
    if (!last || last === cursor) {
      break;
    }
    cursor = last;
    useGte = false;
  }

  const raw = toBookPoints(collected);
  const shaped = shapeBookSeries(raw.hk, raw.us, mode);
  return {
    hkEquitySeries: shaped.hk,
    usEquitySeries: shaped.us,
    equitySeries: shaped.hk,
    equityMeta: {
      ...metaFor(mode, ymd, raw.hk.length + raw.us.length, shaped.hk.length + shaped.us.length, shaped.bucketMinutes),
      hkRawCount: raw.hk.length,
      usRawCount: raw.us.length,
      hkShownCount: shaped.hk.length,
      usShownCount: shaped.us.length,
    },
  };
}

export async function fetchEquitySeries(mode: RangeMode, day?: string): Promise<EquitySeriesResult> {
  const primary = await fetchEquityWindow(mode, day);
  if (primary.error || mode !== "today") {
    return primary;
  }
  const ymd = parseDayParam(day);
  if (ymd !== hkYmd()) {
    return primary;
  }
  let hk = primary.hkEquitySeries;
  let us = primary.usEquitySeries;
  let hkRaw = primary.equityMeta.hkRawCount ?? hk.length;
  let usRaw = primary.equityMeta.usRawCount ?? us.length;
  let hkShown = primary.equityMeta.hkShownCount ?? hk.length;
  let usShown = primary.equityMeta.usShownCount ?? us.length;
  let cursor = ymd;
  for (let step = 0; step < 3 && (hk.length < 2 || us.length < 2); step += 1) {
    cursor = shiftHkYmd(cursor, -1);
    const fallback = await fetchEquityWindow("today", cursor);
    if (fallback.error) {
      break;
    }
    if (hk.length < 2 && fallback.hkEquitySeries.length >= 2) {
      hk = fallback.hkEquitySeries;
      hkRaw = fallback.equityMeta.hkRawCount ?? hk.length;
      hkShown = fallback.equityMeta.hkShownCount ?? hk.length;
    }
    if (us.length < 2 && fallback.usEquitySeries.length >= 2) {
      us = fallback.usEquitySeries;
      usRaw = fallback.equityMeta.usRawCount ?? us.length;
      usShown = fallback.equityMeta.usShownCount ?? us.length;
    }
  }
  const bucketMinutes = primary.equityMeta.bucketMinutes;
  return {
    hkEquitySeries: hk,
    usEquitySeries: us,
    equitySeries: hk,
    equityMeta: {
      ...metaFor(mode, ymd, hkRaw + usRaw, hkShown + usShown, bucketMinutes),
      hkRawCount: hkRaw,
      usRawCount: usRaw,
      hkShownCount: hkShown,
      usShownCount: usShown,
    },
  };
}

export async function fetchSnapshots(mode: RangeMode = "today", day?: string): Promise<SnapshotResponse> {
  const [latest, series] = await Promise.all([fetchLatest(), fetchEquitySeries(mode, day)]);
  return {
    ...latest,
    equitySeries: series.equitySeries,
    hkEquitySeries: series.hkEquitySeries,
    usEquitySeries: series.usEquitySeries,
    equityMeta: series.equityMeta,
    error: latest.error || series.error,
  };
}

export function emptySnapshot(): Snapshot {
  return {
    kind: "live",
    updated_at: "",
    cash: 0,
    equity: 0,
    holdings: {},
    last_action: {},
    last_reason: "",
    last_bar_datetime: "",
    news_scores: {},
    headlines: {},
    headline_baskets: [],
    fills: [],
    initial_cash: 1_000_000,
    pnl: 0,
  };
}
