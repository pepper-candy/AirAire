import { metaFor, rangeBounds, shapeSeries } from "./equityRange";
import { anyCashSessionOpen, snapshotAt } from "./marketHours";
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

function isSessionSnapshot(row: SnapshotRow): boolean {
  const kind = String(row.kind || row.payload?.kind || "live").toLowerCase();
  if (SKIP_EQUITY_KINDS.has(kind)) {
    return false;
  }
  const at = snapshotAt(row.payload?.updated_at) || snapshotAt(row.created_at);
  if (!at) {
    return false;
  }
  return anyCashSessionOpen(at);
}

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

function toPoints(rows: SnapshotRow[]): EquityPoint[] {
  return rows
    .filter(isSessionSnapshot)
    .map((row) => ({
      t: row.created_at,
      equity: Number(row.payload?.equity ?? 0),
    }))
    .filter((point) => Number.isFinite(point.equity));
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
    stale: isStale(latest?.updated_at),
    staleAfterSeconds: STALE_AFTER_SECONDS,
  };
}

export async function fetchEquitySeries(mode: RangeMode, day?: string): Promise<{
  equitySeries: EquityPoint[];
  equityMeta: EquityMeta;
  error?: string;
}> {
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
      const shaped = shapeSeries([], mode);
      return { equitySeries: shaped.shown, equityMeta: metaFor(mode, ymd, 0, 0, shaped.bucketMinutes), error: got.error };
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

  const raw = toPoints(collected);
  const shaped = shapeSeries(raw, mode);
  return {
    equitySeries: shaped.shown,
    equityMeta: metaFor(mode, ymd, raw.length, shaped.shown.length, shaped.bucketMinutes),
  };
}

export async function fetchSnapshots(mode: RangeMode = "today", day?: string): Promise<SnapshotResponse> {
  const [latest, series] = await Promise.all([fetchLatest(), fetchEquitySeries(mode, day)]);
  return {
    ...latest,
    equitySeries: series.equitySeries,
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
