import { anyCashSessionOpen, snapshotAt } from "./marketHours";
import { STALE_AFTER_SECONDS, type Snapshot, type SnapshotResponse, type SnapshotRow } from "./types";

const SKIP_EQUITY_KINDS = new Set(["heartbeat", "seed"]);

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

export async function fetchSnapshots(limit = 120): Promise<SnapshotResponse> {
  const url = supabaseUrl();
  const key = anonKey();
  if (!url || !key) {
    return {
      latest: null,
      equitySeries: [],
      stale: true,
      staleAfterSeconds: STALE_AFTER_SECONDS,
      error: "NEXT_PUBLIC_SUPABASE_URL or NEXT_PUBLIC_SUPABASE_ANON_KEY is unset.",
    };
  }

  const endpoint = `${restBase()}/bot_snapshots?select=id,created_at,kind,payload&order=created_at.desc&limit=${limit}`;
  const resp = await fetch(endpoint, {
    headers: {
      apikey: key,
      Authorization: `Bearer ${key}`,
    },
    cache: "no-store",
  });

  if (!resp.ok) {
    const text = await resp.text();
    return {
      latest: null,
      equitySeries: [],
      stale: true,
      staleAfterSeconds: STALE_AFTER_SECONDS,
      error: `Supabase HTTP ${resp.status}: ${text.slice(0, 180)}`,
    };
  }

  const rows = (await resp.json()) as SnapshotRow[];
  const latest = rows[0]?.payload ?? null;
  const equitySeries = [...rows]
    .reverse()
    .filter(isSessionSnapshot)
    .map((row) => ({
      t: row.created_at,
      equity: Number(row.payload?.equity ?? 0),
    }))
    .filter((p) => Number.isFinite(p.equity));

  return {
    latest,
    equitySeries,
    stale: isStale(latest?.updated_at),
    staleAfterSeconds: STALE_AFTER_SECONDS,
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
