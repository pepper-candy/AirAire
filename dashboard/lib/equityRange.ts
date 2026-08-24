import type { EquityMeta, EquityPoint, RangeMode } from "@/lib/types";

const DAY_MS = 24 * 60 * 60 * 1000;
const MINUTE_MS = 60_000;

export function hkYmd(at: Date = new Date()): string {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Hong_Kong",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(at);
}

export function parseRangeMode(raw: string | null | undefined): RangeMode {
  if (raw === "week" || raw === "day" || raw === "today") {
    return raw;
  }
  return "today";
}

export function parseDayParam(raw: string | null | undefined): string {
  if (raw && /^\d{4}-\d{2}-\d{2}$/.test(raw)) {
    return raw;
  }
  return hkYmd();
}

export function hkDayUtcBounds(ymd: string): { start: Date; end: Date } {
  const start = new Date(`${ymd}T00:00:00+08:00`);
  return { start, end: new Date(start.getTime() + DAY_MS) };
}

export function rangeBounds(mode: RangeMode, day?: string): { start: Date; end: Date; ymd: string } {
  const today = hkYmd();
  if (mode === "day") {
    const ymd = parseDayParam(day);
    const { start, end } = hkDayUtcBounds(ymd);
    return { start, end, ymd };
  }
  if (mode === "week") {
    const { start: todayStart } = hkDayUtcBounds(today);
    return { start: new Date(todayStart.getTime() - 6 * DAY_MS), end: new Date(), ymd: today };
  }
  const { start } = hkDayUtcBounds(today);
  return { start, end: new Date(), ymd: today };
}

export function downsample(points: EquityPoint[], bucketMs: number): EquityPoint[] {
  if (points.length <= 1 || bucketMs <= 0) {
    return points;
  }
  const buckets = new Map<number, EquityPoint>();
  for (const point of points) {
    const ms = Date.parse(point.t);
    if (Number.isNaN(ms)) {
      continue;
    }
    buckets.set(Math.floor(ms / bucketMs) * bucketMs, point);
  }
  return [...buckets.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, point]) => point);
}

export function shapeSeries(points: EquityPoint[], mode: RangeMode): { shown: EquityPoint[]; bucketMinutes: number } {
  const raw = points.length;
  let bucketMs = 10 * MINUTE_MS;
  if (mode === "today" && raw <= 400) {
    bucketMs = MINUTE_MS;
  }
  const shown = downsample(points, bucketMs);
  return { shown, bucketMinutes: bucketMs / MINUTE_MS };
}

export function metaFor(mode: RangeMode, ymd: string, rawCount: number, shownCount: number, bucketMinutes: number): EquityMeta {
  return { range: mode, day: ymd, rawCount, shownCount, bucketMinutes };
}

export function filterPoints(points: EquityPoint[], mode: RangeMode, day?: string): EquityPoint[] {
  const { start, end } = rangeBounds(mode, day);
  const lo = start.getTime();
  const hi = end.getTime();
  return points.filter((point) => {
    const ms = Date.parse(point.t);
    return Number.isFinite(ms) && ms >= lo && ms < hi;
  });
}
