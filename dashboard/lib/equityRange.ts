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
  if (raw === "week") {
    return "week";
  }
  return "today";
}

export function parseDayParam(raw: string | null | undefined): string {
  if (raw && /^\d{4}-\d{2}-\d{2}$/.test(raw)) {
    return raw;
  }
  return hkYmd();
}

export function shiftHkYmd(ymd: string, days: number): string {
  const noon = new Date(`${parseDayParam(ymd)}T12:00:00+08:00`);
  return hkYmd(new Date(noon.getTime() + days * DAY_MS));
}

export function hkDayUtcBounds(ymd: string): { start: Date; end: Date } {
  const start = new Date(`${ymd}T00:00:00+08:00`);
  return { start, end: new Date(start.getTime() + DAY_MS) };
}

function capEnd(end: Date, now: Date = new Date()): Date {
  return end.getTime() > now.getTime() ? now : end;
}

export function rangeBounds(mode: RangeMode, day?: string): { start: Date; end: Date; ymd: string } {
  const ymd = parseDayParam(day);
  const { start, end: dayEnd } = hkDayUtcBounds(ymd);
  if (mode === "week") {
    const weekEnd = new Date(start.getTime() + 7 * DAY_MS);
    return { start, end: capEnd(weekEnd), ymd };
  }
  return { start, end: capEnd(dayEnd), ymd };
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

export function bucketMsFor(mode: RangeMode): number {
  return mode === "week" ? 10 * MINUTE_MS : MINUTE_MS;
}

export function shapeSeries(points: EquityPoint[], mode: RangeMode): { shown: EquityPoint[]; bucketMinutes: number } {
  const bucketMs = bucketMsFor(mode);
  return { shown: downsample(points, bucketMs), bucketMinutes: bucketMs / MINUTE_MS };
}

export function shapeBookSeries(
  hk: EquityPoint[],
  us: EquityPoint[],
  mode: RangeMode,
): { hk: EquityPoint[]; us: EquityPoint[]; bucketMinutes: number } {
  const bucketMs = bucketMsFor(mode);
  return {
    hk: downsample(hk, bucketMs),
    us: downsample(us, bucketMs),
    bucketMinutes: bucketMs / MINUTE_MS,
  };
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
