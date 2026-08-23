/** Same cash windows as src/utils.py — used to drop closed-market heartbeats. */

function clockInZone(at: Date, timeZone: string): { weekday: number; minutes: number } | null {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(at);
  const weekdayName = parts.find((p) => p.type === "weekday")?.value;
  const hour = Number(parts.find((p) => p.type === "hour")?.value);
  const minute = Number(parts.find((p) => p.type === "minute")?.value);
  const weekdayMap: Record<string, number> = {
    Sun: 0,
    Mon: 1,
    Tue: 2,
    Wed: 3,
    Thu: 4,
    Fri: 5,
    Sat: 6,
  };
  const weekday = weekdayName ? weekdayMap[weekdayName] : undefined;
  if (weekday === undefined || !Number.isFinite(hour) || !Number.isFinite(minute)) {
    return null;
  }
  return { weekday, minutes: hour * 60 + minute };
}

function isWeekday(weekday: number): boolean {
  return weekday >= 1 && weekday <= 5;
}

export function isHkCashOpen(at: Date): boolean {
  const local = clockInZone(at, "Asia/Hong_Kong");
  if (!local || !isWeekday(local.weekday)) {
    return false;
  }
  return (local.minutes >= 9 * 60 + 30 && local.minutes < 12 * 60) || (local.minutes >= 13 * 60 && local.minutes < 16 * 60);
}

export function isUsCashOpen(at: Date): boolean {
  const local = clockInZone(at, "America/New_York");
  if (!local || !isWeekday(local.weekday)) {
    return false;
  }
  return local.minutes >= 9 * 60 + 30 && local.minutes < 16 * 60;
}

export function anyCashSessionOpen(at: Date): boolean {
  return isHkCashOpen(at) || isUsCashOpen(at);
}

export function snapshotAt(iso: string | undefined): Date | null {
  if (!iso) {
    return null;
  }
  const ms = Date.parse(iso);
  if (Number.isNaN(ms)) {
    return null;
  }
  return new Date(ms);
}
