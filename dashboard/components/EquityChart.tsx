"use client";

import { useMemo, useState, type PointerEvent } from "react";
import type { EquityPoint } from "@/lib/types";

function money(n: number): string {
  return new Intl.NumberFormat("en-HK", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(n);
}

function stamp(value: string): string {
  const ms = Date.parse(value);
  if (Number.isNaN(ms)) {
    return value;
  }
  return new Intl.DateTimeFormat("en-GB", {
    timeZone: "Asia/Hong_Kong",
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(ms));
}

type Props = {
  points: EquityPoint[];
  currency: "HKD" | "USD";
  stroke?: string;
};

export function EquityChart({ points, currency, stroke = "#8fbf9f" }: Props) {
  const [hover, setHover] = useState<number | null>(null);

  const geom = useMemo(() => {
    if (points.length < 2) {
      return null;
    }
    const xs = points.map((p) => p.equity);
    const min = Math.min(...xs);
    const max = Math.max(...xs);
    const span = max - min || 1;
    const coords = points.map((p, i) => {
      const x = (i / (points.length - 1)) * 100;
      const y = 38 - ((p.equity - min) / span) * 32;
      return { x, y, p };
    });
    const d = coords.map((c, i) => `${i === 0 ? "M" : "L"} ${c.x.toFixed(2)} ${c.y.toFixed(2)}`).join(" ");
    return { min, max, coords, d };
  }, [points]);

  if (!geom) {
    return <div className="equity-empty">Waiting for two {currency} session snapshots</div>;
  }

  const active = hover !== null ? geom.coords[hover] : null;

  function indexFromEvent(event: PointerEvent<HTMLDivElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width <= 0) {
      return;
    }
    const x = ((event.clientX - rect.left) / rect.width) * 100;
    let best = 0;
    let bestDist = Infinity;
    geom!.coords.forEach((c, i) => {
      const dist = Math.abs(c.x - x);
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    });
    setHover(best);
  }

  return (
    <div
      className="equity-chart"
      onPointerMove={indexFromEvent}
      onPointerDown={indexFromEvent}
      onPointerLeave={() => setHover(null)}
    >
      <svg className="equity-svg" viewBox="0 0 100 42" preserveAspectRatio="none" aria-hidden>
        <path d={geom.d} fill="none" stroke={stroke} strokeWidth="1.1" />
        {active ? (
          <>
            <line x1={active.x} x2={active.x} y1="4" y2="40" stroke="#9aa3b2" strokeWidth="0.35" />
            <circle cx={active.x} cy={active.y} r="1.3" fill={stroke} />
          </>
        ) : null}
      </svg>
      <div className="equity-readout">
        {active ? (
          <>
            <span>{stamp(active.p.t)} HKT</span>
            <strong>
              {money(active.p.equity)} {currency}
            </strong>
          </>
        ) : (
          <span>Choose a point</span>
        )}
      </div>
    </div>
  );
}
