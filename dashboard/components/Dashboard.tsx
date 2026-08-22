"use client";

import { useEffect, useState } from "react";
import { CORE_TICKERS, TICKER_NAMES, type SnapshotResponse } from "@/lib/types";

function money(n: number): string {
  return new Intl.NumberFormat("en-HK", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(n);
}

function shares(n: number): string {
  return new Intl.NumberFormat("en-HK", { maximumFractionDigits: 4 }).format(n);
}

function hkTime(value: string | undefined): string {
  if (!value) {
    return "—";
  }
  const ms = Date.parse(value);
  if (Number.isNaN(ms)) {
    return value;
  }
  return new Intl.DateTimeFormat("en-HK", {
    timeZone: "Asia/Hong_Kong",
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(new Date(ms));
}

function scoreClass(n: number): string {
  if (n > 0.05) {
    return "pos";
  }
  if (n < -0.05) {
    return "neg";
  }
  return "";
}

function Sparkline({ points }: { points: { t: string; equity: number }[] }) {
  if (points.length < 2) {
    return <div className="empty">Sparkline waits for a few snapshots.</div>;
  }
  const xs = points.map((p) => p.equity);
  const min = Math.min(...xs);
  const max = Math.max(...xs);
  const span = max - min || 1;
  const d = points
    .map((p, i) => {
      const x = (i / (points.length - 1)) * 100;
      const y = 28 - ((p.equity - min) / span) * 24;
      return `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
  return (
    <svg className="spark" viewBox="0 0 100 32" preserveAspectRatio="none" aria-hidden>
      <path d={d} fill="none" stroke="#b7c4a1" strokeWidth="1.2" />
    </svg>
  );
}

export function Dashboard({ initial }: { initial: SnapshotResponse }) {
  const [data, setData] = useState(initial);

  useEffect(() => {
    const tick = async () => {
      try {
        const res = await fetch("/api/snapshot", { cache: "no-store" });
        if (res.ok) {
          setData((await res.json()) as SnapshotResponse);
        }
      } catch {
        // keep last good frame
      }
    };
    const id = window.setInterval(tick, 20_000);
    return () => window.clearInterval(id);
  }, []);

  const snap = data.latest;
  const stale = data.stale || !snap;
  const pnl = snap?.pnl ?? 0;

  return (
    <div className="shell">
      <div className={`banner ${stale ? "stale" : "live"}`}>
        {stale
          ? `VM idle or unreachable. Showing last snapshot. Freshness window is ${data.staleAfterSeconds}s.`
          : "VM snapshot is fresh. This page does not trade."}
      </div>

      <header className="top">
        <div>
          <div className="eyebrow">AirAire · paper book</div>
          <h1>Read-only blotter</h1>
        </div>
        <div className="meta">
          <div>Last snapshot {hkTime(snap?.updated_at)}</div>
          <div>Last bar {snap?.last_bar_datetime || "—"}</div>
        </div>
      </header>

      {data.error ? <p className="err">{data.error}</p> : null}

      <section className="kpis">
        <div className="card">
          <div className="kpi-label">Equity</div>
          <div className="kpi-value">{money(snap?.equity ?? 0)}</div>
        </div>
        <div className="card">
          <div className="kpi-label">Cash</div>
          <div className="kpi-value">{money(snap?.cash ?? 0)}</div>
        </div>
        <div className="card">
          <div className="kpi-label">P&amp;L vs start</div>
          <div className={`kpi-value ${scoreClass(pnl)}`}>
            {pnl >= 0 ? "+" : ""}
            {money(pnl)}
          </div>
        </div>
        <div className="card">
          <div className="kpi-label">Equity path</div>
          <Sparkline points={data.equitySeries} />
        </div>
      </section>

      <div className="grid">
        <section className="block">
          <h2>Book</h2>
          <p className="reason">{snap?.last_reason || "No reason yet."}</p>
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th className="num">Holdings</th>
                <th className="num">Action</th>
                <th className="num">News</th>
              </tr>
            </thead>
            <tbody>
              {CORE_TICKERS.map((ticker) => (
                <tr key={ticker}>
                  <td>
                    {TICKER_NAMES[ticker] || ticker}
                    <div className="tiny">{ticker}</div>
                  </td>
                  <td className="num">{shares(snap?.holdings?.[ticker] ?? 0)}</td>
                  <td className={`num ${scoreClass(snap?.last_action?.[ticker] ?? 0)}`}>
                    {(snap?.last_action?.[ticker] ?? 0).toFixed(3)}
                  </td>
                  <td className={`num ${scoreClass(snap?.news_scores?.[ticker] ?? 0)}`}>
                    {(snap?.news_scores?.[ticker] ?? 0).toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="block">
          <h2>Fills</h2>
          {(snap?.fills || []).length === 0 ? (
            <div className="empty">No SIMULATE fills in the local blotter yet.</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>When</th>
                  <th>Fill</th>
                </tr>
              </thead>
              <tbody>
                {[...(snap?.fills || [])].reverse().map((fill, i) => (
                  <tr key={`${fill.order_id}-${i}`}>
                    <td className="tiny">{hkTime(fill.time)}</td>
                    <td>
                      {fill.side} {fill.qty} {TICKER_NAMES[fill.ticker] || fill.ticker} @ {money(fill.price)}
                      <div className="tiny">{fill.reason}</div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </div>

      <section className="block" style={{ marginTop: 12 }}>
        <h2>Headlines the model just scored</h2>
        {CORE_TICKERS.map((ticker) => {
          const items = snap?.headlines?.[ticker] || [];
          return (
            <div key={ticker} style={{ marginBottom: 16 }}>
              <div className="eyebrow">
                {TICKER_NAMES[ticker]} · {(snap?.news_scores?.[ticker] ?? 0).toFixed(3)}
              </div>
              {items.length === 0 ? (
                <div className="empty">No cached headlines for this name.</div>
              ) : (
                items.slice(0, 8).map((item, i) => (
                  <a
                    key={`${ticker}-${i}-${item.url}`}
                    className="headline"
                    href={item.url || undefined}
                    target="_blank"
                    rel="noreferrer"
                  >
                    <span className="title">{item.title || "(untitled)"}</span>
                    <span className={`tiny ${scoreClass(item.sentiment_score)}`}>
                      {item.source || "source?"} · {item.time_published || "—"} · {item.sentiment_score.toFixed(3)}
                    </span>
                  </a>
                ))
              )}
            </div>
          );
        })}
      </section>
    </div>
  );
}
