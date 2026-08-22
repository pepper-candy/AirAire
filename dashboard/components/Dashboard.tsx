"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { CORE_TICKERS, TICKER_NAMES, type Fill, type Headline, type SnapshotResponse } from "@/lib/types";

/** Keep in sync with `@media (max-width: 640px)` in dashboard/app/globals.css */
const COMPACT_MAX_PX = 640;
const MOBILE_MQ = `(max-width: ${COMPACT_MAX_PX}px)`;
const HANDLE_PX = 16;
const MIN_PANE = 200;

function money(n: number): string {
  return new Intl.NumberFormat("en-HK", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(n);
}

function qty(n: number): string {
  return new Intl.NumberFormat("en-HK", { maximumFractionDigits: 4 }).format(n);
}

function signed(n: number, digits = 3): string {
  const body = Math.abs(n).toFixed(digits);
  if (n > 0) {
    return `+${body}`;
  }
  if (n < 0) {
    return `-${body}`;
  }
  return Number(0).toFixed(digits);
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

function toneClass(n: number): "tone-pos" | "tone-neg" | "tone-neu" {
  if (n > 0.05) {
    return "tone-pos";
  }
  if (n < -0.05) {
    return "tone-neg";
  }
  return "tone-neu";
}

function groupScore(score?: number, members?: { name: string; score: number }[]): number {
  if (typeof score === "number") {
    return score;
  }
  if (members && members.length > 0) {
    return members.reduce((sum, member) => sum + member.score, 0) / members.length;
  }
  return 0;
}

function formatHkStamp(value: string | undefined): string {
  if (!value) {
    return "—";
  }
  const ms = Date.parse(value);
  if (Number.isNaN(ms)) {
    return value;
  }
  const text = new Intl.DateTimeFormat("en-GB", {
    timeZone: "Asia/Hong_Kong",
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(new Date(ms));
  return `${text} HKT`;
}

function hkDay(value: string): string {
  const ms = Date.parse(value);
  if (Number.isNaN(ms)) {
    return "—";
  }
  return new Intl.DateTimeFormat("en-GB", {
    timeZone: "Asia/Hong_Kong",
    day: "2-digit",
    month: "short",
  }).format(new Date(ms));
}

function hkClock(value: string): string {
  const ms = Date.parse(value);
  if (Number.isNaN(ms)) {
    return value;
  }
  return new Intl.DateTimeFormat("en-GB", {
    timeZone: "Asia/Hong_Kong",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(new Date(ms));
}

function useCompact(): boolean {
  const [compact, setCompact] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia(MOBILE_MQ);
    const sync = () => setCompact(mq.matches);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);
  return compact;
}

function Sparkline({ points }: { points: { t: string; equity: number }[] }) {
  if (points.length < 2) {
    return <div className="spark-empty">Waiting for snapshots</div>;
  }
  const xs = points.map((p) => p.equity);
  const min = Math.min(...xs);
  const max = Math.max(...xs);
  const span = max - min || 1;
  const d = points
    .map((p, i) => {
      const x = (i / (points.length - 1)) * 100;
      const y = 32 - ((p.equity - min) / span) * 26;
      return `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
  return (
    <svg className="spark" viewBox="0 0 100 36" preserveAspectRatio="none" aria-hidden>
      <path d={d} fill="none" stroke="#8fbf9f" strokeWidth="1.4" />
    </svg>
  );
}

function groupFills(fills: Fill[]): { day: string; rows: Fill[] }[] {
  const sorted = [...fills].sort((a, b) => Date.parse(b.time) - Date.parse(a.time) || 0);
  const groups: { day: string; rows: Fill[] }[] = [];
  for (const fill of sorted) {
    const day = hkDay(fill.time);
    const last = groups[groups.length - 1];
    if (last && last.day === day) {
      last.rows.push(fill);
    } else {
      groups.push({ day, rows: [fill] });
    }
  }
  return groups;
}

function HeadlineGroup({
  name,
  score,
  items,
  compact,
  members,
}: {
  name: string;
  score?: number;
  items: Headline[];
  compact: boolean;
  members?: { name: string; score: number }[];
}) {
  const [open, setOpen] = useState(false);
  const [page, setPage] = useState(1);
  const collapsedCount = compact ? 1 : 3;
  const pageSize = compact ? 5 : 10;
  const pageCount = Math.max(1, Math.ceil(items.length / pageSize));
  const safePage = Math.min(page, pageCount);
  const visible = open
    ? items.slice((safePage - 1) * pageSize, safePage * pageSize)
    : items.slice(0, collapsedCount);
  const canExpand = items.length > collapsedCount;
  const canToggle = canExpand || open;
  const accent = groupScore(score, members);

  function toggle() {
    if (open) {
      setOpen(false);
      setPage(1);
      return;
    }
    if (canExpand) {
      setOpen(true);
    }
  }

  return (
    <div className={`headline-group ${toneClass(accent)}`}>
      <div className={canToggle ? "headline-bar is-clickable" : "headline-bar"}>
        {canToggle ? (
          <button
            type="button"
            className="headline-toggle"
            aria-expanded={open}
            aria-label={`${open ? "Collapse" : "Expand"} ${name} headlines`}
            onClick={toggle}
          />
        ) : null}
        <div className="headline-stock-col">
          <div className="headline-stock">
            <span className="headline-stock-name">{name}</span>
            {members || score === undefined ? null : (
              <span className={`headline-stock-score ${scoreClass(score)}`}>{signed(score)}</span>
            )}
          </div>
          {members && members.length > 0 ? (
            <p className="headline-note">
              {members.map((member, index) => (
                <span key={member.name}>
                  {index > 0 ? " • " : null}
                  {member.name}{" "}
                  <span className={scoreClass(member.score)}>{signed(member.score)}</span>
                </span>
              ))}
            </p>
          ) : null}
        </div>
        <div className="headline-tools">
          {open && pageCount > 1 ? (
            <div className="headline-pager">
              <button
                type="button"
                className="page-btn"
                disabled={safePage <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                aria-label="Previous headlines"
              >
                ‹
              </button>
              <span className="page-label">
                {safePage}/{pageCount}
              </span>
              <button
                type="button"
                className="page-btn"
                disabled={safePage >= pageCount}
                onClick={() => setPage((p) => Math.min(pageCount, p + 1))}
                aria-label="Next headlines"
              >
                ›
              </button>
            </div>
          ) : null}
          {canToggle ? (
            <span className="icon-fallback" aria-hidden>
              {open ? "▴" : "▾"}
            </span>
          ) : null}
        </div>
      </div>
      {items.length === 0 ? (
        <div className="empty">No cached headlines for this name.</div>
      ) : (
        visible.map((item, i) => {
          const href = item.url || undefined;
          const Tag = href ? "a" : "div";
          return (
            <Tag
              key={`${name}-${i}-${item.url || item.title}`}
              className="headline"
              href={href}
              target={href ? "_blank" : undefined}
              rel={href ? "noreferrer" : undefined}
            >
              <span className="title">{item.title || "(untitled)"}</span>
              <span className="headline-meta">
                {item.source || "source?"} · {item.time_published || "—"}{" "}
                <span className={scoreClass(item.sentiment_score)}>{signed(item.sentiment_score)}</span>
              </span>
            </Tag>
          );
        })
      )}
    </div>
  );
}

export function Dashboard({ initial }: { initial: SnapshotResponse }) {
  const [data, setData] = useState(initial);
  const [share, setShare] = useState(0.5);
  const splitRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef(false);
  const compact = useCompact();

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

  useEffect(() => {
    const onMove = (event: PointerEvent) => {
      if (!dragRef.current || !splitRef.current) {
        return;
      }
      const rect = splitRef.current.getBoundingClientRect();
      const usable = rect.width - HANDLE_PX;
      if (usable <= 0) {
        return;
      }
      const x = event.clientX - rect.left;
      const book = Math.min(usable - MIN_PANE, Math.max(MIN_PANE, x));
      setShare(book / usable);
    };
    const onUp = () => {
      dragRef.current = false;
      document.body.classList.remove("dragging");
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      document.body.classList.remove("dragging");
    };
  }, []);

  const snap = data.latest;
  const stale = data.stale || !snap;
  const pnl = snap?.pnl ?? 0;
  const startCash = snap?.initial_cash ?? 0;
  const pnlPct = startCash ? (pnl / startCash) * 100 : 0;
  const fills = snap?.fills || [];
  const fillGroups = useMemo(() => groupFills(fills), [fills]);

  return (
    <div className="app">
      <header className="brand-bar">
        <div className="brand">
          <span className="brand-mark">AIRAIRE</span>
          <span className="brand-title">Paper Book</span>
        </div>
      </header>

      <div className={`status-strip ${stale ? "stale" : "live"}`}>
        <div className="status-left">
          <span className="status-pip" aria-hidden />
          <p className="status-copy">
            {stale ? "VM idle or unreachable. Showing last snapshot." : "VM snapshot is fresh."}
          </p>
        </div>
        <div className="status-meta">
          <div className="status-meta-row">
            <span>Last snapshot</span> <strong>{formatHkStamp(snap?.updated_at)}</strong>
          </div>
          <div className="status-meta-row">
            <span>Last bar</span> <strong>{snap?.last_bar_datetime || "—"}</strong>
          </div>
        </div>
      </div>

      {data.error ? <p className="page-error">{data.error}</p> : null}

      <div className="page-body">
        <section className="kpis">
          <div className="card">
            <div className="kpi-label">Equity</div>
            <div className="kpi-value">{snap ? money(snap.equity) : "—"}</div>
            <div className="kpi-note">HKD</div>
          </div>
          <div className="card">
            <div className="kpi-label">Cash</div>
            <div className="kpi-value">{snap ? money(snap.cash) : "—"}</div>
            <div className="kpi-note">HKD</div>
          </div>
          <div className="card">
            <div className="kpi-label">P&amp;L vs Start</div>
            <div className={`kpi-value ${snap ? (pnl >= 0 ? "pos" : "neg") : ""}`}>
              {snap ? `${pnl >= 0 ? "+" : ""}${money(pnl)}` : "—"}
            </div>
            <div className="kpi-note">
              {snap && startCash ? `${signed(pnlPct, 2)}% since start` : "—"}
            </div>
          </div>
          <div className="card">
            <div className="kpi-label">Equity Path</div>
            <Sparkline points={data.equitySeries} />
            <div className="kpi-note">
              {data.equitySeries.length ? `${data.equitySeries.length} snapshots` : "Waiting for snapshots"}
            </div>
          </div>
        </section>

        <div
          className="split"
          ref={splitRef}
          style={{ ["--book-share" as string]: String(share) }}
        >
          <section className="pane pane-book" aria-label="Book">
            <table className="book-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th className="num">Qty</th>
                  <th className="num">Act</th>
                  <th className="num">News</th>
                </tr>
              </thead>
              <tbody>
                {CORE_TICKERS.map((ticker) => {
                  const action = snap?.last_action?.[ticker] ?? 0;
                  const news = snap?.news_scores?.[ticker] ?? 0;
                  return (
                    <tr key={ticker}>
                      <td>
                        <div className="book-name">{TICKER_NAMES[ticker] || ticker}</div>
                        <div className="book-code">{ticker}</div>
                      </td>
                      <td className="num">{qty(snap?.holdings?.[ticker] ?? 0)}</td>
                      <td className={`num ${scoreClass(action)}`}>{signed(action)}</td>
                      <td className={`num ${scoreClass(news)}`}>{signed(news)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </section>

          <button
            type="button"
            className="split-handle"
            aria-label="Resize book and fills"
            onPointerDown={(event) => {
              event.preventDefault();
              dragRef.current = true;
              document.body.classList.add("dragging");
              event.currentTarget.setPointerCapture(event.pointerId);
            }}
          >
            <span />
          </button>

          <section className="pane pane-fills" aria-label="Fills">
            <div className="pane-head">
              <h2>Fills</h2>
            </div>
            <div className="fills-scroll">
              {fills.length === 0 ? (
                <div className="empty">No SIMULATE fills yet.</div>
              ) : (
                fillGroups.map((group) => (
                  <div key={group.day}>
                    <div className="fill-day">{group.day}</div>
                    {group.rows.map((fill, i) => {
                      const side = (fill.side || "").toUpperCase();
                      const sideClass = side === "SELL" ? "sell" : side === "BUY" ? "buy" : "";
                      return (
                        <div key={`${fill.order_id}-${fill.time}-${i}`} className="fill-row">
                          <div className="fill-main">
                            <span className="fill-time">{hkClock(fill.time)}</span>
                            <span className={`fill-side ${sideClass}`}>{side || "—"}</span>
                            <span className="fill-name">
                              {qty(fill.qty)} {TICKER_NAMES[fill.ticker] || fill.ticker}
                            </span>
                            <span className="fill-px">@ {money(fill.price)}</span>
                          </div>
                          {fill.reason ? <p className="fill-reason">{fill.reason}</p> : null}
                        </div>
                      );
                    })}
                  </div>
                ))
              )}
            </div>
          </section>
        </div>

        <section className="headlines">
          <h2 className="headlines-title">Headlines · Model Scored</h2>
          {(snap?.headline_baskets || []).map((basket) => (
            <HeadlineGroup
              key={basket.id}
              name={basket.title}
              items={basket.headlines || []}
              compact={compact}
              members={(basket.members || []).map((ticker) => ({
                name: TICKER_NAMES[ticker] || ticker,
                score: snap?.news_scores?.[ticker] ?? 0,
              }))}
            />
          ))}
          {CORE_TICKERS.filter(
            (ticker) => !(snap?.headline_baskets || []).some((basket) => (basket.members || []).includes(ticker)),
          ).map((ticker) => (
            <HeadlineGroup
              key={ticker}
              name={TICKER_NAMES[ticker] || ticker}
              score={snap?.news_scores?.[ticker] ?? 0}
              items={snap?.headlines?.[ticker] || []}
              compact={compact}
            />
          ))}
        </section>
      </div>
    </div>
  );
}
