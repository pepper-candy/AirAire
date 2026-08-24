"use client";

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { EquityChart } from "@/components/EquityChart";
import { filterPoints, hkYmd, metaFor, shapeSeries } from "@/lib/equityRange";
import { makeTestSnapshot } from "@/lib/testSnapshot";
import { CORE_TICKERS, TICKER_NAMES, type Fill, type Headline, type RangeMode, type SnapshotResponse } from "@/lib/types";

const MOBILE_MQ = "(max-width: 800px)";
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

function equityNote(meta: { rawCount: number; shownCount: number; bucketMinutes: number } | undefined, shown: number): string {
  if (!meta) {
    return shown ? `${shown} session snapshots` : "Waiting for a session snapshot";
  }
  if (meta.shownCount === meta.rawCount) {
    return `${meta.rawCount} session snaps on chart · all of them · ${meta.bucketMinutes}-min`;
  }
  return `${meta.shownCount} on chart of ${meta.rawCount} session snaps · ${meta.bucketMinutes}-min buckets`;
}

function ExpandGlyph({ open }: { open: boolean }) {
  return (
    <svg className="equity-expand" viewBox="0 0 16 16" aria-hidden>
      {open ? (
        <path
          d="M5 2v3H2M11 2v3h3M11 14v-3h3M5 14v-3H2"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeLinecap="square"
        />
      ) : (
        <path
          d="M2 6V2h4M10 2h4v4M14 10v4h-4M6 14H2v-4"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeLinecap="square"
        />
      )}
    </svg>
  );
}

function MiniSpark({ points }: { points: { t: string; equity: number }[] }) {
  if (points.length < 2) {
    return <div className="spark-empty">—</div>;
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
  const [listH, setListH] = useState<number | null>(null);
  const listInnerRef = useRef<HTMLDivElement>(null);
  const collapsedCount = compact ? 1 : 3;
  const pageSize = compact ? 5 : 10;
  const pageCount = Math.max(1, Math.ceil(items.length / pageSize));
  const safePage = Math.min(page, pageCount);
  const visible = open
    ? items.slice((safePage - 1) * pageSize, safePage * pageSize)
    : items.slice(0, Math.min(collapsedCount, items.length));
  const canExpand = items.length > collapsedCount;
  const canToggle = canExpand || open;
  const accent = groupScore(score, members);

  useLayoutEffect(() => {
    const node = listInnerRef.current;
    if (!node) {
      setListH(null);
      return;
    }
    setListH(node.scrollHeight);
  }, [visible, compact, items.length]);

  function renderHeadline(item: Headline, key: string) {
    const href = item.url || undefined;
    const Tag = href ? "a" : "div";
    return (
      <Tag
        key={key}
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
  }

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
          {!open && canToggle ? <span className="headline-expand">Expand</span> : null}
        </div>
      </div>
      {items.length === 0 ? (
        <div className="empty">No cached headlines for this name.</div>
      ) : (
        <div className={listH === null ? "headline-list" : "headline-list is-ready"} style={listH === null ? undefined : { height: listH }}>
          <div className="headline-list-inner" ref={listInnerRef}>
            {visible.map((item, i) => renderHeadline(item, `${name}-${safePage}-${i}-${item.url || item.title}`))}
          </div>
        </div>
      )}
    </div>
  );
}

export function Dashboard({ initial }: { initial: SnapshotResponse }) {
  const [data, setData] = useState(initial);
  const [demo, setDemo] = useState(false);
  const [share, setShare] = useState(0.5);
  const [range, setRange] = useState<RangeMode>(initial.equityMeta?.range || "today");
  const [day, setDay] = useState(initial.equityMeta?.day || hkYmd());
  const [demoSeries, setDemoSeries] = useState(initial.equitySeries);
  const [equityOpen, setEquityOpen] = useState(true);
  const splitRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef(false);
  const compact = useCompact();

  useLayoutEffect(() => {
    if (window.matchMedia(MOBILE_MQ).matches) {
      setEquityOpen(false);
    }
  }, []);

  useEffect(() => {
    if (demo) {
      return;
    }
    const tick = async () => {
      try {
        const res = await fetch("/api/snapshot", { cache: "no-store" });
        if (res.ok) {
          const next = (await res.json()) as SnapshotResponse;
          setData((prev) => ({
            ...prev,
            latest: next.latest,
            stale: next.stale,
            error: next.error,
          }));
        }
      } catch {
        // keep last good frame
      }
    };
    const id = window.setInterval(tick, 20_000);
    return () => window.clearInterval(id);
  }, [demo]);

  useEffect(() => {
    if (demo) {
      const raw = filterPoints(demoSeries, range, day);
      const shaped = shapeSeries(raw, range);
      setData((prev) => ({
        ...prev,
        equitySeries: shaped.shown,
        equityMeta: metaFor(range, day, raw.length, shaped.shown.length, shaped.bucketMinutes),
      }));
      return;
    }
    let cancelled = false;
    const load = async () => {
      try {
        const params = new URLSearchParams({ range });
        if (range === "day") {
          params.set("day", day);
        }
        const res = await fetch(`/api/equity?${params.toString()}`, { cache: "no-store" });
        if (!res.ok) {
          return;
        }
        const next = (await res.json()) as Pick<SnapshotResponse, "equitySeries" | "equityMeta" | "error">;
        if (cancelled) {
          return;
        }
        setData((prev) => ({
          ...prev,
          equitySeries: next.equitySeries || [],
          equityMeta: next.equityMeta,
          error: next.error || prev.error,
        }));
      } catch {
        // keep last series
      }
    };
    void load();
    const poll = range === "today" || range === "week";
    const id = poll ? window.setInterval(load, range === "today" ? 20_000 : 60_000) : 0;
    return () => {
      cancelled = true;
      if (id) {
        window.clearInterval(id);
      }
    };
  }, [demo, demoSeries, range, day]);

  async function toggleDemo() {
    if (demo) {
      setDemo(false);
      try {
        const res = await fetch("/api/snapshot", { cache: "no-store" });
        if (res.ok) {
          const next = (await res.json()) as SnapshotResponse;
          setData((prev) => ({
            ...prev,
            latest: next.latest,
            stale: next.stale,
            error: next.error,
          }));
          return;
        }
      } catch {
        // fall through to the last server frame
      }
      setData((prev) => ({
        ...initial,
        equitySeries: prev.equitySeries,
        equityMeta: prev.equityMeta,
      }));
      return;
    }
    setDemo(true);
    const fake = makeTestSnapshot();
    setDemoSeries(fake.equitySeries);
    setData(fake);
  }

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
        <button
          type="button"
          className={demo ? "test-btn is-on" : "test-btn"}
          onClick={toggleDemo}
          aria-pressed={demo}
        >
          TEST
        </button>
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
            <span className="status-label-full">Last Snapshot</span>
            <span className="status-label-short">Last Snap</span>{" "}
            <strong>{formatHkStamp(snap?.updated_at)}</strong>
          </div>
          <div className="status-meta-row">
            <span>Last Bar</span> <strong>{snap?.last_bar_datetime || "—"}</strong>
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
          <button
            type="button"
            className={equityOpen ? "card equity-mini is-on" : "card equity-mini"}
            onClick={() => setEquityOpen((open) => !open)}
            aria-expanded={equityOpen}
            aria-label={equityOpen ? "Hide equity path" : "Show equity path"}
          >
            <div className="kpi-label">Equity Path</div>
            <div className="equity-mini-spark">
              <MiniSpark points={data.equitySeries} />
            </div>
            <ExpandGlyph open={equityOpen} />
          </button>
          <div className={`card equity-card${equityOpen ? "" : " is-stowed"}`}>
            <div className="equity-head">
              <div>
                <div className="kpi-label">Equity Path</div>
                <div
                  className="kpi-note"
                  title={
                    data.equityMeta && data.equityMeta.shownCount !== data.equityMeta.rawCount
                      ? "On chart = downsampled points drawn. Session snaps = every session row in this range."
                      : "On chart and session snaps match: every snapshot in this range is drawn."
                  }
                >
                  {equityNote(data.equityMeta, data.equitySeries.length)}
                </div>
              </div>
              <div className="equity-controls">
                <button
                  type="button"
                  className={range === "today" ? "range-btn is-on" : "range-btn"}
                  onClick={() => {
                    setRange("today");
                    setDay(hkYmd());
                  }}
                >
                  Today
                </button>
                <button
                  type="button"
                  className={range === "week" ? "range-btn is-on" : "range-btn"}
                  onClick={() => setRange("week")}
                >
                  Week
                </button>
                <label className="range-day">
                  <span>Day</span>
                  <input
                    type="date"
                    value={range === "day" ? day : hkYmd()}
                    max={hkYmd()}
                    onChange={(event) => {
                      const next = event.target.value;
                      if (!next) {
                        return;
                      }
                      setDay(next);
                      setRange("day");
                    }}
                  />
                </label>
              </div>
            </div>
            <EquityChart points={data.equitySeries} />
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
                  <th className="num" title="Last PPO output. −1 = want full short, +1 = want full long. Qty is the book.">
                    Tgt
                  </th>
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
            <div className="pane-fills-body">
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
