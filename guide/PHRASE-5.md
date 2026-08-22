# Phase 5 — 24/7 read-only dashboard

**Date:** 2026-08-22
**Status:** Source of truth for the friend-facing site. Phase 4 still wins for paper trading.

If any other file disagrees with this page about the dashboard, **this page wins**. Trading, Promote, goldens, and bats stay in [`PHRASE-4-EXECUTION-&-DAILY-WORK.md`](PHRASE-4-EXECUTION-&-DAILY-WORK.md).

---

## New-agent prompt (copy this first)

> Read `guide/PHRASE-5.md` and `guide/PHRASE-4-EXECUTION-&-DAILY-WORK.md`. Do not retrain, do not change Promote, do not add `--dry-run` to `run_trader.bat`. Do not put `ALPHAVANTAGE_API_KEY`, Futu credentials, Telegram tokens, or the Supabase **service role** on Vercel. The site is read-only. The GPU VM is the only writer.

---

## 0. What Phase 5 is

A **24/7 read-only** Next.js site on Vercel. The university GPU VM keeps doing all trading. When the VM is up, `src/inference.py` POSTs a snapshot (book, news scores, headlines, fills, P&L) to Supabase. The site only SELECTs.

University VDI idle-shutdown is **accepted**. No mouse-jiggler. Catch-up already exists on the next login. The site shows last snapshot time and a stale banner when the VM is dead.

Streamlit-on-VM is rejected (dies with VDI). The site must **not** call Alpha Vantage (keys, 75/min, scores would disagree with the bot).

```text
GPU VM (when up)                 Supabase                 Vercel (24/7)
run_trader.bat
  inference.py  --HTTPS POST-->  bot_snapshots  --SELECT-->  dashboard/
  trades.jsonl (local blotter)
```

---

## 1. Frozen conclusions

- Trading stays on the VM (`run_trader.bat` + Futu OpenD + `src/inference.py`). Vercel never loads `best_model.zip`, never calls Futu, never places orders.
- One-way data. Vercel never writes snapshots.
- Humans need headlines; the model needs a number. `latest_ticker_news()` keeps titles from the same Alpha Vantage call that produces the score. Telegram still only gets fill reasons.
- Push is fail-open: if Supabase is down, the trader logs a warning and keeps trading.
- `--dry-run` / `test_inference.bat` / `--predict-now` do **not** push.
- Do not retrain from Window 90. Do not overwrite goldens.

---

## 2. Env names

**VM `.env` only (writer):**

```text
DASHBOARD_PUSH_URL=https://YOUR_PROJECT.supabase.co
DASHBOARD_PUSH_KEY=          # service_role — never commit, never put on Vercel
DASHBOARD_SNAPSHOTS_TABLE=bot_snapshots
```

**Vercel / `dashboard/.env.local` (reader + site password):**

```text
NEXT_PUBLIC_SUPABASE_URL=https://YOUR_PROJECT.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=
DASHBOARD_GATE=              # shared password for you and your friend
```

`DASHBOARD_GATE` is also fine on the VM `.env` so you have one copy. It is **not** used by Python.

---

## 3. SQL (run once in Supabase SQL Editor)

Project setting **Automatically expose new tables** is off. After this SQL, open **Project Settings → Data API** and expose `public.bot_snapshots` (or turn that toggle on). Otherwise the REST ping returns 404.

```sql
create table if not exists public.bot_snapshots (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  kind text not null default 'live',
  payload jsonb not null
);

create index if not exists bot_snapshots_created_at_idx
  on public.bot_snapshots (created_at desc);

alter table public.bot_snapshots enable row level security;

drop policy if exists anon_select_bot_snapshots on public.bot_snapshots;
create policy anon_select_bot_snapshots
  on public.bot_snapshots
  for select
  to anon
  using (true);

grant select on table public.bot_snapshots to anon;
grant select, insert on table public.bot_snapshots to service_role;

notify pgrst, 'reload schema';
```

Anon can SELECT only. There is no INSERT policy for anon. The VM uses the **service role**, which bypasses RLS.

Do not enable Realtime for v1.

---

## 4. Snapshot contract

Each POST body is `{ "kind": "live", "payload": { ... } }`. Always INSERT (history = equity sparkline). Latest row = current book.

Stale if `payload.updated_at` is older than **3 minutes**.

```json
{
  "kind": "live",
  "updated_at": "2026-08-22T12:40:01+08:00",
  "cash": 1000000.0,
  "equity": 1000000.0,
  "holdings": {"HK.00700": 0, "HK.03690": 0, "HK.03750": 0, "US.COST": 0, "US.KO": 0},
  "last_action": {"HK.00700": 0, "HK.03690": 0, "HK.03750": 0, "US.COST": 0, "US.KO": 0},
  "last_reason": "cold start",
  "last_bar_datetime": "",
  "news_scores": {"HK.00700": 0, "HK.03690": 0, "HK.03750": 0, "US.COST": 0, "US.KO": 0},
  "headlines": {
    "HK.00700": [
      {
        "title": "...",
        "source": "...",
        "url": "https://...",
        "time_published": "20260822T043000",
        "sentiment_score": 0.12
      }
    ]
  },
  "fills": [
    {
      "time": "2026-08-22T10:31:02+08:00",
      "ticker": "HK.00700",
      "side": "BUY",
      "qty": 100,
      "price": 412.2,
      "reason": "Action: Buy 100 Tencent. Reason: ...",
      "order_id": "..."
    }
  ],
  "initial_cash": 1000000.0,
  "pnl": 0.0
}
```

Cadence: every live 60s cycle (`--poll-seconds 60`), including a heartbeat while both cash sessions are closed (so lunch does not look like VM death), and immediately after a SIMULATE fill. Local blotter: `data/logs/trades.jsonl` (survives a failed push).

---

## 5. Code map

**This repo (Python, already wired):**

| File | Role |
|---|---|
| `src/news_loader.py` | `latest_ticker_news()` returns score + titles; `latest_ticker_score()` is a wrapper |
| `src/inference.py` | `NewsPoller` caches headlines; `place_order` returns `(ok, order_id)`; appends jsonl; pushes |
| `src/dashboard_push.py` | Build + POST. Never raises. `python -m src.dashboard_push --ping` |

**`dashboard/` (Next.js App Router):**

- Middleware cookie/query `DASHBOARD_GATE` (or `/login`).
- Server `/api/snapshot` SELECTs latest + last 120 rows for the sparkline (anon key).
- UI: stale banner, book, headlines, blotter, P&L vs `initial_cash`.
- Client poll every 20s.

---

## 6. Human checklist

1. Supabase project exists. Data API on. Automatic RLS on. **Automatically expose new tables** may stay off if you expose `bot_snapshots` by hand.
2. Paste the SQL above. Confirm Table Editor shows `bot_snapshots`.
3. Expose the table on the Data API if REST 404s.
4. VM `.env` has `DASHBOARD_PUSH_URL` + service role as `DASHBOARD_PUSH_KEY`. Laptop copy is local-only. **Never commit `.env`.**
5. From the repo: `python -m src.dashboard_push --ping` must print HTTP 200.
6. Vercel: import this GitHub repo, **Root Directory = `dashboard`**. Env:
   - `NEXT_PUBLIC_SUPABASE_URL`
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - `DASHBOARD_GATE`
7. Share the Vercel URL + gate with your friend. Do not share OpenD or `.env`.
8. You do **not** need VDI open for the website. You **do** need VDI for new trades and new headlines.

Local site preview (optional):

```text
cd dashboard
npm install
npm run dev
```

Open `http://localhost:3000/login`.

---

## 7. Operator notes

- After a code pull on the VM, copy the new `DASHBOARD_PUSH_*` lines into the **VM** `.env`. Restart `run_trader.bat`.
- `test_inference.bat` still places no orders and does not push.
- `predict_now.bat` still does not write `state.pkl` and does not push.
- If the site is empty: trader has not pushed yet, or `--ping` is failing (SQL / expose / keys).
- If the site is stale at lunch but the trader window is open: heartbeat should keep `updated_at` fresh. If not, check trader logs for `Dashboard push`.
- Out of scope for v1: live ticks, friend placing orders, Streamlit, VM keep-alive, tax lots, `predict_now` preview snapshots.

---

## 8. What not to do

- Do not put the service role on Vercel or in `NEXT_PUBLIC_*`.
- Do not call Alpha Vantage from the Next app.
- Do not host the shared UI on the VDI.
- Do not retrain, Promote, or change goldens to “make the dashboard work”.
