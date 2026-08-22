# Phase 5 — what landed after the plan (cross-check)

Use this page to verify the work after [`.cursor/plans/phase_5_dashboard_98933837.plan.md`](../.cursor/plans/phase_5_dashboard_98933837.plan.md) was approved. Phrase 4 trading logic was not redesigned.

**Commit on `main`:** `e0bf7a2` — *Add Phase 5 read-only dashboard without committing secrets.*  
**Parent:** `19d360a` — *working on phrase 5, dashboard* (plans + bat moves only)  
**Remote:** force-pushed to `origin/main` (`8c30236` → `e0bf7a2`) so GitHub no longer tracks `.env`.

`.env` is local-only. It is not in this commit.

---

## Story after the plan (short)

1. Plan locked: VM pushes snapshots; Vercel only reads; no trading on Vercel; VM idle is accepted.
2. First build wrote Python hooks + `dashboard/` + `guide/PHRASE-5.md`. Most of that was **not committed** yet.
3. `git filter-repo --path .env --invert-paths --force` stripped `.env` from all 17 commits, dropped `origin`, and reset **tracked** files to the last commit. Phrase 4 **committed** files stayed. Uncommitted Phase 5 edits on tracked files were wiped from the working tree.
4. Untracked files (`dashboard/`, `PHRASE-5.md`, `dashboard_push.py`, `.env.example`) were still on disk.
5. Tracked-file Phase 5 edits were written again from this chat (same patches as the first build), then committed as `e0bf7a2` and force-pushed.

---

## What this commit did **not** change

Tick these if you only want to confirm Phrase 4 is intact.

- [ ] No retrain, no Promote change, no goldens touched
- [ ] `run_trader.bat` / other bats were **not** in `e0bf7a2` (they already lived under `execution/` from `19d360a`)
- [ ] `--dry-run` was **not** added to the live trader bat
- [ ] Catch-up, 60s poll, news-jump ≥ 0.25, closed-market keep-holdings are still the Phrase 4 loop
- [ ] `latest_ticker_score()` still exists (now a wrapper)

---

## New files (27 files in `e0bf7a2`: 22 added, 5 modified)

### Docs and secrets hygiene

| File | What to check |
|---|---|
| [`.env.example`](../.env.example) | Placeholder names only. No real keys. Includes `DASHBOARD_PUSH_*` and `DASHBOARD_GATE`. |
| [`.gitignore`](../.gitignore) | Ignores `.env`, `dashboard/.env.local`, `node_modules`, `.next`, `data/logs/*.jsonl`, `__pycache__`. Keeps `.env.example`. |
| [`guide/PHRASE-5.md`](PHRASE-5.md) | Phase 5 source of truth: snapshot contract, SQL, env names, Vercel root = `dashboard`. |
| [`guide/PHRASE-4-EXECUTION-&-DAILY-WORK.md`](PHRASE-4-EXECUTION-&-DAILY-WORK.md) | **§6b only.** Title is now “Dashboard is Phase 5”. Points at `PHRASE-5.md`. Adds `data/logs/trades.jsonl` to the table. Rest of Phrase 4 unchanged. |
| [`data/logs/.gitkeep`](../data/logs/.gitkeep) | Empty folder so the blotter path exists. jsonl stays untracked. |

- [ ] `.env` is **not** in `git ls-files`
- [ ] §6b points at Phase 5; §0–§6 / §7–§8 still Phrase 4 paper-trade

### Python (VM writer)

| File | How it changed |
|---|---|
| [`src/utils.py`](../src/utils.py) | Added `DATA_LOGS` and `TRADES_JSONL` only. |
| [`src/news_loader.py`](../src/news_loader.py) | Added `_headline_rows` + `latest_ticker_news()` (same AV call, returns score **and** titles). `latest_ticker_score()` now calls that and returns the float. |
| [`src/dashboard_push.py`](../src/dashboard_push.py) | **New.** Builds snapshot JSON, appends/reads `trades.jsonl`, POSTs to Supabase. Never raises. `python -m src.dashboard_push --ping` is GET-only. Skip if env unset. |
| [`src/inference.py`](../src/inference.py) | See hooks below. No new CLI flags. |

**`inference.py` hooks (the only live-loop change):**

- [ ] Import `latest_ticker_news`, `append_fill`, `push_live_snapshot`
- [ ] `NewsPoller` caches `_headlines`; `_call_alpha_vantage` uses `latest_ticker_news`
- [ ] `place_order` returns `(ok, order_id)` instead of `bool`
- [ ] After a real SIMULATE fill: append one jsonl line, then Telegram (same as before)
- [ ] `_maybe_push_dashboard` after each live cycle, and on closed-market heartbeat
- [ ] Push **off** for `--dry-run` and `--predict-now` (`push_dashboard = persist_state`)

### Next.js site (`dashboard/`)

Read-only UI. Vercel Root Directory must be `dashboard`.

| File | Role |
|---|---|
| `package.json` / `package-lock.json` / `tsconfig.json` / `next.config.ts` | Next 15 App Router app |
| `middleware.ts` + `lib/gate.ts` + `app/login/page.tsx` + `app/api/gate/route.ts` | Shared `DASHBOARD_GATE` cookie / `?gate=` |
| `lib/types.ts` + `lib/supabase.ts` + `app/api/snapshot/route.ts` | Anon SELECT latest + last 120 rows; stale if `updated_at` > 180s |
| `app/page.tsx` + `components/Dashboard.tsx` + `app/globals.css` | Stale banner, book, headlines, fills, P&L, sparkline; poll 20s |
| `dashboard/.env.example` | `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `DASHBOARD_GATE` |
| `dashboard/.gitignore` | `.env.local`, `node_modules`, `.next` |

- [ ] No `ALPHAVANTAGE` / Futu / service role in any `dashboard/` source file
- [ ] `NEXT_PUBLIC_*` is URL + **anon** only

---

## Still your job (not in the commit)

1. Run the `bot_snapshots` SQL in [`PHRASE-5.md`](PHRASE-5.md) §3.
2. Expose `public.bot_snapshots` on the Data API (auto-expose new tables is off).
3. `python -m src.dashboard_push --ping` → HTTP 200.
4. Copy `DASHBOARD_PUSH_*` into the **VM** `.env` (not git).
5. Vercel: this repo, Root Directory `dashboard`, three env vars above. Never the service role.

---

## Quick git commands

```text
git show --stat e0bf7a2
git diff 19d360a..e0bf7a2 -- src/inference.py src/news_loader.py src/utils.py
git ls-files .env
```

`git ls-files .env` should print nothing.
