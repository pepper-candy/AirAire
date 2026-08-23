# AirAire

- Paper-trading desk: PPO sizes five names (HK tech vs US defensive)
- 10-minute prices, calendar features, and a news score
- Alpha Vantage reads headlines; Futu OpenD places **SIMULATE** orders
- Read-only site keeps the last book when the trader is offline

**Watch the paper book:** [airaire.vercel.app](https://airaire.vercel.app)

- Paper only. No real money. No warranty
- Tomorrow the market will do something we did not predict

---

## What we built


| Piece                              | What it does                                                                                                     |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `src/trading_env.py`               | Gymnasium env. State = 10-min prices + long-term HK×US features + calendar + `news_score` + inventory.           |
| `src/train.py` / `train_gpu_v2.py` | Rolling 30-session-day PPO windows. v2 is the live trainer (8 updates / window, collapse guards).                |
| `src/finetune_latest.py`           | Overnight 1-window fine-tune. Human **Promote** on Telegram. Never silent-overwrite `best_model.zip`.            |
| `src/inference.py`                 | Live loop: 60s poll, order on a **new 10-min bar** or a **news jump ≥ 0.25**. Catch-up if you start late.        |
| `src/news_loader.py`               | Alpha Vantage `NEWS_SENTIMENT` for training parquet **and** the live poller. Same score, plus titles for humans. |
| `src/dashboard_push.py`            | Fail-open POST to Supabase. Closed-market rows are `kind=heartbeat` (keep-alive only).                           |
| `dashboard/`                       | Next.js blotter on Vercel. Reads snapshots. Does not trade, does not call Alpha Vantage.                         |


**Two APIs, do not mix them**

- **Futu OpenD** — 10-min OHLCV, snapshots, `TrdEnv.SIMULATE` orders.
- **Alpha Vantage** — headline sentiment only.

---

## Universe

Five names we can hold, plus two observers the policy cannot buy.


| Role         | Ticker     | Name      |
| ------------ | ---------- | --------- |
| HK tech      | `HK.00700` | Tencent   |
| HK tech      | `HK.03690` | Meituan   |
| HK tech      | `HK.03750` | CATL      |
| US defensive | `US.COST`  | Costco    |
| US defensive | `US.KO`    | Coca-Cola |
| Observer     | `HK.HSI`   | Hang Seng |
| Observer     | `US.SPX`   | S&P 500   |


A closed market **keeps** the position. Gating that name to action `0` would flatten the other book (US names during the HK session, and the reverse after 16:00 HKT).

---

## How the live loop thinks

```text
run_trader.bat
  → load models/news_gpu_v2/best_model.zip
  → load state.pkl first (never double-buy after a restart)
  → catch-up missing Futu 10-min bars (no orders)
  → every 60s:
        if both cash sessions closed → heartbeat snapshot, sleep
        else score news, predict
        order only if new 10-min close OR |Δnews| ≥ 0.25
        push live snapshot to Supabase
```

Fine-tune is a different bat, after the US cash close (or overnight):

```text
run_finetune.bat
  → Futu overlay (if OpenD is up)
  → Alpha Vantage last ~30 days
  → 1 PPO window from the newest checkpoint / finetuned zip
  → PROMOTION CHECK vs live_best.json
  → Telegram Promote / Keep (10 min). Timeout = keep.
```

We **do not** retrain from Window ~90 (entropy collapse). We **do not** run a full `train_gpu_v2` into `models/news_gpu_v2`. Story: `[guide/PHRASE-4-FUTURE-FINETUNE-GUIDE.md](guide/PHRASE-4-FUTURE-FINETUNE-GUIDE.md)`.

---

## Goldens (in this repo)

`models/news_gpu_v2/` is live. Fine-tune will not overwrite these three zips.


| File                        | Role                                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------- |
| `best_model.zip`            | What inference loads. Started as Window 113 (Calmar ≈ 2.05). Changes only on Promote. |
| `checkpoint_2026-08-12.zip` | Original trading golden. Museum copy.                                                 |
| `checkpoint_2026-08-18.zip` | Window 118 seed (Calmar ≈ 1.83) until a newer `finetuned_*.zip` exists.               |
| `live_best.json`            | Calmar hurdle for the next Promote.                                                   |
| `finetune_log.csv`          | Daily log + pinned golden rows.                                                       |
| `training_log_history.csv`  | Long GPU run. Do not reorder.                                                         |


`state.pkl` is the **paper book** (cash, holdings, last bar). Local only. Recreated as a cold start if missing — dangerous if Futu already has positions.

---

## Setup (programmer)

Python 3.12, Windows-friendly. GPU is optional for inference; daily fine-tune wants CUDA.

```text
python -m venv venv_gpu
venv_gpu\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` (names only in git):

```text
ALPHAVANTAGE_API_KEY=
FUTU_HOST=127.0.0.1
FUTU_PORT=11111
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
DASHBOARD_PUSH_URL=https://YOUR_PROJECT.supabase.co
DASHBOARD_PUSH_KEY=          # service_role — trader machine only
DASHBOARD_SNAPSHOTS_TABLE=bot_snapshots
DASHBOARD_GATE=
```

Install [Futu OpenD](https://www.futunn.com/), log in, paper account on. Smoke tests in `test/` (early wiring). Button-only Promote test:

```text
python test/tg-promote-button-test.py
```

Do not run that during a real Promote wait.

### Bats

Copy from `[execution/](execution/)` onto the machine that runs OpenD. They `cd` into the repo and call `venv_gpu`.


| File                 | Command                                                                        |
| -------------------- | ------------------------------------------------------------------------------ |
| `run_trader.bat`     | `python -m src.inference --poll-seconds 60`                                    |
| `run_finetune.bat`   | `python -m src.finetune_latest --windows 1 --device cuda`                      |
| `test_inference.bat` | `python -m src.inference --dry-run --once` — no orders, no push                |
| `predict_now.bat`    | `python -m src.inference --predict-now` — one live score, no `state.pkl` write |


### Dashboard

The public blotter is [airaire.vercel.app](https://airaire.vercel.app) (shared `DASHBOARD_GATE` if we gave you one). To run the same app on your machine:

```text
cd dashboard
copy .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000/login`. Gate password = `DASHBOARD_GATE`.

Vercel: import this repo, **Root Directory =** `dashboard`. Env: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `DASHBOARD_GATE`. **Never** put the Supabase service role or Alpha Vantage key on Vercel.

SQL and snapshot contract: `[guide/PHRASE-5.md](guide/PHRASE-5.md)`.

The header **TEST** button paints a local fake book so you can tune layout. It does not write Supabase. Turn it off before you demo the live row.

Equity Path counts **session snapshots only** (HK or US cash open). Heartbeats while both sessions are closed keep the stale banner honest and are not a path.

---

## Daily rhythm (operator)

1. Before HK 09:30 (or any time — catch-up will sync): `run_trader.bat`. Leave it open.
2. After US cash close, or overnight: `run_finetune.bat`. Promote only if you want that zip live, then restart the trader.
3. After a `git pull`: `test_inference.bat` with the trader **stopped**, recopy bats if they changed.
4. Curiosity while the trader is looping: `predict_now.bat`.

The website stays up when the trader is offline. New trades need the trader process.

---

## Layout

```text
src/                 trader, trainers, news, dashboard push
execution/           bats (copy next to OpenD)
dashboard/           Next.js read-only site
data/raw/bloomberg/  10-min base bars
data/raw/news/       articles_*.csv + news_*.csv
data/enhanced/       enhanced_data.parquet (price + news_score)
models/news_gpu_v2/  live goldens
models/old/          earlier runs (museum)
guide/               runbooks — PHRASE-4-EXECUTION and PHRASE-5 win
test/                one-shot API / Telegram / Futu pings
```

---

## What we learned (short)

- Price-only PPO was a baseline. News in the observation is the point of the project.
- A long GPU run can **entropy-collapse**. Rolling back to the last healthy checkpoint (Window 112 → re-do 113–118) beat grinding forward from a dead policy.
- Sharpe on a 30-day window lied to us more than once. **Calmar** is the Promote bar.
- Poll every 60s so you notice the 10-min close; do not rebalance every 60s on the same bar.

---

[MIT](LICENSE) — Pepper Candy, 2026.