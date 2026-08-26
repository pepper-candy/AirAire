# AirAire

A paper-trading desk that sizes five names from **10-minute prices**, a **calendar**, and a **news score**. PPO is the trader. Alpha Vantage is the analyst. Futu OpenD places **SIMULATE** orders only.

The bet is not “predict tomorrow.” It is **see the headline and the bar a little earlier** than a slow trend-follower, in a noisy HK-tech vs US-defensive book.

**Watch the paper book:** [AIRAIRE](https://airaire.vercel.app) (pw:Public)
**Dashboard Demo Video:** [AIRAIRE - Paper Book Demo](https://youtube.com/shorts/8_Viz2aZWdI?feature=share)

- Paper only. No real money. No warranty.
- Tomorrow the market will do something we did not predict.

**Now (Aug 2026):** V2 is what the live loop loads (`models/news_gpu_v2/best_model.zip`, promoted Calmar ≈ 4.91). Phase 6 / V3 is a **parallel research retrain** — volume plus Hang Seng and S&P 500 in the observation. It is not live yet.

---

## The idea

HK tech and US defensives often pull against each other. A closed cash session **keeps** that side of the book (zeroing a closed name would flatten the open side). News is in the observation so the policy can rebalance when the tape has not moved yet.

We treat **in-sample Calmar as homework**. A model that reprints +100% on the same 30 days it just trained has not earned the live zip. Promote is a human Telegram button. Paper P&L is the only honest test.

---

## How it works

```text
Bloomberg 10-min backbone  +  Alpha Vantage NEWS_SENTIMENT
        ↓
enhanced parquet (price + news_score + calendar / long-term features)
        ↓
PPO (rolling 30 session-day windows, 8 updates each)
        ↓
models/news_gpu_v2/best_model.zip     ← only this is live
        ↓
inference (60s poll) → Futu SIMULATE → Supabase snapshot → Vercel blotter
```

**Two APIs, do not mix them**

- **Futu OpenD** — 10-min OHLCV, snapshots, `TrdEnv.SIMULATE` orders.
- **Alpha Vantage** — headline sentiment only.

The live loop does **not** rebalance every 60 seconds. It polls so it notices a new 10-minute close (or a news jump ≥ 0.25), then it may order.

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

Overnight (after the US cash close) is a different bat: one PPO window, then **Promote / Keep** on Telegram. Timeout = keep. We do **not** silent-overwrite the live zip, and we do **not** grind a full retrain into `models/news_gpu_v2` after the entropy collapse around window 90.

---

## Universe

Five names we can hold, plus two observers the policy cannot buy. V2 **trains and infers on the five**. V3 **sees all seven** (still trades five).


| Role         | Ticker     | Name      |
| ------------ | ---------- | --------- |
| HK tech      | `HK.00700` | Tencent   |
| HK tech      | `HK.03690` | Meituan   |
| HK tech      | `HK.03750` | CATL      |
| US defensive | `US.COST`  | Costco    |
| US defensive | `US.KO`    | Coca-Cola |
| Observer     | `HK.HSI`   | Hang Seng |
| Observer     | `US.SPX`   | S&P 500   |

---

## What we built


| Piece                              | What it does                                                                                              |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `src/trading_env.py`               | Gymnasium env. State = 10-min prices + long-term HK×US features + calendar + `news_score` + inventory.    |
| `src/train.py` / `train_gpu_v2.py` | Rolling 30-session-day PPO. Live trainer: 8 updates / window, collapse guards.                            |
| `src/finetune_latest.py`           | Overnight 1-window fine-tune. Human **Promote**. Never silent-overwrite `best_model.zip`.                  |
| `src/inference.py`                 | Live loop: 60s poll, order on a **new 10-min bar** or a **news jump ≥ 0.25**. Catch-up if you start late. |
| `src/news_loader.py`               | Alpha Vantage for the training parquet **and** the live poller. Same score, plus titles for humans.       |
| `src/dashboard_push.py`            | Fail-open POST to Supabase. Closed-market rows are `kind=heartbeat`.                                      |
| `dashboard/`                       | Next.js blotter on Vercel. Reads snapshots. Does not trade, does not call Alpha Vantage.                  |
| `src/*_v3.py`                      | Isolated Phase 6 research track. Writes `enhanced_v3.parquet` and `models/news_gpu_v3/` only.              |


---

## Phase 6 — V3 research (not live)

V2’s volume channel was mostly zeros (Bloomberg). HSI and SPX sat in the parquet and never entered `model.predict()`. Changing that **changes the observation** (782 → **1082**), so V3 is a full retrain in new files. The live trader cannot load a V3 zip.

**Data.** Same 6-month regime as V2 (`2026-02-24` → `2026-08-21`). Bloomberg keeps the price clock. TradingView overlays volume (ignore `TSE_DLY_3750` — Tokyo, not CATL). Futu supplies CATL volume only. News is copied from the V2 parquet so the extra information is volume + observers, not a new news tape.

**Selection.** After each window we score the **next 5 session days** (not the homework window). `best_model.zip` is the rolling **median of the last 10 holdouts**, so one Trump Monday or weekend gap cannot veto a useful policy. That median is still not live P&L.

Do not point `run_trader.bat` at V3. There is no live path yet that fetches HSI/SPX on the 60s poll. Runbook: [guide/PHRASE-6-IMPLEMENTED.md](guide/PHRASE-6-IMPLEMENTED.md). Fine-tune / collapse story: [guide/PHRASE-4-FUTURE-FINETUNE-GUIDE.md](guide/PHRASE-4-FUTURE-FINETUNE-GUIDE.md).

---

## Goldens (in this repo)

`models/news_gpu_v2/` is live. Fine-tune will not overwrite these three zips.


| File                        | Role                                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------- |
| `best_model.zip`            | What inference loads. Promoted only. Current bar ≈ Calmar 4.91 (window 121 lineage).  |
| `checkpoint_2026-08-12.zip` | Original trading golden. Museum copy.                                                 |
| `checkpoint_2026-08-18.zip` | Window 118 seed (Calmar ≈ 1.83) until a newer `finetuned_*.zip` exists.               |
| `live_best.json`            | Calmar hurdle for the next Promote.                                                   |
| `finetune_log.csv`          | Daily log + pinned golden rows.                                                       |
| `training_log_history.csv`  | Long GPU run. Do not reorder.                                                         |


`state.pkl` is the **paper book** (cash, holdings, last bar). Local only. Recreated as a cold start if missing — dangerous if Futu already has positions.

`models/news_gpu_v3/` is research only.

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

SQL and snapshot contract: [guide/PHRASE-5.md](guide/PHRASE-5.md).

The header **TEST** button paints a local fake book so you can tune layout. It does not write Supabase. Turn it off before you demo the live row.

Equity Path counts **session snapshots only** (HK or US cash open). Heartbeats while both sessions are closed keep the stale banner honest and are not a path.

---

## Daily rhythm (operator)

1. Before HK 09:30 (or any time — catch-up will sync): `run_trader.bat`. Leave it open.
2. After US cash close, or overnight: `run_finetune.bat`. Promote only if you want that zip live, then restart the trader.
3. After a `git pull`: `test_inference.bat` with the trader **stopped**, recopy bats if they changed.
4. Curiosity while the trader is looping: `predict_now.bat`.

The website stays up when the trader is offline. New trades need the trader process.

V3 GPU train can share the same `venv_gpu` in another terminal. It writes a different folder and does not use OpenD once the parquet is cached.

---

## Layout

```text
src/                  trader, trainers, news, dashboard push
execution/            bats (copy next to OpenD)
dashboard/            Next.js read-only site
data/raw/bloomberg/   10-min price backbone
data/raw/tradingview/ volume overlay for V3 (ignore TSE_DLY_3750)
data/raw/futu/v3/     CATL volume cache
data/enhanced/        V2 parquet (live) + enhanced_v3.parquet
models/news_gpu_v2/   live goldens
models/news_gpu_v3/   Phase 6 research (not live)
guide/                PHRASE-4 execution, PHRASE-5 dashboard, PHRASE-6-IMPLEMENTED
test/                 one-shot API / Telegram / Futu pings
```

---

## What we learned

- Price-only PPO was a baseline. News in the observation is the point of the project.
- A long GPU run can **entropy-collapse**. Rolling back to the last healthy checkpoint beat grinding forward from a dead policy.
- Sharpe on a 30-day window lied more than once. **Calmar** is the Promote bar — and even Calmar on the train window is in-sample.
- Poll every 60s so you notice the 10-min close; do not rebalance every 60s on the same bar.
- Extra features that change obs dim need an isolated retrain. A one-week holdout is a fair live picture and a noisy contest; elect on a rolling median, then paper-trade.

---

[MIT](LICENSE) — Pepper Candy, 2026.
