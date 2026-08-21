# Phase 4 — paper trading runbook (final)

**Date:** 2026-08-21  
**Status:** Code and daily workflow are locked for first paper-trade. This page is the team source of truth.

If any other file in `guide/` disagrees with this page, **this page wins**. Do not treat older markdown as a to-do list.


| Read                                                                     | Treat as history only                                                           |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| **This file**                                                            | `guide/PHRASE-4.md` (18 Aug — still says “news not implemented”; that is false) |
| `guide/PHRASE-4-FUTURE-FINETUNE-GUIDE.md` (entropy / resurrection story) | DeepSeek execution drafts (replaced)                                            |
| `guide/TRAINING-RESULTS-20260819.md` / `EVALUATION-POLICY-FIX.md`        | Retraining from Window 90 (**forbidden**)                                       |


**Machines**


| Where                                                                  | Path                                                    |
| ---------------------------------------------------------------------- | ------------------------------------------------------- |
| Laptop repo                                                            | `C:\Users\mongk\Desktop\airaire`                        |
| GPU VM repo (same tree)                                                | `C:\Users\klmong\Desktop\airaire`                       |
| VM desktop (the `.bat` files live **here**, not inside the repo) | `C:\Users\klmong\Desktop` |
| Futu OpenD on the VM                                                   | `C:\Users\klmong\Desktop\Futu_OpenD_10.10.7008_Windows` |


Copy `run_trader.bat`, `run_finetune.bat`, `test_inference.bat`, and `predict_now.bat` onto the VM desktop after every pull if those files changed.

---



## 0. First blink (do this in order)

1. Pull this repo onto the GPU VM. Copy the four `.bat` files to the VM desktop.
2. Confirm `.env` has `ALPHAVANTAGE_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, Futu host/port if not default.
3. Confirm goldens exist: `models/news_gpu_v2/best_model.zip`, `checkpoint_2026-08-12.zip`, `checkpoint_2026-08-18.zip`.
4. Double-click `test_inference.bat`. The banner must show `models/news_gpu_v2/best_model.zip`. Dry-run places **no** SIMULATE orders.
5. Before HK 09:30 (or any time — catch-up will sync): `run_trader.bat`. Leave it running.
6. After US cash close, or overnight: `run_finetune.bat`. Watch Futu overlay, then Alpha Vantage 30-day refresh, then PPO. If Calmar beats **live** best, Telegram **Promote** or **Keep** (10 minutes). Restart the trader after a Promote.

Do **not** run `train_gpu_v2` into `models/news_gpu_v2`. Do **not** retrain from Window 90.

---



## 1. Story so far



### Price-only → news → GPU

- Phase 3: price-only 30-day PPO (`models/price-only/`). Those Sharpes are not the live metric.
- News from **Alpha Vantage** `NEWS_SENTIMENT` (education / paid, **75 calls per minute**) is merged into `data/enhanced/enhanced_data.parquet`.
- Env observation: 10-min price window, long-term features (HK×US corr), calendar, **news_score**, inventory.
- `models/news_gpu/` is the first GPU folder. Leave it. `models/news_gpu_v2/` **is live.**



### Entropy collapse (Window ~90)

The first long GPU v2 run died in exploration: `entropy_loss` from about **-7 toward 0**, `approx_kl` spiked. Windows 113–118 continued from that dead brain were weak (W118 Calmar ~1.37).

**We do not retrain from Window 90. We do not resume that brain.**

### Resurrection (20 Aug 2026)

`train_gpu_v2 --resume 113` loaded **healthy Window 112** and re-trained 113–118.


| Window  | End        | Return | Max DD | Calmar                         | Role                   |
| ------- | ---------- | ------ | ------ | ------------------------------ | ---------------------- |
| **113** | 2026-08-12 | ~18.0% | ~8.77% | **2.05** (`2.053856720691549`) | Paper-trading golden   |
| **118** | 2026-08-18 | ~16.4% | —      | **1.83** (`1.832871817457733`) | Continue-training seed |


Exact rows: `models/news_gpu_v2/training_log_history.csv` (**museum** — daily fine-tune does not append here).

### What Phase 4 code does

1. `src/inference.py` loads `models/news_gpu_v2/best_model.zip` (fallback `checkpoint_2026-08-12.zip`). Startup banner logs path / size / role.
2. **Catch-up:** late start (e.g. 12:45) pulls missing Futu 10-min bars, seeks the env, restores `state.pkl`. **No orders during catch-up.**
3. `src/finetune_latest.py`**:** last 1–3 windows, GPU v2 PPO (8 updates/window). Warm-start newest `checkpoint_*.zip` / `finetuned_*.zip`. **Never trains from** `best_model.zip`**.**
4. **Before PPO:** Futu bars, then a forced Alpha Vantage refresh of the last ~30 days (see §4).
5. **Promote is human.** Compare new Calmar to **live** `best_model` (`live_best.json`), not a frozen 2.05 forever. Telegram Promote / Keep. No silent copy.
6. `finetune_log.csv` is append-only: pinned `trading_golden` (W113), `training_seed` (W118), `live_best` (moves on Promote), then daily `finetune` rows.
7. **Protected zips** in `news_gpu_v2`: `best_model.zip`, `checkpoint_2026-08-12.zip`, `checkpoint_2026-08-18.zip`. Fine-tune and `train_gpu_v2` refuse to overwrite them. Only Promote / `--promote-zip` may copy onto `best_model.zip`.
8. **VM desktop bats** always `cd` to `C:\Users\klmong\Desktop\airaire`.



### Corrections vs early drafts


| Draft / old habit                                    | What we actually run                                                          |
| ---------------------------------------------------- | ----------------------------------------------------------------------------- |
| Futu supplies news                                   | **No.** Futu = prices + SIMULATE orders. News = Alpha Vantage only            |
| News only if Futu added bars; cache often skipped AV | Fine-tune **always** re-queries ~30 days of `NEWS_SENTIMENT` before PPO       |
| 5-minute live news cap (old academic caution)        | Live poller **60s/ticker**. 5 names ≈ 5/min, under 75/min                     |
| Always beat frozen Calmar 2.05                       | Beat `live_best.json` (starts at 2.05, moves after Promote)                   |
| Open CSVs every morning                              | Terminal **PROMOTION CHECK** + Telegram buttons                               |
| Auto-copy a better zip onto `best_model.zip`         | You tap **Promote**. Keep / timeout / Ctrl+C = no copy                        |
| Poll 60s and send orders every cycle                 | 60s notices a new 10-min close. Orders on **new bar** or **news jump ≥ 0.25** |
| Gate closed names to action 0                        | That **flattens** the other market. Closed names **keep** the position        |
| Model path `models/news/`                            | `models/news_gpu_v2/best_model.zip`                                           |
| Retrain from Window 90                               | Forbidden                                                                     |


---



## 2. Two APIs (do not mix them up)


| Job                                   | Source                             | Used for                                   |
| ------------------------------------- | ---------------------------------- | ------------------------------------------ |
| 10-min OHLCV, snapshots, paper orders | **Futu OpenD**                     | Catch-up, live prices, `TrdEnv.SIMULATE`   |
| Headline sentiment → `news_score`     | **Alpha Vantage** `NEWS_SENTIMENT` | Training parquet **and** live `NewsPoller` |


There is no Futu news feed in this project.

Alpha Vantage education is treated as paid: **75 requests / 60 seconds**. Historical fetch uses 7-day chunks (so liquid names are less likely to hit the 1000-article cap). Mean of the last **10** headlines matches live scoring.

Daytime NewsPoller writes into the live env / `state.pkl`. It does **not** write the training parquet. Overnight fine-tune is what stamps fresh sentiment onto new Futu bars for the next window — so the policy is trained on the same feed it will see in realtime.

---



## 3. Golden files (do not “clean up”)

Folder: `models/news_gpu_v2/`


| File                        | Role                                                               |
| --------------------------- | ------------------------------------------------------------------ |
| `best_model.zip`            | **What inference loads.** Starts as W113. Changes only on Promote. |
| `checkpoint_2026-08-12.zip` | Same weights as the original trading golden. Museum.               |
| `checkpoint_2026-08-18.zip` | W118 seed until a newer `finetuned_*.zip` exists.                  |
| `live_best.json`            | Calmar hurdle for the next Promote.                                |
| `finetune_log.csv`          | Daily log + three pinned rows.                                     |
| `training_log_history.csv`  | Long GPU run. Do not reorder.                                      |


Fine-tune writes `finetuned_YYYY-MM-DD.zip` and `checkpoint_{window_end}.zip` unless that name is protected.

---



## 4. Live loop (paper trade)

Universe: **5 tradable** (`HK.00700`, `HK.03690`, `HK.03750`, `US.COST`, `US.KO`) + **2 observers** (`HK.HSI`, `US.SPX`). The policy cannot buy anything else.

### Hours

- HK: 09:30–12:00 and 13:00–16:00 HKT, weekdays. **Lunch is closed.**
- US: 09:30–16:00 America/New_York (DST-aware).
- Both shut: HOLD, sleep (capped at 5 minutes so we wake for HK 13:00).
- **Per name:** if that market is shut, **keep holdings**. Do not flatten. Tencent stays on the book after 16:00 HKT; Costco stays on the book during the HK morning.



### Catch-up (no orders)

1. Load `state.pkl` **before** any order.
2. Futu 10-min history → overlay panel → `seek_to_datetime(now)` → restore cash/holdings.
3. Persist `last_bar_datetime`. No SIMULATE orders in this step.
4. The first **open-market** cycle after that may trade (catch-up never synced the book to the policy).



### 60-second poll vs 10-minute brain

Training `TradingEnv.step()` jumps bar `t` → `t+1` instantly. Live we poll every **60s** so a new Futu kline is noticed within about a minute of the close.

We do **not** rebalance every minute on price noise. Same completed bar + news unchanged → `[preview, no order]`.

Orders fire when:

1. The completed-bar id changed, **or**
2. Any news score moved by **≥ 0.25** (and that name’s market is open).

News is fetched at most every **60 seconds** per **open** ticker. Extra loops reuse cache. A jump ≥ 0.25 can still move the book **inside** a 10-min bar — that is the information-asymmetry path the env was trained with.

Sizing: `action * equity / live_price`, lot-rounded. `TrdEnv.SIMULATE` **only.**

### Resume contract

1. Load `state.pkl` first.
2. After a new-bar / news cycle, persist holdings, cash, last action, `last_order_bar`.
3. Ctrl+C flushes again. `state.pkl` wins over OpenD if they drift (log a warning; do not double-buy).

Restart `run_trader.bat` **after a Promote** — the policy is loaded once at start.

---



## 5. Daily fine-tune

```text
python -m src.finetune_latest --windows 1 --device cuda
```

Same command as `run_finetune.bat`. Typical GPU time: ~2–3 minutes of PPO, plus a short AV burst (~30 days × 5 tickers × 7-day chunks, well under 75/min).

1. Load `enhanced_data.parquet`.
2. Futu 10-min refresh (OpenD). If OpenD is down, **news + PPO still run**.
3. **Always** re-query Alpha Vantage for the last `--news-days` (default **30**). Merge onto **recent** bars only; older `news_score` stays. `--skip-news` skips AV. `--cache-news` uses local cache if it already covers the window. `--no-futu` skips OpenD but **still** refreshes news unless `--skip-news`.
4. Last 1–3 rolling 30-day windows (`--windows 1` daily).
5. Collapse / NaN weights are discarded.
6. Save `finetuned_{HK-date}.zip`.
7. Append `finetune_log.csv`. Print **PROMOTION CHECK** vs **live** Calmar.
8. Telegram **only if new Calmar > live Calmar**: **Promote to best_model** / **Keep current**.
9. Wait **600s**. Promote = copy onto `best_model.zip` + update `live_best.json`. Keep / timeout / Ctrl+C = no copy.

Missed the wait:

```text
python -m src.finetune_latest --promote-zip models/news_gpu_v2/finetuned_YYYY-MM-DD.zip
```

`--no-telegram` and `--promote-wait 0` notify (or stay quiet) without copying.

Do **not** run a full `train_gpu_v2` into `models/news_gpu_v2` to “refresh best_model”. That path **refuses** to overwrite the goldens.

Telegram: trader fill alerts use `sendMessage`. Promote wait uses `getUpdates`. Do not run `test/tg-promote-button-test.py` during a Promote wait. Trader + fine-tune together is fine.

Button-only test (does **not** copy `best_model.zip`):

```text
python test/tg-promote-button-test.py
```

---



## 6. Batch files (GPU VM desktop)

They always:

- `cd /d C:\Users\klmong\Desktop\airaire`
- start `C:\Users\klmong\Desktop\Futu_OpenD_10.10.7008_Windows\FutuOpenD.exe` (trader and fine-tune)


| File | Command |
|---|---|
| `run_trader.bat` | `python -m src.inference --poll-seconds 60` |
| `run_finetune.bat` | OpenD + Alpha Vantage 30-day news, then `python -m src.finetune_latest --windows 1 --device cuda` |
| `test_inference.bat` | `python -m src.inference --dry-run --once` — smoke test only. No OpenD, no orders. **Do not** put `--dry-run` on `run_trader.bat`. |
| `predict_now.bat` | `python -m src.inference --predict-now` — one live **predict**, no trade. Quotes + news, no Futu order, no `state.pkl` write. Safe while the trader is running. |


Needs `venv_gpu` and `.env`.

Cadence (HK):

1. `run_trader.bat` before 09:30, or any time (catch-up will sync).
2. After US cash close, or overnight: `run_finetune.bat`. Promote only if you want that zip live, then restart the trader.
3. After a code pull: `test_inference.bat` (trader **stopped**), then re-copy bats to the desktop if they changed.
4. Anytime curiosity: `predict_now.bat` even if the trader is already looping. Look for `[predict-now]` and the summary. No trade.

---

## 6b. There is no live dashboard

`guide/PLAN.md` listed a Streamlit app (`dashboard/streamlit_app.py`). **It was never built.** There is no web page that streams news or positions.

What you have instead:

| Place | What you see |
|---|---|
| Trader / predict terminal | Live quotes, news scores, `[predict-now]` or order reasons |
| Telegram | Fill alerts (when the trader actually sends a SIMULATE order); Promote/Keep after fine-tune |
| Futu OpenD (SIMULATE account) | Paper positions and fills on Futu’s own UI |
| `state.pkl` | Book the bot believes (cash, holdings, last news, last bar) |
| `logs/` | Fine-tune / train text logs |

Live news exists in the **process** (Alpha Vantage → `NewsPoller` → env `news_score`). It is not pushed to a dashboard. You read it in the console (and Telegram only after a real fill).

---



## 7. Why this is the paper-trade setup

- The policy was trained on **completed 10-min bars**, not ticks. Execute near that close, once per bar, unless news jumps.
- HK lunch and US hours are real. Flattening the other market would be a fake fill and a real book error.
- Overnight AV refresh + live 60s poller mean the fine-tune window sees the **same news family** the bot will react to the next day.
- Calmar Promote with a human tap keeps one hot window from silently replacing W113.
- Futu **SIMULATE** fills are optimistic. Treat P&L as a process check, not a lock of the 2.05 backtest.

We will not: trade outside the five names, fire in lunch, use Window-90 weights, use Futu as a news source, or overwrite `best_model.zip` without Promote.

---



## 8. Operator checklist

- [ ] Four `.bat` files on the **VM desktop** (`run_trader`, `run_finetune`, `test_inference`, `predict_now`)
- [ ] OpenD up before trader / fine-tune
- [ ] `.env`: Alpha Vantage + Telegram
- [ ] Banner: `models/news_gpu_v2/best_model.zip` exists
- [ ] Late start: catch-up logs, then orders only on an allowed cycle
- [ ] Closed names: `market closed — keep holdings` (not a sell)
- [ ] Same bar: `[preview, no order]`
- [ ] Fine-tune log: Futu overlay, then `Refreshing Alpha Vantage NEWS_SENTIMENT`
- [ ] PROMOTION CHECK vs **LIVE** Calmar
- [ ] Promote copies; Keep leaves the zip
- [ ] Restart trader after Promote
- [ ] `state.pkl` current after Ctrl+C