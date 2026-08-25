# Next chat: HK-long / US-short hybrid + execution plan

Hand this file to a **new Cursor chat**. Do not start by editing live `trading_env.py` or `models/news_gpu_v2/`. Live paper is V2 and it is making money.

**Date of this note:** 2026-08-25 (HKT).  
**Repo:** `airaire_fixing` / GPU box `C:\Users\klmong\Desktop\airaire`.  
**Live trader:** `execution/run_trader.bat` → `python -m src.inference` → `models/news_gpu_v2/best_model.zip` + `state.pkl`.

---

## 1. Goal (what the user asked)

Futu SIMULATE (and later live) is **not** one action space:

| Market | Allowed |
| :--- | :--- |
| HK (00700, 03690, 03750) | **Long only** `[0, 1]`. Shorts are rejected. |
| US (COST, KO) | **Long and short** `[-1, 1]`. Last night V2 made ~USD 300 / ~14.5% **because shorts were allowed**. |

User wants **one strategy that already knows this at train time**, not two brains (V2 vs V2.10) and not “clip HK shorts only at inference.”

They also want the **execution layer** (OpenD) to stop looking like instant fills:

- Rejected orders currently **vanish from the terminal** unless someone is watching.
- Size is often **single-digit US shares** on a HKD 1,000,000 book.
- **Buy 7 @ 98.12 then sell 14 @ 97.50** on the next bar.
- **Unfilled limits sit** until someone cancels them. OpenD does **not** clean them for us.
- After a Futu **warning/error**, the bot should **log it, keep the session file, and compensate** (retry legal size/price, or cancel, not pretend it filled).

---

## 2. What not to do

| Bad idea | Why |
| :--- | :--- |
| Edit live `src/trading_env.py` `_rebalance` then `finetune_latest` into `models/news_gpu_v2` | That **is** the winning US-short book. Next `run_trader.bat` restart would change live V2. |
| Load V2 zip into V2.10 | V2 obs **782** (5 names). V2.10 obs **932** (5 + HSI). Action V2 `[-1,1]` vs V2.10 all-five `[0,1]`. `PPO.load` will fail or be junk. |
| Resume V2.10 from Window 12 / 55 | Long-only on **all five** collapsed after ~W55. Structural, not “wrong checkpoint.” |
| DeepSeek “change Box lows to 0 for HK and load Aug-20 zip” | SB3 squashes into the **saved** `Box(-1,1)` head. New per-dim bounds often **refuse to load**. |
| `python -m src.reconcile_futu --apply` | Known CATL pickle vs OpenD drift. Fill log / dashboard is CATL truth. Do not mix US USD into HK cash. |
| Point `run_trader.bat` at a new zip mid-session | Only after an isolated paper path and `--predict-now`. |

---

## 3. Current live stack (facts)

- **Train recipe that generalized:** GPU V2 (`src/train_gpu_v2.py`), 8 PPO updates/window, news, 5 tradable names, **no HSI/SPX in the lookback**.
- **Paper:** `src/inference.py`. Target weight `action * equity / price`, then `round_to_lot`. **Books cash/holdings on OpenD submit**, not on fill. Working orders overlay as PENDING on the dashboard.
- **V3 inference** (`src/inference_v3.py`) already has `decide_order` (skip round-trip, cancel/replace pending). **V2 live loop does not call `decide_order`.** That is why 7-buy / 14-sell still happens on V2.
- **V2.10** (`src/train_gpu_v2_10.py`): V2 PPO recipe, **all five long-only**, Bloomberg 5 + **HSI only** (no SPX), volume 0, news copied from V2 parquet. Isolated: `enhanced_v2_10.parquet`, `models/news_gpu_v2_10/`. Research only. Do not replace live V2 with it.
- **VM dump of “good V2”:** `models/news_gpu_v2_20260823135426/`. Live bar in that folder is **`best_model.zip` ← `finetuned_2026-08-22.zip`** (window **121**, end **2026-08-21**, Calmar ~4.92). `checkpoint_2026-08-20.zip` is **not** in that dump; it lives under `models/old/news_gpu_v2_test/`. **Seed any fork from the zip actually loaded by the GPU `run_trader.bat`**, then confirm with the startup banner.

**Fine-tune incomplete day:** `drop_incomplete_klines` drops only the **open 10-min candle**. At **25 Aug 14:00 HKT** a 1-window fine-tune **will** use HK bars through ~13:50 and **will not** wait for US 25 Aug. Window end date becomes **25 Aug**. For a complete **24 Aug US close**, run after **04:00 HKT**.

---

## 4. Proposed model track: V2.11 (name TBD)

**Same as V2 (so the Aug-20 / W121 zip can load):**

- Obs dim **782**. No HSI/SPX in the cube (add observers only in a later v2.12 after this works).
- Policy head stays **`Box(-1, 1, shape=(5,))`** so `PPO.load` matches.
- News block unchanged. Panel: existing `enhanced_data.parquet` + Futu overlay (same as `finetune_latest.py`).

**Different from V2 (Futu-honest, isolated files only):**

- New env e.g. `src/trading_env_v2_11.py`: after the policy action, **HK weights `clip(w, 0, 1)`**, US unchanged `[-1, 1]`. Also `holdings[HK] = max(0, h)`.
- New dir e.g. `models/news_gpu_v2_11/`. Refuse to write `news_gpu_v2`.
- Warm-start: copy VM `best_model.zip` or the dated zip you trust → `--checkpoint`.
- Daily: `finetune` **1 window** into v2.11 after US close. Policy still *emits* HK shorts at first; they become no-ops; reward should shift hedges onto COST/KO. **One or two windows is adaptation, not a rewrite of +14.5%.**
- Paper later: `inference_v2_11.py` + `state_v2_11.pkl`. Never share SIMULATE with live V2 until the user says switch.

**Do not** train this by slicing V2.10 windows. **Do not** change Gym action lows unless you accept training from scratch.

---

## 5. Why orders “just disappear” (Futu + our code)

OpenD `place_order` returns `(ret, data)`. If `ret != RET_OK`, we log `place_order failed ...: {data}` and return `(False, "")`. There is **no Telegram**, **no fill row**, **no blotter line** for a reject. If the console is closed, the message is gone. **Session file logging is now required** (see §8).

We did **not** keep a historical reject catalog. Next time, paste the `data` string into `logs/trader_*.txt` (it will already be there). Known classes:

### 5.1 We already block or snap (order never sent, or sent with snapped price)

| Condition | What happens | Code |
| :--- | :--- | :--- |
| `qty == 0` after lot rounding | Silent skip in `place_order` | HK lot **100**; US lot **1**. Tiny HK targets vanish. Tiny US targets become 1–14 shares. |
| Limit not on tick | Snap then send | `round_to_tick`. US ≥ $1 → **2 decimals**. KO rejects like `98.123` were this. HK uses HKEX bands (`price_tick`). |
| Market closed for that name | No order, keep holdings | HK vs US independently. |
| First 10-min cash bar still open | No order | HK 09:30 / 13:00, US 09:30 ET. |
| No live snapshot price | HOLD | |
| `--predict-now` / dry-run | No SIMULATE order | |
| Same 10-min bar and news Δ < 0.25 | No rebalance | `NEWS_RETRADE_DELTA`. |

### 5.2 OpenD / SIMULATE typically rejects (`place_order failed`)

Exact English/Chinese strings vary by OpenD version. Treat these as **search terms** in the session log:

| Likely cause | Typical names | Notes |
| :--- | :--- | :--- |
| Price precision | price, tick, decimal, 价格精度 | Should be rare after `round_to_tick`. If it returns, log raw vs snapped. |
| Lot / qty | lot, 手数, quantity | HK not multiple of 100. |
| **HK short** | short, 卖空, close only, 平仓 | User constraint. V2 still **asks** for HK shorts; Futu rejects. US shorts OK. |
| Buying power / max | buying power, 购买力, insufficient | Cash locked in **working** limits + we already deducted on submit. |
| Price too far from market | spread, 偏离, max price | Limit miles from last. |
| Session / halt | not tradable, 停牌, market closed | |
| Rate limit | too frequent, 频率 | OpenD ~60 / 30s. We have `RateLimiter`. |
| Duplicate / locked | 重复, processing | |
| Sell more than available | position, 持仓不足 | Worse if we booked a fill that never happened. |
| Account / SIMULATE | trd_env, simulate | Wrong `TrdEnv` or unlocked account. |

**Compensation (not built on V2):** parse `data`, classify, then: snap qty/price and retry once; if HK short → force qty 0 and log; if power → cancel oldest working order then retry; never `holdings += delta` on reject.

### 5.3 Order accepted but “nothing happened”

This is **not** a reject. Limit sits in OpenD as SUBMITTED / FILLED_PART.

- Training assumes **instant fill at last price**. Live is a **limit at snapshot**.
- V2 **adds shares and moves cash on submit**. If nothing fills, pickle and Futu **diverge**. Dashboard PENDING is supposed to show this; **cash should not treat PENDING as a fill** (`isRealFill`).
- **OpenD will not cancel leftovers by itself** while the session is open. Day orders **may** die at **session close**; intra-day they stay. We already have `FutuPaperBroker.cancel_order` (`modify_order` + `CANCEL`). **V2 loop never times them out.** V3 `decide_order` can cancel/replace when the new target disagrees.

**What the user wants:** model thinks “instant”; market is slow → **cancel stale working orders** (e.g. still open after N minutes or next completed 10-min bar) then **re-decide** from current holdings **as OpenD reports them**, not as pickle hoped.

---

## 6. Why size is tiny and round-trips look dumb

- Action is a **portfolio weight**, not a “trade size.” `target_shares = action * equity / price`. COST ~$900: `0.01 * 1e6 / 900 ≈ 11` shares. Noise around 0 → **7 buy, 14 sell**.
- US lot = 1, so those trades **go through**. HK lot = 100, so similar noise **does nothing**.
- Fees are **not** in the env. V3 `decide_order` skips sell if `px <= last_buy` (and buy if `px >= last_sell`). **Wire this into V2 / V2.11** before blaming the policy.
- The policy **does not observe** working qty, last fill price, or “available to trade.” Inventory block is holdings ratio + cash only.

**Improvements (execution first, then optional obs):**

1. Port `decide_order` + last_buy/last_sell + cancel/replace into live V2 **or** only into V2.11 paper (safer).
2. Min notional (e.g. skip if `|delta| * px < N` HKD/USD) and/or min action step.
3. Stale cancel: if working > 1 completed bar (or > T minutes), cancel, sync positions from OpenD, then allow a new decision.
4. Later: extra obs for pending qty/side/age and last fill px (new dim = **cannot** warm-start V2 zip — that is v2.12).

The **model** cannot “know how much it can control” until the **env and broker** expose cash, lots, ticks, pending, and HK-long. Clipping only at Futu is too late.

---

## 7. Implementation order for the next chat

Do **A** before **B**. Do not mix live V2 weights.

### A. Execution (can land on live V2 carefully, or only on the fork)

1. **Session logs** — already in this repo (see §8). Copy `src/utils.py`, `src/inference.py`, `execution/run_trader.bat` to the VM. Restart trader. Confirm `logs/trader_*.txt`.
2. **Reject taxonomy** — when user pastes a new Futu string, add it to a table in code (`classify_place_error`) and never silent-fail.
3. **Do not book on reject.** Already true. **Stop booking on submit** for the new paper path (V3-style: book on fill). Live V2 still books on submit — changing that mid-session will desync CATL; do it on the fork first.
4. **`decide_order` on the live loop** (or v2.11 only): skip underwater round-trip; cancel/replace working.
5. **Stale cancel** after 1 bar / N minutes; then `accinfo` + `positions` before the next place.
6. **Min share / min notional** so COST/KO stop flickering 7 vs 14.

### B. V2.11 hybrid train

1. Isolated env + trainer + `models/news_gpu_v2_11/`.
2. Seed from **current VM paper zip** (banner path).
3. HK clip in `step`/`_rebalance` only; keep `[-1,1]` Box.
4. Fine-tune 1 window/day after US close (`--end` complete session).
5. `--predict-now` on v2.11. Confirm HK actions logged ≥ 0 after clip; US may be negative.
6. Only then discuss switching `run_trader.bat`.

### C. Later (optional)

- HSI observer (new obs dim — new zip family, not a V2 warm-start).
- Error-conditioned “compensation policy” (rule-based first; do not wait for RL to learn OpenD error codes).

---

## 8. Session logs (implemented 2026-08-25)

On `python -m src.inference` (and V3 loop):

- Creates **`logs/trader_YYYYMMDD_HHMMSS.txt`** (HKT stamp).
- **Appends** every `airaire.*` INFO/WARNING/ERROR for that process (same format as the console).
- **Flushes** after each dashboard push so a kill still leaves the last cycle.

Copy to VM with the trader: `src/utils.py`, `src/inference.py`, `execution/run_trader.bat`. Restart `run_trader.bat`. Path is printed at startup. User: when a Futu warning flashes, **leave the trader up** and send that `logs/trader_*.txt` (or the `place_order failed` lines).

Also: `data/logs/trades.jsonl` is fills only, not rejects.

---

## 9. Files map

| Path | Role |
| :--- | :--- |
| `src/inference.py` | Live V2 paper. Book on submit. No `decide_order`. |
| `src/order_lifecycle.py` | Pending vs fill; `decide_order` (used by V3). |
| `src/utils.py` | Lots, ticks, `attach_session_file_log`. |
| `src/finetune_latest.py` | Daily V2 1-window into `news_gpu_v2`. |
| `src/trading_env.py` | Live V2 env `[-1,1]`, 782-dim. |
| `src/train_gpu_v2_10.py` + `trading_env_v2_10.py` | Long-only 5 + HSI. Isolated. Not live. |
| `execution/run_trader.bat` | Live V2. |
| `execution/run_finetune.bat` | Live V2 daily FT (starts OpenD). |
| `guide/NEXT-CHAT-HK-US-HYBRID.md` | This file. |

---

## 10. Prompt to paste into the new chat

```
Read guide/NEXT-CHAT-HK-US-HYBRID.md first.

Constraints:
- Do not edit live src/trading_env.py or write models/news_gpu_v2 except as the user explicitly allows.
- Do not load V2 zips into V2.10.
- V2.11 = 782-dim, Box [-1,1] head, HK clipped to >=0 in step/rebalance, US free, new model dir.
- Execution: session logs already exist; next is decide_order + stale cancel + no book-on-submit on the fork; min notional; classify Futu place_order failed strings.
- Confirm the seed zip from the GPU trader banner, not from a guessed checkpoint_2026-08-20 filename.
```
