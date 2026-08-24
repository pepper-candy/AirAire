# Phase 6 — what we actually built (V3 research)

[PHRASE-6.md](PHRASE-6.md) was the original ask (volume + HSI/SPX, isolated from live V2). This page is the source of truth for what shipped. V2 paper trade stays on `models/news_gpu_v2/best_model.zip`. Do not point `run_trader.bat` at V3.

**Date:** 2026-08-24  
**Window:** same 6-month regime as V2 (`2026-02-24` → `2026-08-21`)  
**Obs dim:** 782 → **1082**. V2 zips cannot load.

---

## Files (V2 untouched)

| File | Role |
| --- | --- |
| `src/data_loader_v3.py` | Builds `data/enhanced/enhanced_v3.parquet` only |
| `src/trading_env_v3.py` | 7-name OHLCV lookback (5 core + HSI + SPX). Actions still 5 |
| `src/train_gpu_v3.py` | Same PPO recipe as V2. Writes `models/news_gpu_v3/` |

Not modified: `src/trading_env.py`, `src/inference.py`, `src/train_gpu_v2.py`, `src/finetune_latest.py`, `models/news_gpu_v2/`, `data/enhanced/enhanced_data.parquet`.

---

## Data design (not “replace Bloomberg with TradingView”)

Prices stay on the **Bloomberg / V2 clock**. TradingView and Futu only overlay **volume**. That keeps V3 comparable to V2 instead of inventing a new tape.

| Source | What it is used for |
| --- | --- |
| Bloomberg / V2 parquet | OHLC backbone for all 7 names |
| TradingView (6 CSVs) | Volume overlay, nearest 10 min |
| Futu OpenD | CATL (`HK.03750`) volume only, cache `data/raw/futu/v3/03750_HK_10min.csv` |
| V2 parquet `news_score` | Copied by default (same news as V2) |
| HSI TradingView | Volume cells are empty → HSI volume stays 0 |

**Ignore `TSE_DLY_3750`.** That is Tokyo GMO Internet, not HK CATL.

US volume also tries a **−12h** match because Bloomberg afternoon bars are stored as `3:50` not `15:50`. Session filter (`is_hk_market_open` / `is_us_market_open`) is DST-safe and applied to **volume sources only**. Do not session-filter Bloomberg US with those helpers — that drops the whole afternoon.

Successful panel on the GPU machine: **32,496 rows**, 7 tickers, span 2026-02-24 09:15 → 2026-08-21 16:00, `volume>0 ≈ 73.9%`. HK 700/3690 ~95% vol match; CATL 4064/4538; COST/KO/SPX ~76–79%; HSI 0% volume.

---

## Env + trainer

- Observation is V2 plus HSI/SPX OHLCV in the lookback cube. News, calendar, inventory, and the **5-name action** are unchanged.
- Trainer patches V2 helpers at runtime so we did not edit V2 files. `_env_thunk` lives in `train_gpu_v3.py` so Subproc workers import the V3 env.
- Refuses write to `models/news_gpu_v2`, `models/news`, `models/news_gpu`.
- Refuses `--init-checkpoint` from a V2 path.

**PPO health on the first GPU pass (windows 1–4, then killed):** `obs_dim=1082`, KL ≈ 0.011, entropy ≈ −7, ~870 fps, no OOM. Those in-sample Calmars (100–200, some hitting the clip) are **not** comparable to live 4.91.

---

## How we score (why in-sample 4.91 is the wrong bar)

Same-window Calmar is homework: `evaluate_policy(model, win.df)`. Fine-tune Promote vs `live_best.json` is also in-sample. Live paper trade is the only true out-of-sample.

V3 still **prints** in-sample so you can compare to the old log. It does **not** elect `best_model.zip` from it.

After each window:

1. **Holdout eval** — next **5 session days after** the train window (not inside it, not before it). Friday → Monday is included. Sat/Sun are already skipped. `seek_to_datetime` so the 200MA is valid.
2. **Holdout smooth** — `best_model.zip` is the rolling **median of the last 10 holdout returns**. First 3 holdouts cannot crown a winner. One Trump Monday or weekend gap cannot veto a useful policy.

Trust the median line for who gets the zip. Use the weekly holdout line to watch noisy weeks, not to delete the model.

Last windows with no future days get a flat score and cannot win.

---

## What Deepseek got wrong (do not “fix” back)

| Claim in [PHRASE-6.md](PHRASE-6.md) | What is true |
| --- | --- |
| Filter US to `13:30–20:00 UTC` | DST-blind. Use `is_hk_market_open` / `is_us_market_open` on volume sources only |
| TradingView is the new price tape | Bloomberg stays the clock; TV is volume only |
| HSI has volume | HSI TV volume cells are empty |
| `TSE_DLY_3750` is CATL | Japan, ignore it |
| Compare V3 Calmar to live 4.91 on the same protocol | Old protocol was in-sample. V3 holdout / paper trade is the fair test |
| 2-year retrain | Same 6-month regime as V2 (pre-election is a different market) |

---

## Run it (GPU / OpenD machine)

Copy these three files onto the train box (`airaire`, not this laptop-only tree if they differ):

- `src/data_loader_v3.py`
- `src/trading_env_v3.py`
- `src/train_gpu_v3.py`

Panel is already built if you see `data/enhanced/enhanced_v3.parquet`. Rebuild only if you must:

```text
python -m src.data_loader_v3 --force-rebuild --tv-dir PATH\TO\tradingview
```

Optional: overlay recent Alpha Vantage on the **existing** V3 parquet (does not rebuild prices, does not touch V2):

```text
python -m src.data_loader_v3 --news-days 14
```

**Do not** `--force-rebuild --force-news-fetch`. Full Feb–Aug AV refetch is slow and breaks the same-news control vs V2.

A news fetch cannot update a train already sitting in GPU memory. Overlay first, then start train.

Train from window 1 so `best_model` uses one selection rule:

```text
python -m src.train_gpu_v3 --device cuda
```

`--resume N` **redoes** window N (loads the **previous** zip). The first GPU pass died after window 4. If you still have those old in-sample zips and you want a clean election, start from 1. `--resume 5` keeps training but mixes old in-sample W1–4 with the new holdout rule.

Smoke (one window, no zips):

```text
python -m src.train_gpu_v3 --device cuda --test
```

CLI extras: `--holdout-days 5` (default), `--holdout-smooth 10` (default), `--skip-futu` (CATL cache only), `--resume N`.

---

## What you will see in the log

```text
In-sample (not used for best_model)  return=...
Holdout eval  YYYY-MM-DD → YYYY-MM-DD  steps=...  return=...
Holdout smooth n=...  median_return=...  median_calmar=...  (this week return=...)
```

`models/news_gpu_v3/` gets `checkpoint_YYYY-MM-DD.zip`, `training_log_history.csv`, and later `best_model.zip`. Live V2 folder is not written.

---

## Still not done

1. **No live V3 yet.** Inference still loads a 782-dim V2 zip and does not fetch HSI/SPX in the live poll. Building that path is a later job.
2. Do not promote on a log line. Paper-trade first, then decide.
3. Laptop `models/news_gpu_v2_20260823135426/` is a **manual backup**. Live goldens stay in `models/news_gpu_v2/` on the trader machine.
4. Judge V3 on holdout / paper, not in-sample 4.91.

---

## Why V3 at all

V2 env/train/inference use the five core names only. HSI/SPX were in the parquet and never entered `model.predict()`. Bloomberg has no usable volume, so the volume channel was mostly zeros. Changing obs dim requires this isolated retrain.
