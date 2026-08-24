# 📄 Cursor Sync: Phase 6 — Data Upgrade & Retraining (V3)

> **Superseded as the build spec.** What we actually shipped, and how to run it, is [`PHRASE-6-IMPLEMENTED.md`](PHRASE-6-IMPLEMENTED.md). Keep this file as the original ask (2026-08-24). Several items below were wrong or unsafe (UTC session filter, TradingView as the price tape, TSE 3750 as CATL, compare in-sample Calmar to live 4.91).
>
> **Date:** 2026-08-24 (Monday)  
> **Priority:** **DO NOT TOUCH V2 FILES.**  
> **Status:** Paper trading launched at 09:30 HKT using `models/news_gpu_v2/best_model.zip` (Calmar 4.91). This V3 task runs **in parallel** without interfering.

---

## 1. Background (Why V3?)

We have received **6 new 10-minute CSV files** from TradingView (via a friend). They contain **Volume** data, which our current Bloomberg/Futu data lacks. Volume is a critical feature for liquidity and conviction.

- **Current V2:** Uses 5 core stocks, ignores volume (set to 0), ignores observers (HSI/SPX). 
- **Future V3:** Will incorporate **Volume** and **include HSI/SPX** in the observation space. This requires a full retrain.

**Crucial Constraint:**
- The user is running `run_trader.bat` right now (09:30 HKT) using **V2** models. 
- **Do not modify** `src/trading_env.py`, `src/inference.py`, `src/train_gpu_v2.py`, or `models/news_gpu_v2/`. 
- **Create isolated V3 files** so we can develop and test without risking the live paper trade.

---

## 2. The 7 TradingView Files (What we have)

Location: `C:\Users\mongk\Desktop\airaire_fixing\data\raw\tradingview\`

| File | Corresponding Ticker | Role | Status |
| :--- | :--- | :--- | :--- |
| `BATS_COST, 10_ee846.csv` | `US.COST` | Core Trade | ✅ Has Volume |
| `BATS_KO, 10_e396d.csv` | `US.KO` | Core Trade | ✅ Has Volume |
| `HKEX_DLY_700, 10_46fa9.csv` | `HK.00700` | Core Trade | ✅ Has Volume |
| `HKEX_DLY_3690, 10_ccc44.csv` | `HK.03690` | Core Trade | ✅ Has Volume |
| `HSI_HSI, 10_1a47d.csv` | `HK.HSI` | Observer | ✅ Has Volume |
| `SP_DLY_SPX, 10_beeae.csv` | `US.SPX` | Observer | ✅ Has Volume |
| ~~`TSE_DLY_3750, 10_48c1d.csv`~~ | ~~`HK.03750` (CATL)~~ | Core Trade | ❌ **WRONG DATA (Japan)** |

**Action Required for CATL:** The friend accidentally exported `TSE_DLY_3750` (Tokyo Stock Exchange, GMO Internet). We must fetch `HK.03750` ourselves via **Futu API** (which supports volume) or ignore it for now. **Do not use the Japan file.**

---

## 3. The Data Problem (Extended Hours)

All TradingView CSVs use **Unix timestamps (UTC)**. Crucially, they include **Post-Market (Extended Hours)** data.

- **HK Regular Session:** 09:30 – 16:00 HKT → **01:30 – 08:00 UTC**
- **US Regular Session:** 09:30 – 16:00 ET → **13:30 – 20:00 UTC**

**Requirement:** When loading these CSVs in V3, we must **filter out** bars whose `time` (converted to UTC time-of-day) falls outside these regular session windows. 
*Reason:* Training data (Bloomberg) only contained regular session bars. Including extended hours would create an Out-of-Distribution (OOD) shift, breaking the strategy during live regular-session trading.

---

## 4. Technical Requirements (Build V3, Isolated)

Create new files to keep V2 trading safe:

### A. New Data Loader: `src/data_loader_v3.py`
- Load the 6 valid TradingView CSVs (and the Futu-fetched CATL CSV).
- Convert Unix timestamps to **naive UTC** datetimes.
- Apply a **`filter_regular_session(df, market)`** function:
  - `market="HK"`: Keep `01:30 <= time_utc < 08:00`.
  - `market="US"`: Keep `13:30 <= time_utc < 20:00`.
- Merge them into the standard `STANDARD_COLUMNS` (datetime, ticker, open, high, low, close, **volume**). 
- Save as `data/enhanced/enhanced_v3.parquet` (do not overwrite V2 parquet).

### B. New Environment: `src/trading_env_v3.py`
- Copy `trading_env.py` but modify the state space:
  1. **Volume:** Fill `volume` in `_ohlcv_cube` using the new TradingView data (no more zeros).
  2. **Observers (HSI/SPX):** Add them to the observation.
     - Expand `OHLCV_FIELDS` or create a separate `macro` block.
     - The observation dimension will increase. This is intentional.
- Leave `trading_env.py` untouched for the live trader.

### C. New Trainer: `src/train_gpu_v3.py`
- Copy `train_gpu_v2.py` but import `trading_env_v3` and `data_loader_v3`.
- Output directory: **`models/news_gpu_v3/`**.
- Run a **full 118-window retrain** (no `--resume`). Compare Calmar against V2's 4.91.

---

## 5. Cursor Task List

1. **Write `src/data_loader_v3.py`**: 
   - Load and filter TradingView CSVs.
   - Add a placeholder function for missing CATL (we can fetch it later, but ensure the pipeline doesn't break).
2. **Write `src/trading_env_v3.py`**: 
   - Include Volume and Observers in the state.
   - Ensure `observation_dim()` is updated.
3. **Write `src/train_gpu_v3.py`**: 
   - Run on `--device cuda`.
   - Save checkpoints to `models/news_gpu_v3/`.
4. **Do NOT touch** `src/inference.py`, `src/trading_env.py`, `src/train_gpu_v2.py`, or `models/news_gpu_v2/`.