# 📄 Cursor Sync Document — Phase 4 Execution & Daily Workflow (2026-08-21)

## 🧠 Core Philosophy Alignment

We have refined our understanding of the "10-minute bar" execution strategy. The following points represent the **final, agreed-upon logic** for the project.

### 1. Polling Frequency (`--poll-seconds 60` vs `600`)
- **Initial thought:** 600 seconds (10 min) is aligned with training data.
- **Refined logic (CORRECTED):** 60 seconds is strictly better. 
- **Why:** 
    - The decision is **always** based on the *last completed 10-minute bar*. 
    - At 9:01, the decision uses the 9:00 bar close. At 9:09, the decision still uses the 9:00 bar close. 
    - Executing at 9:01 captures the price closer to the 9:00 close. If the model predicts UP, buying earlier is better. If it predicts DOWN, selling earlier is better.
    - **Conclusion:** `--poll-seconds 60` minimizes slippage and is now the **default** for `run_trader.bat`. We do NOT poll every 10 minutes; we poll every 1 minute to catch the bar close ASAP.

### 2. Training vs. Live Execution Timing
- **Training:** The `TradingEnv.step()` moves from bar `t` to `t+1` **instantly**. There is zero artificial delay.
- **Live:** The 60-second delay is purely for API data fetching and order placement. It does NOT change the model's perception of time; it just ensures we execute as close to the bar close as possible.

### 3. News Timeliness
- News is fetched via Alpha Vantage every poll cycle (every 60 seconds).
- If news drops at 10:05, the bot sees it at 10:06 and adjusts the action immediately. 
- **User Note:** We tested 1-second polling and it works, but 60 seconds is the optimal balance to respect rate limits while capturing the 10-minute bar close.

### 4. Model Capabilities (Boundaries)
- **Can do:** Hold (0), Adjust allocation (weights -1 to 1), Long/Short (positive/negative), Hedge (HK vs US correlation).
- **Cannot do:** Look at stocks outside the predefined 5 `CORE_TICKERS` + 2 Observers.

---

## 🏆 Promotion Workflow (No CSV Checking Required)

We have eliminated manual CSV checking for daily model promotion.

**Terminal Dashboard:**
Every time `finetune_latest.py` finishes, it prints a clear board:

```
PROMOTION CHECK (best_model.zip is never overwritten)
  Trading golden  W113  end 2026-08-12  Calmar 2.0539  (best_model.zip)
  Training seed   W118  end 2026-08-18  Calmar 1.8329  (checkpoint_2026-08-18.zip)
  This fine-tune  W119  …  Calmar x.xxxx  (finetuned_YYYY-MM-DD.zip)
  Verdict: BEATS Window 113 … copy the zip yourself
        or KEEP best_model.zip … No Telegram
```

**Telegram Alert (Conditional):**
- **Trigger:** Only when the new Calmar **> 2.0539**.
- **Action:** Sends a message with the zip name. Best model is **not** automatically copied. The user must manually copy the file if they agree.
- **Requirement:** `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`.

**CSV Structure (`finetune_log.csv`):**
- Appended each run.
- W113 and W118 are **pinned** as the first two rows of this CSV for easy reference.
- `training_log_history.csv` remains untouched (pure chronological history).

---

## 🚀 User Daily Routine (Batch Files)

The user operates via remote desktop and prefers double-clicking `.bat` files over typing commands. 

**1. `run_trader.bat` (Continuous Trading)**
```batch
@echo off
title AirAire - Paper Trader (Optimized)
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
start "" "C:\Users\klmong\Desktop\Futu_OpenD_10.10.7008_Windows\FutuOpenD.exe" -lang=en
timeout /t 5 /nobreak > nul
python -m src.inference --poll-seconds 60
pause
```

**2. `run_finetune.bat` (Daily Model Update)**
```batch
@echo off
title AirAire - Daily Fine-Tune
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
start "" "C:\Users\klmong\Desktop\Futu_OpenD_10.10.7008_Windows\FutuOpenD.exe" -lang=en
timeout /t 5 /nobreak > nul
python -m src.finetune_latest --windows 1 --device cuda
pause
```

**3. `test_inference.bat` (Safe Dry-Run)**
```batch
@echo off
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
python -m src.inference --dry-run --once
pause
```

---

## ✅ Codebase Verification

Cursor's latest changes are approved:
- `src/inference.py`: Has state catch-up, 60s polling logic, and loads `best_model.zip`.
- `src/finetune_latest.py`: Appends to CSV, prints promotion board, sends Telegram alerts when warranted.
- Protected zips (`best_model.zip`, `checkpoint_2026-08-12.zip`, `checkpoint_2026-08-18.zip`) are never overwritten.

---

## 💬 Final Word

The user has demonstrated excellent intuition by correcting the polling logic and understanding the separation between training data and execution latency. The system is ready for full paper trading. Ensure that all code adheres to the "state.pkl" resume contract and that no orders are placed when markets are closed.