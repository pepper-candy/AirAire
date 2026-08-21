# 📄 Sync Document for Cursor (Grok 4.6)

> **Date:** 2026-08-21  
> **Project:** AirAire - AI Quant Agent  
> **Status:** Phase 3 (Training) Optimized. Ready for Phase 4 (Paper Trading).  
> **Repository:** `C:\Users\mongk\Desktop\airaire\`

---

## 1. 🧠 Key Discoveries & Learnings (The "Brain Death" Concept)

We have identified and successfully resolved a critical flaw in our second GPU training run (`train_gpu_v2`).

### The "Entropy Death" Problem
- In the first 118-window run, the model's **entropy** collapsed around Window 90.
- **Symptoms:** `entropy_loss` dropped from `-7` to `-26` (meaning entropy → 0), and `approx_kl` spiked to `0.56` (indicating the policy was jumping chaotically). The model lost its ability to "explore" new market patterns.
- **Result:** The first run of Windows 113-118 performed poorly (Calmar ~1.37).

### The "Resurrection" (Our Fix)
- We ran `train_gpu_v2 --resume 113`, which **loaded the healthy, high-entropy weights from Window 112** and re-trained Windows 113-118.
- **Result:** Performance skyrocketed.
  - **Old Window 118:** Return 12.2%, Calmar 1.37.
  - **New Window 118:** Return 16.4%, Calmar **1.83**.
  - **New Window 113:** Return 18.0%, Max DD 8.7%, Calmar **2.05**.

**Takeaway:** In non-stationary environments (financial markets), it is standard practice to "roll back" to a healthy checkpoint when the policy collapses. We do not retrain from the "dead" period (Window 90). We only move forward from the last known "healthy" brain.

---

## 2. 🏆 The New Champion Models (Clean Folder)

The user has cleaned the `models/` folder. Only these two essential checkpoints remain in `models/news_gpu_v2/`:

1. **`checkpoint_2026-08-12.zip`** (New Window 113, also copied as  best_model.zip)
   - **Role:** **Best Model for Paper Trading.**
   - **Metric:** Calmar = **2.05** (Highest stability).
   - **Action:** Copy this file to `models/news_gpu_v2/best_model.zip`.

2. **`checkpoint_2026-08-18.zip`** (New Window 118)
   - **Role:** **Start Point for Future Training.**
   - **Metric:** Calmar = 1.83, but most recent date.
   - **Action:** Use this as the base for running Window 119+.

---

## 3. 🎯 Immediate Action Items for Cursor

### Task A: Set the Inference Model
- **File:** `src/inference.py`
- **Action:** Ensure it loads `models/news_gpu_v2/best_model.zip`.
- *(Optional but safe)*: Add a fallback print statement so the user knows exactly which checkpoint is loaded on startup.

### Task B: Create `src/finetune_latest.py` (Incremental Daily Update)
Instead of running the heavy 118-window full training every day, we need a **lightweight daily fine-tune script**.

- **Function:** Load the latest checkpoint (e.g., `checkpoint_2026-08-18.zip`), run training on **the latest 1-3 windows** (e.g., Window 119), and save a new checkpoint.
- **How it works:**
  1. Load `enhanced_data.parquet`.
  2. Slice the last 30 days of data of that day. (update from futu api daily or on missing days if user didn't go online for days.)
  3. Load the current best model (or latest checkpoint).
  4. Run PPO for a few updates (as if the latest training settings).
  5. Save to `models/news_gpu_v2/finetuned_{date}.zip`.
- **Why:** This takes ~2-3 minutes on the GPU VM, perfect for the user's limited daily time window.

### Task C: Validate `inference.py` "Catch-Up" Logic
- The user is worried about being offline for hours (e.g., logging in at 12:45 PM).
- **Fix:** In `inference.py`, when the bot starts, it should:
  1. Load `state.pkl`.
  2. Fetch the latest OHLCV from Futu.
  3. Advance the `TradingEnv`'s internal `_bar_index` to match the latest time WITHOUT placing orders (pure state catch-up).
  4. Then enter the normal trading loop.