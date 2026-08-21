# GPU v2 resurrection notes (entropy collapse → Window 113/118)

> **Date:** 2026-08-21  
> **Status:** Tasks A/B/C below are **done**. Do not treat this file as a to-do list.  
> **Paper trading / daily ops / full timeline:** [`PHRASE-4-EXECUTION-&-DAILY-WORK.md`](PHRASE-4-EXECUTION-&-DAILY-WORK.md)

Keep this page for the entropy-collapse / resurrection story only.

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

## 3. Immediate Action Items — all done (21 Aug)

See `PHRASE-4-EXECUTION-&-DAILY-WORK.md` for how these actually work now (live_best, Telegram Promote, 60s poll / 10-min orders, closed-market keep).

### Task A: Set the Inference Model — DONE
- Loads `models/news_gpu_v2/best_model.zip` with a startup banner.

### Task B: Create `src/finetune_latest.py` — DONE
- Latest 1–3 windows, GPU v2 settings, Promote/Keep vs live Calmar.

### Task C: Catch-up logic — DONE
- Futu klines, seek, restore `state.pkl`, no orders during catch-up.