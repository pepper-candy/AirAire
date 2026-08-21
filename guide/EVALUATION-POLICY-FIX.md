# 🧠 Cursor Task: Restart Training on VM with Fixes

## Context
We are migrating to a powerful virtual machine and need to **restart training from scratch** with a corrected evaluation pipeline. Our previous run produced absurd metrics (`return=2042`, `sharpe=20`, `equity=2 billion`) for Window 2, caused by a bug in `evaluate_policy()` inside `src/train.py`.

The **model weights themselves are healthy** (no NaNs, loss decreasing, explained_variance rising), but the evaluation logic is flawed, making it impossible to compare checkpoints. Since the new VM is much faster, restarting is the safest path.

---

## New Hardware (VM Specs)

- **GPU**: NVIDIA A40-2Q (2GB + 8GB) – CUDA capable
- **RAM**: 16 GB
- **CPU**: Intel Xeon Gold 6348 @ 2.60 GHz

This is ~10–20× faster than our previous CPU-only laptop. We will use GPU acceleration (`device="cuda"`).

---

## What We Need You to Do

### 1. Diagnose & Fix `evaluate_policy()`

Open `src/train.py` and locate the `evaluate_policy()` function. The current implementation over‑annualizes returns, producing nonsense for some windows. It may also suffer from environment state bleed‑through.

**Fixes to apply:**

- **Remove exaggerated annualization** – do not multiply by `np.sqrt(TRADING_DAYS * BARS_PER_DAY)` inside the Sharpe calculation. Use raw average / standard deviation of returns *within the window* (or use the existing annualization but ensure the frequency factor matches the actual data frequency). Since our data is 10‑minute bars, and we are evaluating a 30‑day window, the simplest safe approach is:
  ```python
  sharpe = rets.mean() / (rets.std() + 1e-9)   # no annualization
  ```
  If you want annualized, use `np.sqrt(len(rets))` (number of bars) instead of a fixed constant.

- **Reset environment state reliably** – ensure `TradingEnv.reset()` correctly re‑initialises `_equity_curve` and `_returns` to avoid contamination from previous calls. The function should return a clean environment for each evaluation.

- **Add sanity clipping** – after computing `cum_ret`, `sharpe`, and `max_dd`, clip them to reasonable ranges (e.g., `return` between -1 and 3, `sharpe` between -3 and 5). Log a warning if out of range.

### 2. Prepare for Fresh Training on GPU

- Modify `src/train.py` to accept `--device cuda` and default to `cuda` if available.
- (Optional) Increase `n_steps` or `batch_size` to leverage the A40, but keep them stable (we can tune later).
- Ensure the output directory is `models/news_v2` to separate from old runs.
- Keep all other hyperparameters as they were (they were working well – `explained_variance` was rising).

### 3. Run the Full Training

After fixing, execute:

```bash
python -m src.train --output models/news_v2 --device cuda
```

This will train 118 rolling windows (30 days each) on the 6‑month dataset. The training will produce checkpoints and a `training_log.csv` with accurate metrics.

---

## Expected Healthy Metrics

| Metric | Healthy Range |
|--------|---------------|
| `return` | -0.20 to +0.50 per window |
| `sharpe` | -1.0 to +2.0 |
| `max_dd` | 0.00 to 0.15 |
| `equity` | ~900k to 1.5M (starting from 1M) |

If you see anything drastically outside these ranges, please stop and debug further.

---

## Additional Checks

- Confirm **news data is being used** – the log should show `news_coverage=100%` and non‑zero news sample values.
- Ensure checkpoints are saved after every window.
- After training, copy the best model (based on Sharpe) to `models/news_v2/best_model.zip` automatically (as already implemented).
