# 🧠 Task for Cursor (Grok 4.6) : Create Optimized GPU Trainer `train_gpu_v2.py`

## Context

We have a GPU training script `src/train_gpu.py` that finished 118 windows but had issues:

- **Under‑training**: Only ~1 PPO update per window (due to `n_steps=4096, n_envs=4`), leading to policy collapse after window ~67.
- **Sharpe always 0**: `step_return` was calculated at the same price bar, making `_returns` all zeros.
- **Log overwrite**: `training_log.csv` was overwritten each run instead of appended.
- **No terminal log file**: We had to manually copy-paste terminal output.

We now have a new VM hardware profile (from Task Manager, in no load running state):

- **CPU**: 4 vCPUs @ 2.59 GHz, utilisation ~41% (headroom)
- **RAM**: 16 GB, 8.4 GB available
- **GPU Memory**: 2 GB dedicated + 8 GB shared (total 10 GB)
- **Disk**: HDD with high I/O (95% active) – ensure data is cached in RAM.

We want to retrain with a more balanced configuration that gives **~8 PPO updates per window**, and fix the Sharpe, logging, and terminal capture issues. We'll keep the original `train_gpu.py` untouched for history.

---



## 🎯 New File: `src/train_gpu_v2.py`

Create a new file by copying `src/train_gpu.py` and applying the following modifications.

### 1. Hyperparameters (top of file)

Change these constants:

```python
PPO_N_STEPS = 2048
PPO_N_ENVS = 6
# Leave PPO_BATCH_SIZE = 256 (or 128 if you encounter OOM)
```



### 2. Force 8 PPO Updates per Window

Modify `timesteps_for_window()` function:

```python
def timesteps_for_window(df: pd.DataFrame, epochs: int, cfg: GpuPpoConfig) -> int:
    n_bars = int(pd.to_datetime(df["datetime"]).nunique())
    episode = max(n_bars - LOOKBACK_BARS, 1)
    # Force 8 updates per window
    desired_updates = 8
    rollout_size = cfg.n_steps * cfg.n_envs
    desired_steps = desired_updates * rollout_size
    # Keep the old epoch-based logic as a lower bound
    return max(desired_steps, episode * epochs)
```

*(This ensures each window trains for exactly 8 PPO updates, regardless of* `--epochs`*.)*

### 3. Terminal Output Redirection (Log to .txt)

Inside `train()` function, right after `def train(...):` and before any other code, add:

```python
import sys
from datetime import datetime

# Create logs directory
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_filename = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(log_filename, 'w', encoding='utf-8')
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)
```



### 4. Default Output Directory

Change the default output path to `models/news_gpu_v2` (or keep `NEWS_GPU_MODELS_DIR` but ensure it's not the same as before). In `parse_args()`, set `default=NEWS_GPU_MODELS_DIR` but we can leave it as is and pass `--output models/news_gpu_v2` during run.

---



## 📁 Required Changes in Other Files



### `src/trading_env.py` – Fix Sharpe Calculation

In the `step()` method, replace the return calculation block so that `_returns` reflects the **change in equity from the current bar to the next bar**. Full modified snippet:

```python
def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
    # ... (action clipping, leverage, etc. unchanged)

    # ---- Get current prices and equity ----
    prices_current = self._current_closes()
    prev_equity = _safe_float(self._mark_to_market(prices_current), self._last_equity)

    # ---- Rebalance based on current prices ----
    self._rebalance(action, prices_current)

    # ---- Advance bar index ----
    self._bar_index += 1
    terminated = self._bar_index >= len(self.datetimes) - 1

    # ---- Get next prices and new equity ----
    prices_next = self._current_closes()
    equity = _safe_float(self._mark_to_market(prices_next), prev_equity)

    # ---- Compute step return ----
    step_return = (equity - prev_equity) / max(abs(prev_equity), 1e-9)
    step_return = _safe_float(step_return, 0.0)
    self._returns.append(step_return)
    self._equity_curve.append(equity)
    self._last_equity = equity

    # ---- Reward and observation ----
    reward = self._sharpe_drawdown_reward()
    obs = self._get_obs()
    info = {
        "equity": equity,
        "cash": self._cash,
        "holdings": dict(zip(CORE_TICKERS, self._holdings.tolist())),
        "action": dict(zip(CORE_TICKERS, action.tolist())),
        "datetime": self._current_dt(),
        "reward_sharpe_dd": reward,
    }
    return obs, float(reward), terminated, False, info
```

*(Make sure the original* `self._bar_index` *increment is removed from the end of the function.)*

### `src/train.py` and `src/train_gpu_v2.py` – Append Logging

Modify `_save_log()` function to append new rows instead of overwriting:

```python
def _save_log(rows: list[WindowMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([r.__dict__ for r in rows])
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(path, index=False)
    else:
        new_df.to_csv(path, index=False)
    logger.info("Wrote training log -> %s (rows=%d)", path, len(combined) if path.exists() else len(new_df))
```

*(Apply this same change in both* `train.py` *and* `train_gpu_v2.py`*.)*

---



## 🚀 How to Run the New Trainer

After creating `train_gpu_v2.py`, run:

```bash
python -m src.train_gpu_v2 --output models/news_gpu_v2 --device cuda
```

(Optionally add `--test` to test a single window first.)

---



## ✅ Summary of Changes


| File                           | Change                                            | Purpose                                     |
| ------------------------------ | ------------------------------------------------- | ------------------------------------------- |
| `train_gpu_v2.py`              | `PPO_N_STEPS=2048`, `PPO_N_ENVS=6`                | Balance update frequency and parallelism.   |
| `train_gpu_v2.py`              | `timesteps_for_window()` with `desired_updates=8` | Force 8 PPO updates per window.             |
| `train_gpu_v2.py`              | Add `Tee` redirection in `train()`                | Save terminal output to `logs/train_*.txt`. |
| `trading_env.py`               | Fix `step()` to compute return on next bar        | Fix Sharpe ratio (now non‑zero).            |
| `train.py` & `train_gpu_v2.py` | Change `_save_log()` to append                    | Preserve all window records.                |


