# Training Results Review — 2026-08-19

> **Status:** GPU run finished (118/118). CPU run stopped at window 8.  
> **Verdict:** Do **not** paper-trade `best_model.zip`. Sharpe selection is broken (every window = `0.0000`). Return / max-drawdown still differentiate checkpoints. Use nominated mid-run GPU zips for paper trade; fix `_returns` before the next full retrain.

---

## 1. What we ran

| Run | Log | Coverage | Hardware | Speed | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU** | `models/news-vm-cpu/log_20260819_vm_cpu.txt` | Windows **1–8** of 118 | CPU, SB3 1.7 | ~28 fps, ~14 min/window | Stopped (too slow). Stronger **in-sample** returns on the overlapping early windows. |
| **GPU** | `models/news-gpu/log_20260819_vm_gpu.txt` | Windows **1–118** | NVIDIA A40-2Q (2 GB), SB3 2.0 | ~900–1200 fps, ~1 min/window | Finished in ~2 hours (19:13 → 21:12). Policy **entropy-collapsed** by the last windows. |

Shared setup (from `PLAN.md` / `train.py` / `train_gpu.py`):

- News-integrated PPO, 30 session-day rolling windows, step = 1 day.
- Panel: 2026-02-24 → 2026-08-18, 7 tickers, 10-minute bars, `news_coverage=100%`.
- Eval printed as `return / sharpe / max_dd / equity`.
- `best_model.zip` copied from the checkpoint with the highest Sharpe.

CPU and GPU checkpoints are **not interchangeable** (SB3 1.7 vs 2.0). Paper trade must use GPU zips.

---

## 2. The Sharpe = 0 bug (why Window 1 “won”)

You are right: **every window logged `sharpe=0.0000`**, so `best_model.zip` is Window 1 only because the selector is:

```text
best_sharpe starts at -inf
if sharpe > best_sharpe:  # 0.0 > -inf  → Window 1 wins
                          # 0.0 > 0.0   → later windows never replace it
```

GPU log at the end of training:

```text
Best Sharpe=0.0000 -> copied checkpoint_2026-03-30.zip to models\news_gpu\best_model.zip
```

That zip is one of the **worst** GPU checkpoints (`return=-1.57%`, `max_dd=2.3%`). **Do not paper-trade it.**

### Root cause

`evaluate_policy()` (and the PPO reward) uses `env._returns`. In `TradingEnv.step()` the bar return is computed **after rebalance, at the same prices**, then the clock advances:

```python
prices = self._current_closes()
prev_equity = self._mark_to_market(prices)
self._rebalance(action, prices)
equity = self._mark_to_market(prices)          # same prices
step_return = (equity - prev_equity) / prev_equity
self._returns.append(step_return)
self._equity_curve.append(equity)
self._bar_index += 1                           # price move happens *after* the return is recorded
```

There are **no commissions**. Same-price rebalance → `step_return ≈ 0`. If `std < 1e-12`, Sharpe is forced to `0.0`.

**Equity curve is still valid.** Cumulative return and max drawdown come from `_equity_curve`, which *does* include the next bar’s price move. That is why P&L looks real (`-21%` to `+53%`) while Sharpe stays `0.0000`.

The same bug hits the **reward**: PPO mostly sees `-λ × drawdown`, not P&L. `ep_rew_mean` stays around `-140` to `-186` even on windows that made money.

`PLAN.md` Phase 3 gate (**Sharpe > 0.5 on a test set**) cannot be used until Sharpe is computed from the equity curve (daily or annualized).

---

## 3. GPU training health — finished, but the policy collapsed

GPU speed beat the `GPU-TRAIN-CHANGE.md` target (200–500 fps → **~1000 fps**). Each window was under-trained:

| | CPU | GPU |
| :--- | :--- | :--- |
| `n_steps` × `n_envs` | 2048 × 1 | 4096 × 4 = **16384** rollout |
| Timesteps / window | ~18.5k | ~19k |
| PPO iterations / window | **~9** | **2** |
| Net | `[256, 256]` | `[512, 512]` |

The GPU run finished 118 windows by doing **two updates per window**. Sequential fine-tuning then collapsed the policy:

| Metric | Window 1 | Window 118 |
| :--- | ---: | ---: |
| action `std` | 0.997 | **0.243** (almost deterministic) |
| `entropy_loss` | -7.08 | **-0.039** |
| `approx_kl` | 0.007 | **0.56** (clip range is 0.2) |
| `clip_fraction` | 0.05 | **0.70** |

Windows **~100–118** are mostly flat-to-negative with **10–20%** drawdowns. The last checkpoint is a collapsed policy, not a “fully trained” brain.

---

## 4. CPU vs GPU on overlapping windows (in-sample)

CPU, with more updates per window, looked **stronger on the same early dates**. Metrics are still in-sample (scored on the window just trained).

| Window | End date | CPU return / max DD | GPU return / max DD |
| ---: | :--- | :--- | :--- |
| 1 | 2026-03-30 | +4.4% / 2.3% | **-1.6% / 2.3%** |
| 4 | 2026-04-02 | **+19.7% / 5.1%** | +2.7% / 2.6% |
| 6 | 2026-04-06 | **+20.2% / 6.9%** | -0.1% / 3.8% |
| 8 | 2026-04-08 | **+34.6% / 5.7%** | +2.9% / 3.2% |

CPU Window 7 (`+18.9%`) is in the log but missing from `training_log_cpu.csv` (minor write glitch).

Use GPU artifacts for paper trade. Keep the CPU numbers only as evidence that **more PPO updates per window help**.

---

## 5. Nominated GPU checkpoints (ignore Sharpe)

Ranked by **Calmar = return / max drawdown**. These are **in-sample** 30-day scores — candidates, not proof of out-of-sample edge.

### Mid-run peaks (preferred)

| Priority | Checkpoint | Window | Period | Return | Max DD | Calmar |
| ---: | :--- | ---: | :--- | ---: | ---: | ---: |
| **1** | `checkpoint_2026-06-17.zip` | 67 | 2026-05-13 → 2026-06-17 | **+38.8%** | **4.4%** | **8.8** |
| **2** | `checkpoint_2026-07-15.zip` | 89 | 2026-06-09 → 2026-07-15 | **+52.7%** | 6.4% | 8.3 |
| 3 | `checkpoint_2026-06-18.zip` | 68 | 2026-05-14 → 2026-06-18 | +24.6% | 6.6% | 3.7 |
| 4 | `checkpoint_2026-06-16.zip` | 66 | 2026-05-12 → 2026-06-16 | +26.3% | 7.3% | 3.6 |
| 5 | `checkpoint_2026-06-25.zip` | 73 | 2026-05-20 → 2026-06-25 | +28.9% | 8.8% | 3.3 |

Window 67 is the primary pick: high return, lowest drawdown among the peaks, entropy not yet dead (`entropy_loss` ≈ -4). Window 89 has the highest return but entropy was already falling (`≈ -2.3`).

### More recent (weaker, closer to “today”)

| Checkpoint | Window | Period | Return | Max DD |
| :--- | ---: | :--- | ---: | ---: |
| `checkpoint_2026-07-28.zip` | 100 | 2026-06-23 → 2026-07-28 | +10.3% | 13.2% |
| `checkpoint_2026-08-03.zip` | 106 | 2026-06-29 → 2026-08-03 | +6.8% | 8.4% |

A June-specialist deployed in late August is a distribution shift. These later zips are closer in time but already sitting on a collapsing policy.

### Do not use

- `best_model.zip` / `checkpoint_2026-03-30.zip` (Window 1, Sharpe-tie winner)
- Windows **110–118** (collapsed entropy, poor return / high DD)
- CPU zips for the GPU inference stack (SB3 version mismatch)

---

## 6. GPU return path (context)

Rough regimes from the 118-window GPU log:

| Windows | Calendar (approx.) | Typical in-sample result |
| :--- | :--- | :--- |
| 1–9 | Feb 24 → early Apr | Small positive / mixed (`-2%` to `+4%`) |
| 10–28 | Apr | Often **+6% to +19%**, DD ~4–8% |
| 31–56 | late Apr → May | Drawdown cluster (`-5%` to `-21%`, DD up to 24%) |
| **64–89** | May → mid-Jul | **Peak cluster** (W67 +39%, W89 +53%) |
| 90–118 | mid-Jul → Aug 18 | Deterioration; many negatives; entropy collapse |

Window 118 (most recent slice, 2026-07-15 → 2026-08-18): `return=+3.1%`, `max_dd=13.5%`, `approx_kl=0.56`.

---

## 7. Recommendation

**You can train better** — and it is worth a later GPU pass. **That should not block paper trade** as a systems test.

### Paper trade now (`PLAN.md` Phase 4)

Phase 4 is “load a model and run the Futu `SIMULATE` loop” (`inference.py`, `state.pkl`, news poller, Telegram), not “the brain is proven.”

1. Point inference at **`checkpoint_2026-06-17.zip`** (primary).
2. Keep **`checkpoint_2026-07-15.zip`** as backup.
3. Do **not** use `best_model.zip`.

### Retrain / eval fixes (before the next full 118-window run)

1. Put **price P&L into `_returns`** — mark holdings at the *next* bar, or use `pct_change` of `_equity_curve`. This fixes Sharpe, reward, and `best_model` selection.
2. Rank by **Calmar or equity-curve Sharpe**, with a return tie-break. Never promote a checkpoint solely because Sharpe is `0.0`.
3. **Walk-forward eval:** train on window *t*, score on the next day / next window. `PLAN.md` asked for a held-out month and Sharpe > 0.5. We do not have that yet.
4. GPU: keep the speed, but give each window **8–10 PPO iterations**, not 2.
5. Stop or reset when `approx_kl` > ~0.05 and entropy is collapsing. Windows 90+ of this run are not useful as a “latest = best” policy.

### Parallel (no retrain required)

Re-score existing GPU checkpoints from their **equity curves** (fixed Sharpe / Calmar). That can confirm Window 67 vs 89 without another 2-hour GPU bill.

---

## 8. Pointers

| File | Role |
| :--- | :--- |
| `guide_and_test/PLAN.md` | Master plan; Phase 3 Sharpe gate; Phase 4 paper trade |
| `guide_and_test/EVALUATION-POLICY-FIX.md` | Earlier eval blow-up (annualization); current Sharpe=0 is a *new* bug |
| `guide_and_test/GPU-TRAIN-CHANGE.md` | GPU knobs (`n_steps`, `n_envs`, net width) |
| `src/trading_env.py` | `_returns` vs `_equity_curve` in `step()` |
| `src/train.py` | `evaluate_policy()`, `best_sharpe` copy |
| `src/train_gpu.py` | GPU loop; 118 windows → `models/news_gpu/` |
| `src/inference.py` | Paper trader; currently loads `best_model.zip` |
| `models/news-gpu/log_20260819_vm_gpu.txt` | Full GPU log |
| `models/news-vm-cpu/log_20260819_vm_cpu.txt` | CPU log (stopped at window 8) |
| `models/news-vm-cpu/training_log_cpu.csv` | CPU metrics CSV (window 7 row missing) |
