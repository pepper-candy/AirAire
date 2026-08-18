# Cursor Sync — AirAire Project Status

> **Last Updated**: 2026-08-18  
> **Project**: AI Quant Agent (Papertrade) — "The Volatility Harvester"  
> **Repository**: `C:\Users\mongk\Desktop\airaire\`

---

## ✅ What We Have Completed (Phase 1 & Infrastructure)

### 1. Python Environment
- All dependencies installed via `requirements.txt`
- Python 3.12 (compatible with all packages)
- Virtual environment configured

### 2. Futu OpenD Integration
- Connection tested successfully
- Paper-trade (`TrdEnv.SIMULATE`) confirmed working
- Sample order placed (order_id: 8870552)
- Reference: `paper-trade-test.py`

### 3. Alpha Vantage API (Academic Full Tier)
- **Status**: ✅ Approved and tested
- **API Key**: Added to `.env` (`ALPHAVANTAGE_API_KEY`)
- **Test Results**:
  - COST: 50 news items, avg sentiment `0.019`
  - KO: 50 news items, avg sentiment `0.02`
  - TCEHY (Tencent ADR): 0 news (expected for HK stocks via ADR ticker)
- **Test Script**: `alpha-vantage-test.py` (keep for reference)

### 4. Data Pipeline (Fully Operational)
- **Source**: Bloomberg Terminal (`GIP` → `10 Min` → `02/24/2026` to `08/18/2026`)
- **Assets** (7 tickers):
  - `HK.00700` (Tencent)
  - `HK.03690` (Meituan)
  - `HK.03750` (CATL)
  - `US.COST` (Costco)
  - `US.KO` (Coca-Cola)
  - `HK.HSI` (Hang Seng Index)
  - `US.SPX` (S&P 500)
- **Data Frequency**: **10-minute OHLCV** (not 1-minute)
- **Time Range**: ~120 trading days (Feb 24 – Aug 18, 2026)
- **Total Rows**: 31,611 rows merged
- **Output**: `data/processed/unified_data.parquet` ✅

### 5. Code Modules (Phase 1 Complete)

| File | Status | Notes |
| :--- | :--- | :--- |
| `src/utils.py` | ✅ Done | RateLimiter, market hours (HK/US), calendar features, Telegram placeholder, constants |
| `src/data_loader.py` | ✅ Done | Loads Bloomberg CSVs, normalizes columns, freshness check (§9), writes Parquet |
| `src/trading_env.py` | ✅ Done | Gymnasium env, 5-part state space, Sharpe-drawdown reward |
| `src/inference.py` | ✅ Done | State persistence (§8), NewsPoller (5-min cap), Futu SIMULATE orders, `--dry-run` / `--once` |
| `src/__init__.py` | ✅ Done | Package marker |

### 6. Key Fixes Applied
- Bloomberg CSV filenames: `_1min` → `_10min` in `BLOOMBERG_FILES` and `FUTU_FILES`
- CSV header detection: added logic to handle Bloomberg exports **without header row**
- `.env` file created with API keys

### 7. Data Validation
```bash
python -m src.data_loader
```
✅ Output shows all 7 tickers loaded with ~4,000–5,000 rows each.
✅ `unified_data.parquet` written successfully.

---

## 📊 Current Project Structure

```
C:\Users\mongk\Desktop\airaire\
│
├── .env                          # ALPHAVANTAGE_API_KEY, FUTU_HOST, etc.
├── PLAN.md                       # Master plan
├── requirements.txt              # All dependencies
├── alpha-vantage-test.py         # API test script (keep)
├── paper-trade-test.py           # Futu test script (keep)
│
├── data/
│   ├── raw/
│   │   └── bloomberg/
│   │       ├── 0700_HK_10min.csv
│   │       ├── 3690_HK_10min.csv
│   │       ├── 3750_HK_10min.csv
│   │       ├── COST_US_10min.csv
│   │       ├── KO_US_10min.csv
│   │       ├── HSI_10min.csv
│   │       └── SPX_10min.csv
│   └── processed/
│       └── unified_data.parquet  # ← Generated successfully
│
├── src/
│   ├── __init__.py
│   ├── utils.py                  # ✅ Phase 1
│   ├── data_loader.py            # ✅ Phase 1
│   ├── trading_env.py            # ✅ Phase 1
│   ├── inference.py              # ✅ Phase 1
│   └── train.py                  # ⏳ TO BE IMPLEMENTED (Phase 3)
│
└── models/
    └── .gitkeep
```

---

## 🎯 What We Need Now — Phase 3: Training Script

### Task: Implement `src/train.py`

**Objective**: Create a 30-day rolling window training loop for the reinforcement learning agent.

### Functional Requirements

1. **Data Loading**
   - Use `src.data_loader.load_processed()` to load `unified_data.parquet`
   - Convert to wide format (datetime × ticker) for the Gym environment

2. **Environment**
   - Use `src.trading_env.TradingEnv`
   - Pass the loaded data and initial cash (`INITIAL_CASH = 1,000,000`)

3. **RL Algorithm**
   - Use `stable_baselines3.PPO` or `SAC`
   - Reason: Both are suitable for continuous action spaces; PPO is more stable, SAC is more sample-efficient. Choose PPO for MVP.

4. **Rolling Window Training**
   - **Window Size**: 30 calendar days (≈ 30 × 39 bars = ~1,170 data points per ticker)
   - **Step**: Slide by 1 day forward (sequential, NOT random sampling)
   - **For each window**:
     - Create a new environment instance with that window's data
     - Train the model for N epochs (default: 10)
     - Save checkpoint to `models/checkpoint_{window_end_date}.zip`
     - Record metrics: cumulative return, Sharpe ratio, max drawdown

5. **Command-line Arguments**
   - `--test`: Run only one window (quick validation), do not save model
   - `--epochs`: Number of training epochs per window (default: 10)
   - `--window-days`: Window size in days (default: 30)
   - `--output`: Directory for checkpoints (default: `models/`)

6. **Logging**
   - Print progress: window number, dates, metrics after each window
   - Save metrics to `models/training_log.csv`

### Expected Output

After running, the script should produce:
```
models/
├── checkpoint_2026-03-01.zip
├── checkpoint_2026-03-02.zip
├── ...
├── best_model.zip                # Best-performing checkpoint
└── training_log.csv              # Metrics per window
```

---

## 🔧 Important Context for Implementation

### Data Frequency Update
- We are now using **10-minute** bars (not 1-minute)
- `LOOKBACK_BARS = 30` in `trading_env.py` means **30 bars = ~5 hours of trading data**
- This is intentional — it represents a full trading session, which aligns with the strategy's focus on intraday trends

### CSV Format
- Bloomberg exports **no header row**
- Format: `datetime, open, high, low, close, volume` (volume may be `NaN`)
- `data_loader.py` already handles this correctly

### Alpha Vantage Integration
- API key is in `.env`
- `NewsPoller` in `inference.py` will automatically use the real API
- For training, news scores are currently placeholders (0.0) — this is fine for Phase 3

### Volume Column
- Bloomberg exports may not include `volume` column (values are `NaN`)
- This does NOT block training — the RL agent can learn from price action alone
- Future enhancement: backfill volume via Futu API if needed

---

## 🧪 Testing Plan After Implementation

1. **Dry run**:
   ```bash
   python -m src.train --test
   ```
   Expected: one window trains quickly (~2-5 minutes), no model saved.

2. **Full training** (optional, may take time):
   ```bash
   python -m src.train --epochs 10
   ```
   Expected: produces checkpoints and log file.

3. **Integration with inference**:
   After training, copy the best model to `models/best_model.zip`, then:
   ```bash
   python -m src.inference --dry-run --once
   ```
   Should load the model and generate actions.

---

## 📝 Notes for Cursor

- This document is your sync point. Please read it carefully before generating any code.
- If you need any clarification on the existing code (`utils.py`, `data_loader.py`, `trading_env.py`, `inference.py`), ask before writing `train.py`.
- The folder structure and constants (e.g., `CORE_TICKERS`, `INITIAL_CASH`, `DATA_PROCESSED`) are defined in `src/utils.py` — import them rather than redefining.
- The reward function in `trading_env.py` already implements Sharpe − λ × drawdown. Use it directly.

---

## 🚀 Ready to Start Phase 3

All infrastructure is complete. We are now ready for you to write `src/train.py`.

**Please confirm that you have read this sync document before proceeding.**
```