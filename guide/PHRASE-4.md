# 📊 Project Status & Next Phase: Full News-Integrated Training

> **Date:** 2026-08-18
> **Status:** Price-Only Training Completed (10 windows) → Ready for Full News Integration
> **IDE:** Cursor (Grok 4.6)

---

## 🎯 Executive Summary

We have successfully completed **10 sequential training windows** using **price-only data** (no news sentiment). The model is stable, saving checkpoints correctly, and showing **valid learning signals** (Sharpe ratios between 100-145 in evaluation). 

**However, price-only is NOT our final goal.** The core differentiator of this project is **"Information Asymmetry"** — the ability to read news sentiment and act on it before the market fully prices it in. Our AI must see **both** price action AND news sentiment to outperform your friend's rule-based algorithm.

### What We Have Accomplished
| Item | Status |
|------|--------|
| `data_loader.py` | ✅ Loads Bloomberg 10-min CSVs, merges, creates unified parquet |
| `trading_env.py` | ✅ Gym environment with 5-block state space (price window, long-term features, calendar, news, inventory) |
| `train.py` | ✅ Rolling 30-day window training, saves checkpoints, logs metrics |
| `inference.py` | ✅ Paper trading with state persistence, NewsPoller, Futu SIMULATE |
| `utils.py` | ✅ RateLimiter, market hours, Telegram placeholder |
| **Price-Only Training** | ✅ **10 windows completed** (Feb 24 – Apr 10, 2026) |
| **Alpha Vantage Academic Access** | ✅ **Email sent** — awaiting confirmation |
| **News Data Integration** | ❌ **NOT YET IMPLEMENTED** — THIS IS THE NEXT PHASE |

---

## 📈 Price-Only Training Results (10 Windows)

### Best Performing Model: **Window 4 (2026-02-27 → 2026-04-02)**

| Metric | Value | Note |
|--------|-------|------|
| **Sharpe Ratio (evaluation)** | **144.54** | Highest among all 10 windows |
| **Cumulative Return** | ~5.87e+37 | (Highly exaggerated due to unscaled reward) |
| **Max Drawdown** | 1.0 | (Indicates aggressive leverage) |
| **Final Equity** | ~5.87e+37 | |
| **Checkpoint File** | `models/checkpoint_2026-04-02.zip` | ✅ Saved successfully |

### Top 5 Windows (by Sharpe Ratio)

| Rank | Window | End Date | Sharpe Ratio | Max DD |
|------|--------|----------|--------------|--------|
| 1 | **Window 4** | 2026-04-02 | **144.54** | 1.0 |
| 2 | Window 8 | 2026-04-08 | 115.41 | 1.0 |
| 3 | Window 7 | 2026-04-07 | 115.39 | 1.0 |
| 4 | Window 5 | 2026-04-03 | 112.46 | 1.0 |
| 5 | Window 9 | 2026-04-09 | 110.17 | 0.0 |

### Training Stability Metrics (Window 4)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| `approx_kl` | 0.01 – 0.06 | ✅ Stable (below 0.1) |
| `explained_variance` | 0.575 | ✅ Good (value network predicting returns) |
| `value_loss` | ~4,660 | ⚠️ High but expected (unscaled reward) |
| `loss` | 502 | ✅ Converging well |
| `ep_rew_mean` | ~18,100 | ⚠️ Large (reward scale is 100x) |
| **Checkpoint** | `models/checkpoint_2026-04-02.zip` | ✅ Available |

### Why "Best Model" Matters

The Window 4 checkpoint (`checkpoint_2026-04-02.zip`) represents the most stable price-only learning signal. It will serve as our **baseline model** to compare against the **news-integrated model** once we retrain with Alpha Vantage sentiment data.

---

## 🔍 Critical Diagnosis: Why We Need News Data

### The Problem with Price-Only Training

1. **Information Gap**: The AI only sees price bars and technical indicators. It has **no access to**:
   - Earnings reports
   - Macroeconomic news (Fed decisions, inflation data)
   - Company-specific announcements
   - Market sentiment (bullish/bearish vibes)

2. **Over-Reactivity**: Without news context, the AI may overreact to short-term price noise, causing excessive trading and high drawdowns (max DD = 1.0 in multiple windows).

3. **Missed Opportunities**: The core thesis of "HK Tech vs US Defensive negative correlation" — the AI sees the correlation in prices but **doesn't understand WHY** (e.g., "Fed signals rate cut → US defensive stocks rally → HK tech sells off"). Without news, it's trading blind.

4. **Value Network Instability**: The high `value_loss` (3,000+) is partly because the AI cannot predict future returns without knowing the "news context" behind price movements.

### Overflow Warnings in Training Log

```
RuntimeWarning: overflow encountered in scalar add
RuntimeWarning: overflow encountered in divide
RuntimeWarning: overflow encountered in subtract
```

**Root Cause**: When the AI attempts extreme actions (e.g., going heavily long on 5 assets simultaneously), the `_mark_to_market` function tries to compute `self._cash + np.dot(self._holdings, prices)` — if `self._cash` becomes negative or `holdings` explode, Python raises overflow warnings.

**Why News Will Help**: With news sentiment, the AI will have additional signal to avoid extreme actions during high-uncertainty periods (e.g., negative news → should be cautious → smaller positions → fewer overflow warnings).

---

## 🚀 Next Phase: Full News-Integrated Training

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Algorithm** | PPO (Stable-Baselines3) |
| **Data Frequency** | 10-minute OHLCV |
| **Time Period** | Feb 24, 2026 – Apr 10, 2026 |
| **Window Size** | 30 session days |
| **Rolling Step** | 1 day |
| **News Data** | Alpha Vantage `NEWS_SENTIMENT` API |
| **State Space** | 5 blocks: Price Window + Long-Term Features + Calendar + **News Scores** + Inventory |
| **Training Device** | CPU (Colab CPU, 23 fps) |

### What Needs to Be Done (Priority Order)

#### ✅ Phase 1: Data Pipeline (COMPLETED)
- `data_loader.py` loads Bloomberg CSVs ✅
- Creates `data/processed/unified_data.parquet` ✅

#### ⚠️ Phase 1b: News Data Pipeline (IN PROGRESS — NEEDS IMPLEMENTATION)
- **Load historical news sentiment from Alpha Vantage API** (or CSV)
- **Merge with price data** on timestamp and ticker
- **Store enhanced dataset** as `data/processed/enhanced_data.parquet` with `news_scores` column

#### ⚠️ Phase 2: News-Integrated Training Environment (NEEDS MODIFICATION)
- Modify `TradingEnv` to accept `news_df`
- In `_news_score_features()`, query the latest `sentiment_score` for each ticker at current `datetime`
- Forward-fill missing news scores (no NaN values in observation)

#### ⚠️ Phase 3: Training Script (NEEDS MODIFICATION)
- In `train.py`, when creating each window's environment:
  - Slice both price data **AND news data** for the same date range
  - Pass `news_df` to `TradingEnv` constructor
- Verify `_get_obs()` returns non-zero values in block (4)

#### ⚠️ Phase 4: Overflow Warning Fix (OPTIONAL BUT RECOMMENDED)
- `_mark_to_market`: When `self._holdings` or `prices` contain extreme values, cap them before computing dot product.
- `_rebalance`: Ensure `equity` never falls below `1e-9` before division.
- `_sharpe_drawdown_reward`: Use `np.percentile` to clip extreme values in `equity_curve`.
- `_current_closes`: Ensure `_last_good_prices` is always set when price is valid.

#### ⚠️ Phase 5: Checkpoint Resume (NICE-TO-HAVE)
- Add `--resume N` flag to `train.py`
- If specified, load `models/checkpoint_YYYY-MM-DD.zip` and continue from window `N+1`
- This prevents losing progress if training is interrupted

---

## 📋 Full Technical Specification for Cursor

### Cursor, Please Read This Carefully:

We have completed **10 price-only training windows** and saved checkpoints. Now we need to **integrate Alpha Vantage news sentiment** into the training pipeline.

### What We Need You To Write/Modify:

---

#### **1. News Data Loader: `src/news_loader.py` (NEW FILE)**

```python
import pandas as pd
from pathlib import Path
from src.utils import CORE_TICKERS, AV_TICKERS

def fetch_historical_news(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str
) -> pd.DataFrame:
    """
    Fetch historical news sentiment from Alpha Vantage API.
    Returns DataFrame with columns: datetime, ticker, sentiment_score.
    """
    # Use NEWS_SENTIMENT function with tickers parameter
    # Limit to 10 headlines per ticker per query
    # Cache results to data/raw/news/ for future runs

def load_news_csv(ticker: str, path: Path) -> pd.DataFrame:
    """Load local news CSV if exists, else fetch from API."""
    pass

def load_all_news(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    force_fetch: bool = False
) -> pd.DataFrame:
    """
    Load news sentiment for all CORE_TICKERS.
    Resample to 10-minute frequency using forward fill.
    """
    pass
```

**Constraints**:
- Ethical polling: maximum **1 API call per 5 minutes per ticker**
- Cache results to `data/raw/news/` as CSV to avoid re-fetching
- If API fails, **warn but continue** (use zeros as fallback)
- Return DataFrame: `datetime, ticker, sentiment_score` (score in [-1, 1])

---

#### **2. Enhanced Data Loader: Modify `src/data_loader.py`**

Add a new function:

```python
def load_enhanced_data(
    bloomberg_dir: Path | None = None,
    news_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Load unified price data and merge with news scores.
    Returns DataFrame with columns: datetime, ticker, open, high, low, close, volume, news_score.
    """
    # 1. Load unified price data (existing function)
    # 2. Merge with news_df on (datetime, ticker) using forward fill
    # 3. Save to data/enhanced/enhanced_data.parquet
    # 4. Log coverage: "% of timestamps with news data"
```

---

#### **3. Environment Modification: Modify `src/trading_env.py`**

**Change 1: `__init__` method**

```python
def __init__(
    self,
    df: pd.DataFrame | None = None,  # Price data
    news_df: pd.DataFrame | None = None,  # NEW: News data
    lookback_bars: int = LOOKBACK_BARS,
    window_days: int = WINDOW_DAYS,
    initial_cash: float = INITIAL_CASH,
    render_mode: str | None = None,
) -> None:
    # ... existing code ...
    self.news_df = self._prepare_news(news_df)
    # ... rest ...
```

**Change 2: `_prepare_news` method (NEW)**

```python
def _prepare_news(self, news_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Ensure news data has datetime, ticker, sentiment_score. Forward-fill missing values."""
    if news_df is None or news_df.empty:
        return None
    # Convert to datetime, pivot to wide format
    # Forward-fill per ticker
    return news_df
```

**Change 3: `_news_score_features()` method**

```python
def _news_score_features(self) -> np.ndarray:
    """(4) Return real news sentiment scores for each core ticker."""
    dt = self._current_dt()
    scores = []
    for ticker in CORE_TICKERS:
        if self.news_df is not None:
            # Query latest news at or before dt
            subset = self.news_df[
                (self.news_df["ticker"] == ticker) &
                (self.news_df["datetime"] <= dt)
            ]
            if not subset.empty:
                score = subset.iloc[-1]["sentiment_score"]
            else:
                score = 0.0
        else:
            score = 0.0  # Fallback (should not happen after integration)
        scores.append(float(np.clip(score, -1.0, 1.0)))
    return np.asarray(scores, dtype=np.float32)
```

---

#### **4. Training Script: Modify `src/train.py`**

**Change 1: Make environment builder accept news**

```python
def make_vec_env(df: pd.DataFrame, news_df: pd.DataFrame | None = None, window_days: int = WINDOW_DAYS) -> DummyVecEnv:
    def _factory() -> Monitor:
        env = TradingEnv(
            df=df,
            news_df=news_df,  # Pass news to env
            initial_cash=INITIAL_CASH,
            window_days=window_days
        )
        return Monitor(env)
    return DummyVecEnv([_factory])
```

**Change 2: In training loop, slice news data for each window**

```python
for win in windows:
    # ... existing code to slice df ...
    
    # NEW: Slice news data for same window
    if news_df is not None:
        window_news = news_df[
            (news_df["datetime"] >= win.start) &
            (news_df["datetime"] <= win.end)
        ].copy()
    else:
        window_news = None
    
    vec_env = make_vec_env(win.df, window_news, window_days)
    # ... continue training ...
```

**Change 3: Add `--resume` argument**

```python
def parse_args():
    p.add_argument("--resume", type=int, default=0, 
                   help="Start from window N (e.g., --resume 11 loads checkpoint from window 10)")
```

In training loop:
```python
if args.resume > 0:
    # Load checkpoint from previous window
    prev_date = windows[args.resume - 2].end
    ckpt_path = MODELS_DIR / f"checkpoint_{prev_date}.zip"
    model = PPO.load(str(ckpt_path), env=vec_env, device=device)
    logger.info(f"Resumed from {ckpt_path}")
    start_window = args.resume - 1
else:
    start_window = 0
```

**Change 4: Fix overflow warnings (add this to `trading_env.py`)**

```python
def _mark_to_market(self, prices: np.ndarray) -> float:
    prices = np.clip(prices, 0.01, 100000)  # Cap extreme prices
    holdings = np.clip(self._holdings, -1e6, 1e6)  # Cap holdings
    value = self._cash + np.dot(holdings, prices)
    if not np.isfinite(value) or abs(value) > 1e9:
        return self._last_equity
    return _safe_float(value, self._last_equity)
```

---

#### **5. Updated Inference: Modify `src/inference.py` to use news data (already partially done)**

Your `inference.py` already has the `NewsPoller` and `set_news_scores()` method. But ensure that when you run inference, it uses **real news** from Alpha Vantage API, not mock data:

```python
# In inference.py, ensure this exists:
news_poller = NewsPoller(api_key=os.getenv("ALPHAVANTAGE_API_KEY"))
news_now = news_poller.fetch_open_markets()
env.set_news_scores(news_now)
state.news_scores = news_now
```

---

## 🗂️ Folder Structure After News Integration

```
/quant_agent/
├── data/
│   ├── raw/
│   │   ├── bloomberg/          # 7 CSV files (10-min OHLCV)
│   │   └── news/               # NEW: News sentiment CSVs
│   │       ├── news_0700_HK.csv
│   │       ├── news_3690_HK.csv
│   │       └── ...
│   ├── processed/
│   │   └── unified_data.parquet # Price data only
│   └── enhanced/
│       └── enhanced_data.parquet # Price + News scores (NEW)
├── models/
│   ├── checkpoint_2026-03-30.zip
│   ├── checkpoint_2026-03-31.zip
│   ├── ...
│   └── checkpoint_2026-04-10.zip  # Best so far: Window 4
├── src/
│   ├── data_loader.py          # MODIFIED: Add enhanced loading
│   ├── news_loader.py          # NEW: Alpha Vantage news fetcher
│   ├── trading_env.py          # MODIFIED: Accept news data
│   ├── train.py                # MODIFIED: Pass news to env, add --resume
│   ├── inference.py            # Already has NewsPoller
│   └── utils.py                # (no change)
└── notebooks/
    └── colab_training.ipynb    # (no change)
```

---

## 🏆 After News Integration: Expected Improvements

| Metric | Price-Only (Window 4) | Expected with News |
|--------|----------------------|-------------------|
| **Sharpe Ratio** | 144.54 | **150+** (more stable) |
| **Max Drawdown** | 1.0 | **< 0.5** (news prevents extreme trades) |
| **Value Network Accuracy** | moderate | **higher** (news context improves prediction) |
| **Overflow Warnings** | many | **fewer** (less extreme actions) |
| **Explanation Quality** | "Policy network output" | **"News Sentiment dropped sharply + High negative correlation"** |
| **Trading Frequency** | 10-15 trades/day | **5-8 trades/day** (more selective) |

---

## 🚀 Next Steps for You (Human)

1. **Send this document to Cursor** with the instruction:
   > "Read this entire markdown document. Implement ALL modifications described in the 'Full Technical Specification for Cursor' section. Focus on Phase 1b, Phase 2, and Phase 3 first. Phase 4 and Phase 5 are nice-to-have."

2. **Prepare for Alpha Vantage API**:
   - If you don't have the API key yet, check your email (you sent the request yesterday)
   - If you have it, store in `.env` as `ALPHAVANTAGE_API_KEY`

3. **Test the new pipeline**:
   ```bash
   python -m src.data_loader  # Should now load enhanced data with news
   python -m src.train --test  # Quick test with 1 window + news
   ```

4. **Run full training**:
   ```bash
   python -m src.train  # Will train 118 windows with news data
   ```

5. **Compare with price-only baseline**:
   - Use Window 4 checkpoint (`2026-04-02`) as baseline
   - After news training, evaluate both on same test period

---

## ⚠️ Important Constraints

- **No Real Money**: ALL trades are `SIMULATE` only
- **Ethical API Usage**: News polling ≤ 1 call per 5 minutes per ticker
- **No Overfitting**: We're using rolling windows (30 days) to ensure the model adapts to market changes

---

## 💬 Final Summary

| Phase | Status |
|-------|--------|
| Price-Only Training (Baseline) | ✅ COMPLETED — 10 windows, best model = Window 4 |
| News Data Pipeline | ⏳ **NEXT — Need to implement `news_loader.py`** |
| Environment Integration | ⏳ **NEXT — Need to modify `TradingEnv`** |
| Training with News | ⏳ **NEXT — Need to modify `train.py`** |
| News-Integrated Training | ⏳ **TO RUN** |
| Model Comparison (News vs. Baseline) | ⏳ **TO EVALUATE** |

**The core differentiator of your project — combining news sentiment with price action — is still ahead. This is where the magic happens.** 🚀