# 🧠 Project Plan: AI Quant Agent (Papertrade) - "The Volatility Harvester"

> **Name**: AirAire
> **Version**: 1.0
> **Date**: 2026-08-17
> **IDE**: Cursor (Grok 4.6 / High Reasoning Mode)
> **Core Goal**: Build a self-improving AI trading agent that leverages **negative correlation** (HK Tech vs US Defensive) and **News Sentiment** to outperform a traditional rule-based algorithm in a Futu paper-trading environment.

---

## 1. 🎯 Project Vision & Core Philosophy

We are not building a simple "if-else" bot. We are building a **Hybrid AI Agent** that combines:

1.  **Reinforcement Learning (RL)**: The "Trader" (Decides *how much* to buy/sell).
2.  **Large Language Model (LLM) / NLP**: The "Analyst" (Reads news and quantifies *market mood*).
3.  **Feature Engineering**: The "Memory" (Remembers long-term trends, calendars, and cross-asset relationships).

**The Secret Sauce (Information Asymmetry)**:
> "The market forgets. Our AI remembers that the Dim Sum chef loses at horse racing on Wednesdays, making Thursdays terrible for HK stocks."
> *(Meaning: We inject **Calendar Effects** and **Long-term Moving Averages** into the model, not just the last 30 days of prices.)*

---

## 2. 🏗️ System Architecture (High Level)

| Layer | Technology Stack | Responsibility |
| :--- | :--- | :--- |
| **Data Gateway** | **Futu OpenD** + `futu-api` | Real-time & Historical OHLCV for 5 core stocks + 2 observers. |
| **Alternative Data** | **Alpha Vantage (Academic Full Tier)** | **Intraday** News Sentiment Scores. Pulled every 5-15 mins during trading hours. |
| **Feature Store** | **Pandas (Parquet/CSV)** | Local storage of enhanced data (Prices + Features). |
| **Training Ground** | **Google Colab (GPU)** | Runs daily RL training (FinRL + Stable-Baselines3). |
| **Brain (AI)** | **PPO / SAC** (FinRL) | The Policy Network that outputs trading actions. |
| **Execution Layer** | **Python Script (Local/Cloud)** | Loads the trained model and executes trades via Futu API. |
| **Frontend** | **Streamlit** | Local dashboard for monitoring P&L, Positions, and AI signals. |
| **Alert System** | **Telegram Bot** | Sends trade execution summaries. |

---

## 3. 📊 Asset Universe (The 5 Core + 2 Observers)

We trade only these 5, but we observe 2 indices for macro context.

| Pool | Ticker (Futu) | Bloomberg Ticker | Name | Role |
| :--- | :--- | :--- | :--- | :--- |
| **HK Tech** | `HK.00700` | `0700 HK Equity` | Tencent | Core Trade |
| **HK Tech** | `HK.03690` | `3690 HK Equity` | Meituan | Core Trade |
| **HK Tech** | `HK.03750` | `3750 HK Equity` | CATL (Ningde) | Core Trade |
| **US Defensive** | `US.COST` | `COST US Equity` | Costco | Core Trade |
| **US Defensive** | `US.KO` | `KO US Equity` | Coca-Cola | Core Trade |
| **Macro Observer** | `HK.HSI` | `HSI Index` | Hang Seng Index | Context Only |
| **Macro Observer** | `US.SPX` | `SPX Index` | S&P 500 | Context Only |

---

## 4. 🧠 The AI State Space (What the AI "Sees")

To be a "Forgetful Expert with Long-Term Memory", the Agent's `State` at time **t** contains:

### A. Short-Term Memory (30-Day Rolling Window)
- Past 30 days of **Minute-level (1-min)** OHLCV data for the 5 core stocks.
- *Purpose*: Captures immediate market micro-structure and recent volatility.

### B. Long-Term Memory (Injected via Features)
- **Moving Average Distance**: (Current Price - 200MA) / 200MA.
- **Historical Volatility Percentile**: Where does the current 30-day volatility rank in the last 2 years? (0 to 1).
- **90-Day Rolling Correlation**: e.g., Correlation(HK.00700, US.COST) over 90 days.
- *Purpose*: Tells the AI *where* we are in the macro cycle, even if it only "remembers" the last 30 days of raw prices.

### C. Calendar Effects (The "Thursday Dim Sum" Logic)
- `Day of Week` (0-6).
- `Month of Year` (1-12).
- `Days until major holidays` (Christmas, Chinese New Year, National Day).
- *Purpose*: Captures yearly/recurring anomalies.

### D. Intraday News Sentiment (LLM / NLP Layer)
- **Intraday Sentiment Score**: Average sentiment of the last 10 news headlines for each stock, updated every **5 minutes** during trading hours (via Alpha Vantage Full API).
- *Purpose*: Captures the "real-time vibe". If sentiment suddenly drops from 0.2 to -0.8 on Tencent *during* a trading session, the AI will instantly hedge or reverse position in the next inference cycle.

### E. Agent Status
- Current holdings of each of the 5 stocks (-1 to 1 ratio).
- Current cash balance.

---

## 5. 🎮 Action Space & Reward Function

### Actions (Continuous)
- For each of the 5 stocks: Output a continuous value in **[-1, 1]**.
- `1` = 100% Long (Buy).
- `-1` = 100% Short (Sell).
- `0` = Hold.
- *Constraint*: Leverage limited to 2x total portfolio value.

### Reward Function (The "Teacher")
We will **not** just use P&L. We use the **Sharpe Ratio** over the last N steps as the reward, combined with a penalty for large drawdowns.
- Reward = Sharpe_Ratio - λ * Max_Drawdown_Penalty
- *(Note to Cursor: Implement this in the `step()` function of the Gym environment.)*

---

## 6. 🗓️ Implementation Roadmap (MVP in 7-10 Days)

We are using **"Fast-Forward Backtesting"** to simulate 24/7 trading without needing a server on Day 1.

### Phase 0: Environment Setup (Day 1)
- [ ] **Cursor Setup**: Ensure Pro plan is active. Create project folder.
- [ ] **Python Env**: Create `venv` with `futu-api`, `stable-baselines3`, `finrl`, `pandas`, `streamlit`.
- [ ] **Futu OpenD**: Install locally. Ensure login works. Test `quote_ctx.get_market_snapshot()`.
- [ ] **Alpha Vantage**: 
    - [ ] **Crucial**: Email `support@alphavantage.co` with your **HKUST email** to request the Full Academic Tier.
    - [ ] Store the API Key in a `.env` file.

### Phase 1: Data Pipeline (Day 2) - *【HYBRID: Bloomberg + Futu】*

We use a **Hybrid Data Strategy** to avoid heavy API lifting:

#### Task A: Initial Bulk Download (Bloomberg Terminal - Manual, One-Time)
- [ ] **Action**: Go to the HKUST Library Bloomberg Terminal.
- [ ] **Export**: For each of the 7 tickers (5 core + 2 observers), export **2 years** of **1-Minute** OHLCV data.
    - *Bloomberg Ticker Format* (see Section 3):
        - `0700 HK Equity`, `3690 HK Equity`, `3750 HK Equity`
        - `COST US Equity`, `KO US Equity`
        - `HSI Index`, `SPX Index`
    - *Steps*: Type ticker -> hit `HP` (Historical Pricing) -> Set Date Range (2 years) -> Set Frequency (1 Minute) -> Hit `Export` -> Save as CSV.
- [ ] **Storage**: Place these CSV files into `data/raw/bloomberg/`.

#### Task B: Daily Incremental Update (Futu API - Automated)
- [ ] **Script 1: `data_fetcher.py`** (Incremental mode)
    - **Task**: **DO NOT** fetch 2 years of data. Only fetch the **most recent 30 days** of 1-Minute OHLCV data for the 5 core stocks + 2 observers via Futu API.
    - **Reason**: 30 days of 1-min data is ~11,700 rows per stock. Although it requires pagination (1000 rows per page), it's fast and well within rate limits.
    - **Task**: Save this daily-updated data to `data/raw/futu/latest/`.

#### Task C: Unified Data Loader (Merge Bloomberg + Futu)
- [ ] **Script 2: `data_loader.py`**
    - **Task**: When training is triggered, this script does the following:
        1. Load the **base 2-year data** from `data/raw/bloomberg/`.
        2. **Bloomberg Data Expiry Check**: After loading, if the last timestamp is older than **5 days**, log `'Bloomberg data stale. Please update manually.'` (see Section 9).
        3. Check `data/raw/futu/latest/` for newer timestamps.
        4. **Overlay/Append** the Futu data on top of the Bloomberg data (Bloomberg covers the past, Futu covers the very recent present/past 30 days).
        5. Output a single unified DataFrame for the last 30 days (or 2 years if needed) to `data/processed/unified_data.parquet`.

#### Task D: Feature Engineering
- [ ] **Script 3: `feature_engineering.py`**
    - **Task**: Load `unified_data.parquet` and calculate 200MA, 90-day Correlations, Calendar Variables, and Volatility Percentiles.
    - **Output**: `data/enhanced/enhanced_data.parquet`.

### Phase 2: Build the AI Environment (Day 3-4)
- [ ] **Script 4: `trading_env.py`**
    - **Task**: Custom Gym Environment based on FinRL.
    - **Task**: Implement `get_state()` to return the **State Space** defined in Section 4.
    - **Task**: Implement `step()` to execute actions and calculate the Sharpe-based reward.
    - **Task**: Implement **"Rolling Window"** logic. The environment will slide day by day over the 2-year data.

### Phase 3: Training & Validation (Day 5-6)
- [ ] **Notebook: `colab_training.ipynb`**
    - **Task**: Clone the project on Google Colab.
    - **Task**: Train a **Base Model** using the entire 2 years of data (Fast-Forward mode). (Expected time: ~15-30 mins).
    - **Task**: Save Base Model as `models/base_model.zip`.
    - **Task**: Simulate **"Daily Fine-tuning"**: Run the last 30 days of the dataset sequentially, fine-tuning the model each day and recording the "Trading Log".
    - **Task**: Validate performance against the "test set" (the final month of data).
    - **Criteria**: If Sharpe > 0.5 on test set, we proceed.

### Phase 4: Simulated Execution (Day 7+)
- [ ] **Script 5: `inference.py`** (Local PC)
    - **Task**: Load the latest `best_model.zip`.
    - **Task**: **Load `state.pkl` first** (holdings, cash, last action) so a restart never double-buys or loses P&L (see Section 8).
    - **Task**: Connect to Futu OpenD.
    - **Task**: Every **1 minute** (or on new data trigger):
        1. Fetch latest OHLCV bar.
        2. **Call Alpha Vantage API** (respecting 5-min minimum interval) to fetch the latest news sentiment for the 5 tickers.
        3. Construct the State vector (Price + Features + News).
        4. Call `model.predict()` to generate the action.
    - **Task**: Use `TrdEnv = SIMULATE` to place orders automatically.
    - **Task**: **After every trade**, persist holdings / cash / last action to `state.pkl`.

- [ ] **Frontend**: `dashboard.py` (Streamlit)
    - **Task**: Show real-time positions, equity curve, and current AI "Confidence" (Entropy).

---

## 7. 📁 Project Folder Structure

```text
/quant_agent/
│
├── README.md
├── PLAN.md                  # This file
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── bloomberg/          # Manual exports (2-year base)
│   │   │   ├── 0700_HK_1min.csv
│   │   │   ├── 3690_HK_1min.csv
│   │   │   ├── 3750_HK_1min.csv
│   │   │   ├── COST_US_1min.csv
│   │   │   ├── KO_US_1min.csv
│   │   │   ├── HSI_1min.csv
│   │   │   └── SPX_1min.csv
│   │   └── futu/               # Auto daily updates (Recent 30 days)
│   │       └── latest/
│   │           ├── 00700_HK_1min.csv
│   │           ├── 03690_HK_1min.csv
│   │           ├── 03750_HK_1min.csv
│   │           ├── COST_US_1min.csv
│   │           ├── KO_US_1min.csv
│   │           ├── HSI_1min.csv
│   │           └── SPX_1min.csv
│   ├── processed/
│   │   └── unified_data.parquet # Merged Bloomberg + Futu
│   └── enhanced/
│       └── enhanced_data.parquet # With features added
│
├── src/
│   ├── data_fetcher.py      # Incremental Futu fetcher (last 30 days)
│   ├── data_loader.py       # Merges Bloomberg + Futu
│   ├── feature_engineering.py
│   ├── trading_env.py       # The Gym Environment
│   ├── train.py             # Main training loop (works in Colab)
│   ├── inference.py         # Local 24/7 trader script
│   └── utils.py             # Helpers (logging, telegram alerts)
│
├── state.pkl                # Inference resume file (holdings, cash, last action)
│
├── models/
│   ├── base_model.zip       # Pre-trained on 2 years
│   └── best_model.zip       # Latest fine-tuned model
│
├── notebooks/
│   └── colab_training.ipynb # The Colab script
│
└── dashboard/
    └── streamlit_app.py
```

---

## 8. 💾 Data Persistence & Resumption (For Stability)

When writing `inference.py`, the bot must save its current state (holdings, cash, and last action) to a `state.pkl` file after every trade. If the script restarts, it must load this file **first** to avoid double-buying or losing track of P&L.

**Resume contract:**
1. On startup, load `state.pkl` *before* placing any Futu order.
2. Reconcile saved holdings against Futu `position_list_query` / `accinfo_query` and log any drift.
3. After every filled (or attempted) trade, overwrite `state.pkl` atomically.
4. On `Ctrl+C` / crash-path shutdown, persist the latest in-memory state.

---

## 9. ⏰ Bloomberg Data Expiry Check (For Robustness)

In `data_loader.py`, after loading the 2-year Bloomberg data, check if the last timestamp is older than **5 days**. If it is, raise a clear warning log saying `'Bloomberg data stale. Please update manually.'`

Do **not** abort the merge: Futu incremental files may still cover the recent window. The warning is the signal to re-export from the HKUST Bloomberg terminal.

