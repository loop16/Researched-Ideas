# Breakout Strategy Optimization & Analysis

This directory contains the code and results for the **Breakout Strategy Optimization** (ES, CL, GC).

## 🚀 Strategy Logic

The core strategy is a **Trend Following Breakout** system:
1.  **Entry**: 
    - **Time-based**: Enter at fixed times (e.g., 03:00 ET).
    - **Breakout**: Enter when price breaks Asian/London session ranges.
2.  **Risk Management**: **0.8% Stop Loss**. (Tight stop is the key driver of profit).
3.  **Exit**: Time-based exit at **15:55 ET** (End of session).
4.  **Direction**: Determined by an **XGBoost Classifier** (trained on rolling 500-day window).

## 📁 Files

- `run_strategy_optimization.py`: The main script.
  - Loads data.
  - Generates signals.
  - Runs **Walk-Forward Validation**.
  - Outputs P&L stats and plots.
- `run_strategy_eda.py`: Exploratory Data Analysis.
  - Analyzes why the strategy works (HMM Regimes, Volatility Clustering).
- `*.png`: Equity curves and validation plots.

## 📊 Key Results

| Strategy | Config | P&L (Walk-Forward) | Edge vs Random |
|----------|--------|-------------------|----------------|
| **CL (Oil)** | Asia Breakout -> 15:55 | **+1019.6%** | High |
| **ES (S&P)** | 03:00 -> 15:55 | **+307.6%** | Medium |
| **GC (Gold)** | Asia Breakout -> 15:55 | **+172.3%** | Low |

**Key Finding**: The **0.8% Stop Loss** is responsible for ~90% of the profitability. The model adds a small edge, but the asymmetric risk profile (cut losers, hold winners) is the primary engine.

## 🛠 Usage

Run the optimization script:
```bash
python run_strategy_optimization.py
```

Run EDA:
```bash
python run_strategy_eda.py
```
