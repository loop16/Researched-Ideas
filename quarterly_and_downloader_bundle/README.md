# Quarterly Range Breakout Strategy Backtests

This folder contains backtest implementations for quarterly range breakout strategies, both in direct futures/equity trading and options trading formats.

## Overview

The quarterly range breakout strategy is a systematic approach that:
1. Defines a quarterly range using the first Friday of each quarter and the previous trading day
2. Waits for the first breakout from this range
3. Enters positions in the direction of the breakout
4. Holds until quarter expiration or risk management exit

## Strategy Components

### Quarterly Range Definition

Each quarter's range is determined by:
- **First Friday of Quarter**: The first Friday occurring in each calendar quarter (Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec)
- **Previous Trading Day**: The last trading day before the first Friday
- **Range High**: `max(FirstFriday.High, PreviousDay.High)`
- **Range Low**: `min(FirstFriday.Low, PreviousDay.Low)`
- **Range Mid**: `(RangeHigh + RangeLow) / 2`

### Breakout Detection

A breakout is triggered when:
- **Long Breakout**: First close > Range High
- **Short Breakout**: First close < Range Low

The strategy enters a position immediately when either breakout occurs.

## Strategy Variants

### 1. Quarterly Range Backtest (Direct Trading)

**File**: `Quarterly_Range_Backtest.py`

This backtest implements the strategy using direct futures/equity positions.

#### Strategy Rules

1. **Entry**: Enter long when close > range_high, or short when close < range_low (first breakout wins)
2. **Exit Rules**:
   - Hold until quarter expiration (last trading day of quarter) - primary exit
   - Risk management exit: If opposite side breaks after entry:
     - Long position exits if close < range_low
     - Short position exits if close > range_high
3. **Flipping Strategy**: When stopped out by opposite breakout, flip to that direction (if allowed)
4. **Market Structure Filter**: Short trades are disabled if prior quarter's range is below current range (uptrending market)

#### Features

- **Overall Strategy**: Includes both long and short trades with flipping
- **Long-Only Variant**: Only takes long trades for comparison
- **Equity Curves**: Generates compounded equity curves for both variants
- **Drawdown Analysis**: Tracks maximum drawdown over time
- **Performance Metrics**:
  - Total compounded return
  - Annualized return and volatility
  - Sharpe ratio (using 3% risk-free rate)
  - Win rate, average/median returns
  - Maximum drawdown

#### Outputs

- `quarterly_expiration_strategy_performance.png`: Equity curves and drawdown charts
- `quarterly_expiration_strategy_results.csv`: All trades with entry/exit details
- `quarterly_expiration_strategy_long_only_results.csv`: Long-only trades
- `quarterly_expiration_strategy_metrics.csv`: Performance summary metrics

#### Default Parameters

- **Start Year**: 1970
- **End Year**: 2024
- **Data File**: `SPX_1D.csv` (daily SPX data)

#### Usage

```python
from Quarterly_Range_Backtest import QuarterlyRangeBacktest

# Initialize backtest
backtest = QuarterlyRangeBacktest(
    data_file='SPX_Quarterly_Analysis_Results.csv',
    daily_file='SPX_1D.csv',
    start_year=1970,
    end_year=2024
)

# Load data and run backtest
backtest.load_data()
results = backtest.run_backtest()

# Print performance summary
backtest.print_performance_summary()

# Generate plots
backtest.plot_performance()

# Save results
backtest.save_results()
```

### 2. Options Quarterly Backtest

**File**: `Options_Quarterly_Backtest.py`

This backtest implements the same quarterly range breakout logic but using options instead of direct positions.

#### Strategy Rules

1. **Entry**: Same breakout detection (close > range_high for calls, close < range_low for puts)
2. **Options Sizing**: Fixed $500 risk per trade
3. **Moneyness Levels**: Tests three strategies simultaneously:
   - **OTM (Out-of-the-Money)**: 2.5% away from spot
   - **ATM (At-the-Money)**: At spot price
   - **ITM (In-the-Money)**: 2.5% into the money
4. **Strike Selection**: Rounded to nearest $5 increment
5. **Pricing Model**: Black-Scholes with:
   - **Implied Volatility**: VIX3M (3-month VIX)
   - **Risk-Free Rate**: Fed Funds rate
   - **Dividend Yield**: 1.4% (assumed)
6. **Exit Rules**: Same as direct trading (quarter expiration or opposite breakout)
7. **Market Structure Filter**: Same as direct trading (disables shorts in uptrends)

#### Features

- **Multiple Moneyness Testing**: Simultaneously tests OTM, ATM, and ITM strategies
- **Black-Scholes Pricing**: Uses VIX3M and Fed Funds rate for realistic option pricing
- **Cumulative PnL Tracking**: Tracks cumulative profit/loss by moneyness level
- **Individual Trade Analysis**: Tracks each option trade's entry/exit prices and PnL
- **Performance Comparison**: Compares performance across moneyness levels

#### Outputs

- `options_strategy_performance.png`: Cumulative PnL and individual trade scatter plots
- `options_strategy_results.csv`: All option trades with pricing details
- `options_strategy_metrics.csv`: Performance metrics by moneyness level

#### Default Parameters

- **Start Year**: 2008 (limited by VIX data availability)
- **End Year**: 2024
- **Risk Per Trade**: $500
- **Data Files**:
  - `SPX_Quarterly_Analysis_Results.csv`: Quarterly analysis results
  - `SPX_1D.csv`: Daily SPX data
  - `CBOE_VIX3M, 1D.csv`: 3-month VIX data
  - `FRED_FEDFUNDS, 1M.csv`: Fed Funds rate data

#### Usage

```python
from Options_Quarterly_Backtest import OptionsQuarterlyBacktest

# Initialize options backtest
backtest = OptionsQuarterlyBacktest(
    data_file='SPX_Quarterly_Analysis_Results.csv',
    daily_file='SPX_1D.csv',
    vix_file='CBOE_VIX3M, 1D.csv',
    fed_funds_file='FRED_FEDFUNDS, 1M.csv',
    start_year=2008,
    end_year=2024
)

# Load data and run backtest
backtest.load_data()
results = backtest.run_backtest()

# Print performance summary
backtest.print_performance_summary()

# Generate plots
backtest.plot_performance()

# Save results
backtest.save_results()
```

## Supporting Scripts

### SPX_Quarterly_Analysis.py

Analysis script that processes historical SPX data and:
- Identifies quarterly ranges
- Tracks breakout occurrences and outcomes
- Calculates breakout statistics and percentiles
- Generates QPP (Quarterly Percentile Profile) indicator levels

**Outputs**:
- `SPX_Quarterly_Analysis_Results.csv`: Detailed quarterly analysis
- `SPX_Quarterly_Normalized.csv`: Normalized quarterly candle data

### Multi_Asset_Quarterly_Analysis.py

Orchestrates quarterly analysis across multiple assets (SPX, NDX, CL, GC, BTC, etc.) and:
- Runs quarterly analysis for each asset
- Generates QPP indicator levels and percentiles
- Creates combined analysis files
- Generates JSON level files for visualization

**Outputs**:
- `{Asset}_Quarterly_Analysis_Results.csv`: Per-asset analysis
- `All_Assets_Basic_Quantiles.csv`: Combined quantiles
- `All_Assets_Detailed_Percentiles.csv`: Combined percentiles
- `{Asset}_Quarterly_Percentiles.csv`: QPP levels per asset
- JSON level files for visualization

### Data Downloaders

- `Downloader.py`: Generic data downloader
- `SP500_Downloader.py`: SPX-specific data downloader

## Data Requirements

### Direct Trading Backtest
- Daily OHLCV data for the asset (e.g., `SPX_1D.csv`)
- Quarterly analysis results (from `SPX_Quarterly_Analysis.py`)

### Options Backtest
- Daily OHLCV data for the asset
- Quarterly analysis results
- VIX3M historical data (`CBOE_VIX3M, 1D.csv`)
- Fed Funds rate data (`FRED_FEDFUNDS, 1M.csv`)

## Key Differences: Direct vs Options

| Aspect | Direct Trading | Options Trading |
|--------|---------------|-----------------|
| **Position Sizing** | Based on percentage returns | Fixed $500 risk per trade |
| **Leverage** | 1:1 (or configurable) | Inherent via options |
| **Exit Timing** | Same (quarter expiration or opposite breakout) | Same |
| **Performance Metric** | Percentage returns → equity curve | Dollar PnL → cumulative PnL |
| **Risk Management** | Stop on opposite breakout | Same stop logic |
| **Data Requirements** | Daily price data | Daily price + VIX + rates |
| **Backtest Period** | 1970-2024 | 2008-2024 (VIX data limit) |

## Performance Visualization

### Direct Trading Outputs
1. **Equity Curves**: Compounded equity over time for overall and long-only strategies
2. **Drawdown Charts**: Drawdown percentage over time
3. **Trade Logs**: Complete CSV of all trades with entry/exit details

### Options Trading Outputs
1. **Cumulative PnL Charts**: Cumulative dollar PnL by moneyness level (OTM, ATM, ITM)
2. **Individual Trade Scatter**: Trade-by-trade PnL visualization
3. **Trade Logs**: Complete CSV with option pricing details

## Notes

- The strategy uses **close-based breakouts** (not intraday breakouts)
- **Quarter expiration** is the last trading day of the calendar quarter
- Short trades are **automatically filtered** in uptrending markets (prior range below current range)
- Options backtest uses **Black-Scholes pricing** which may differ from market prices
- Both strategies assume **perfect execution** at close prices (no slippage in direct trading, perfect option pricing in options backtest)

## Future Enhancements

- Intraday breakout detection (not just close-based)
- Commission and slippage modeling
- Dynamic position sizing based on volatility
- Additional moneyness levels for options
- Real-time market pricing for options (not just Black-Scholes)
