# Two Bullet Backtest - 2.5 SD Target Strategy

## Overview
This backtest implements a "two bullet" breakout trading strategy across multiple futures assets (ES, NQ, GC, CL, J1) with commission-aware position sizing and equity curve analysis.

## Strategy Parameters

### Entry Clusters
- **Cluster 1**: Entry at +0.2 SD, Stop at -0.2 SD
- **Cluster 2**: Entry at -0.4 SD, Stop at -0.8 SD

### Exit Rules
- **Target**: 2.5 standard deviations from Inner Defining Range (IDR)
- **Stop Loss**: Cluster-specific stop levels
- **Session Close**: 15:55 NY time

### Defining Range (DR)
- **Time Window**: 09:30 - 10:25 NY time
- **DR_HIGH**: Highest high price in the range
- **DR_LOW**: Lowest low price in the range
- **IDR_HIGH**: Maximum of (highest open, highest close) in the range
- **IDR_LOW**: Minimum of (lowest open, lowest close) in the range

### Breakout Detection
- First candle close outside the Defining Range after 10:30 NY time
- Direction: Long if close > DR_HIGH, Short if close < DR_LOW

## Position Sizing & Risk Management

### Risk Parameters
- **Risk per Trade**: 0.2% of equity
- **Max Dollar Risk**: $5,000 per trade (cap)
- **Commission**: $3 per contract per side ($6 round trip)
- **Commission Filter**: Trades with commission > 15% of risk are excluded

### Position Sizing Calculation
1. Calculate dollar risk per contract based on stop size
2. Determine target dollar risk: `risk_per_trade * equity` (capped at $5,000)
3. Calculate contracts: `target_dollar_risk / dollar_risk_per_contract`
4. Update equity after each trade: `equity += (R-multiple * dollar_risk) - commission`

## Assets Traded
- **ES** (E-mini S&P 500): Tick size 0.25, Tick value $12.50
- **NQ** (E-mini Nasdaq): Tick size 0.25, Tick value $5.00
- **GC** (Gold): Tick size 0.1, Tick value $10.00
- **CL** (Crude Oil): Tick size 0.01, Tick value $10.00
- **J1** (Japanese Yen): Tick size 0.0000005, Tick value $6.25

## Backtest Period
- **Start Date**: January 1, 2019
- **End Date**: End of available data
- **Data Frequency**: 5-minute bars (NY time)

## Output Files

### Plots
1. **equity_curves_by_asset.png**: Cumulative R-multiple equity curves by asset
2. **equity_with_commissions.png**: Commission-aware equity curves (log scale)
   - Blue line: All trades
   - Green dashed line: Filtered trades (commission ≤ 15% of risk)
   - Bottom panel: Commission costs over time

### Data Files
1. **trades_two_bullet_2.5_sd.csv**: Complete trade log with all trade details

### Results
1. **backtest_results.txt**: Complete backtest output including:
   - Trade statistics
   - Exit reason breakdown
   - Commission analysis
   - Sharpe ratio comparisons

## Sharpe Ratio Analysis

The backtest calculates Sharpe ratios for three scenarios:

1. **No Commissions**: Based on R-multiples only
2. **With Commissions**: All trades including commission costs
3. **Filtered**: Only trades where commission ≤ 15% of risk

## Key Features

- **Commission-Aware**: Realistic position sizing with commission costs
- **Equity Compounding**: Equity grows/compounds based on trade results
- **Commission Filtering**: Automatically skips trades with excessive commission ratios
- **Multi-Asset**: Simultaneous trading across 5 futures markets
- **Logarithmic Equity Curves**: Better visualization of equity growth

## Running the Backtest

```bash
python backtest_simple_two_bullet.py
```

All outputs will be saved to the `backtest_results_2.5sd` directory.

## Notes

- Equity only compounds from trades that pass the commission filter
- When a trade is skipped (commission > 15% of risk), equity remains unchanged
- Position sizing is calculated based on current equity, with a maximum cap of $5,000 risk per trade
- All timestamps are in America/New_York timezone

