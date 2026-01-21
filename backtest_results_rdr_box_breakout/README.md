# RDR-Box Breakout Strategy (5R + 1-Tick Slippage)

## Strategy Rules

### 1. Range Definition (Day 1)
- **Time:** 09:30-15:55 NY
- Track the **High** and **Low** of all bars in this window
- This becomes the "Range" or "Box"

### 2. Breakout Detection (After 15:55)
- Monitor for the first bar where:
  - **LONG:** `bar.high > range_high` 
  - **SHORT:** `bar.low < range_low`
- This establishes the trade direction

### 3. Stop Calculation
- `box_height = range_high - range_low`
- **LONG Stop:** `range_low + (box_height × 0.75)`
- **SHORT Stop:** `range_high - (box_height × 0.75)`

### 4. Entry Calculation (5R)
- Track the **extreme** price after breakout:
  - LONG: `extreme = max(extreme, bar.high)`
  - SHORT: `extreme = min(extreme, bar.low)`
- Calculate 5:1 R:R entry:
  - `distance = extreme - stop` (LONG) or `stop - extreme` (SHORT)
  - `risk = distance / 6`
  - `entry = stop + risk` (LONG) or `stop - risk` (SHORT)
- **Add 1-tick slippage** to entry price

### 5. Entry Trigger
- **LONG:** When `bar.low <= entry`
- **SHORT:** When `bar.high >= entry`

### 6. Exit Rules
- **Target:** Original extreme price
- **Stop:** Calculated stop level
- **Time Exit:** 15:55 next trading day

---

## Results (2008-2025)

| Metric | Value |
|--------|-------|
| Total Trades | 13,631 |
| Total R | +4,445R |
| Win Rate | 26.8% |
| Sharpe Ratio | 3.69 |

### By Asset
| Asset | Trades | Total R | Win Rate |
|-------|--------|---------|----------|
| J1 | 3,497 | +1,252R | 28.9% |
| GC | 3,186 | +1,215R | 27.3% |
| CL | 2,621 | +831R | 25.3% |
| NQ | 2,044 | +721R | 25.2% |
| ES | 2,283 | +427R | 25.8% |

### Commission Analysis ($500 Max Risk)
| Scenario | Final Equity | Trades | Commissions |
|----------|-------------|--------|-------------|
| All Trades | $1,603,199 | 13,631 | $455,222 |
| 20% Filter | $1,539,953 | 13,176 | $400,279 |

---

## Files
- `backtest_rdr_box_breakout.py` - Main backtest script
- `equity_curves_by_asset.png` - R-curve by asset
- `equity_with_commissions.png` - Compounded equity w/ commissions
- `trades_rdr_box_breakout_5R.csv` - All trade data
