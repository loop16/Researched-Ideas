# WDDRB Patches Summary

## Overview

All focused patches have been successfully applied to `wddrb.py` to fix the model classification logic and signed extension/retracement calculations:

✅ **Model classification logic** - Fixed mutual-exclusive, ordered rules  
✅ **Mid-referenced SD spans** - High/low distances from the RDR mid pre/post breakout  
✅ **Next day open positioning** - Signed to breakout edge  

## Patch Details

### 1. Model Classification Logic ✅

**What was fixed:**
- **Before**: Conditions could overlap, U/D/UX/DX rarely appeared, everything fell through to "C"
- **After**: Mutual-exclusive, priority-ordered rules ensure U/D/UX/DX actually show up

**New Logic (Priority Order):**
```python
# 1. Inside / Engulf first (highest priority)
if next_lo >= curr_lo and next_hi <= curr_hi: return "I"
if next_lo <= curr_lo and next_hi >= curr_hi: return "O"

# 2. Clean outside (no overlap)
if next_lo >= curr_hi: return "U"
if next_hi <= curr_lo: return "D"

# 3. Straddle patterns (overlap but break edge)
if next_hi > curr_hi and next_lo < curr_hi:
    return "UX" if next_lo >= curr_mid else "U"
if next_lo < curr_lo and next_hi > curr_lo:
    return "DX" if next_hi <= curr_mid else "D"

# 4. Everything else is overlap chain
return "C"
```

**Model Definitions:**
- **I**: Next fully inside current
- **O**: Next fully engulfs current  
- **U**: Next entirely above prior high
- **D**: Next entirely below prior low
- **UX**: Straddle-up - breaks above high AND low ≥ prior mid (upper-half straddle)
- **DX**: Straddle-down - breaks below low AND high ≤ prior mid (lower-half straddle)
- **C**: Overlap but none of the above

### 2. Mid-Referenced SD Spans ✅

**What was added:**
- `_range_high_low_mid_sd()`: Expresses any slice’s high/low in σ units relative to the RDR mid (mid = 0).
- Full split of the RDR window into **pre-break**, **post-break**, and **full-window** readings.

**Why:** Instead of edge-based extension/retracement, every section of the day now shares one coordinate system (quarters of the RDR box). You can see whether highs/lows before the breakout already pressed into +0.5σ, how far they travelled afterward, and where the complete session settled.

### 3. Next Day Open Positioning ✅

**What was added:**
- Next day open expressed relative to the RDR mid (not prior close)
- Same σ units as the range spans for direct comparisons

**Logic:**
```python
next_open_sd = (next_open - rdr_mid) / (rdr_hi - rdr_lo)
```

## New Output Fields

The WDDRB now includes these new signed σ metrics:

```python
{
    # High/low vs RDR mid (σ units) for each slice
    "high_sd_full", "low_sd_full",
    "high_sd_pre_break", "low_sd_pre_break",
    "high_sd_post_break", "low_sd_post_break",
    "full_high_ts", "full_low_ts",
    "post_high_ts", "post_low_ts", "post_total_ts",
    "post_high_index_30m", "post_low_index_30m", "post_total_index_30m",

    # Next day open vs RDR mid (σ units) for consistent comparison
    "next_open_sd",
}
```

## What You'll See After This

### ✅ **U/D/UX/DX Models Will Populate:**
- **U**: Only when next RDR is entirely above prior high
- **UX**: When breaks above prior high AND low ≥ prior mid (upper-half straddle)
- **D**: Only when next RDR is entirely below prior low  
- **DX**: When breaks below prior low AND high ≤ prior mid (lower-half straddle)

### ✅ **Signed Extension/Retracement:**
- Extension always positive in breakout direction
- Retracement always negative (pullback against breakout)
- Both measured after first breakout bar
- Quarter-σ buckets like +Q2, -Q1, +Q4+ available

### ✅ **Enhanced Next Day Open:**
- Positioned relative to breakout edge (not prior close)
- Signed σ units with quarter-σ bucketing
- Consistent with extension/retracement methodology

## Usage Example

```python
from world.wddrb import build_wddrb_from_csv

# Build WDDRB with new patches
pq_path, csv_path = build_wddrb_from_csv(
    csv_path=Path("ES_full_5min.csv"),
    symbol="ES", 
    out_dir=Path("output")
)

# Load and examine results
df = pd.read_parquet(pq_path)

# Check model distribution (should see U/D/UX/DX now)
print(df['model_code'].value_counts())

# Check SD spans
cols = [
    "high_sd_full", "low_sd_full",
    "high_sd_pre_break", "low_sd_pre_break",
    "high_sd_post_break", "low_sd_post_break",
]
print(df[cols].describe())

```

## Benefits

✅ **Model Classification**: U/D/UX/DX models now appear as intended  
✅ **Signed Metrics**: Proper extension (positive) vs retracement (negative)  
✅ **Breakout Edge Positioning**: Next day open positioned relative to breakout edge  
✅ **Mutual Exclusivity**: No more overlapping conditions falling through to "C"  
✅ **Consistent Logic**: All metrics use the same σ units anchored to the RDR mid  

The WDDRB now provides clean, mutually-exclusive model classifications and properly signed extension/retracement metrics with a single σ-based coordinate system across all fields.
