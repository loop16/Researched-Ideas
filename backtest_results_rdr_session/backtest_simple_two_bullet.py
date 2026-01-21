# Simple Two Bullet Backtest - Multi Asset
# Two entry clusters with hardcoded parameters, exit at end of session OR 3.0 SD target (whichever comes first)
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

NY_TZ = "America/New_York"

# ----------------------------
# Hardcoded Cluster Parameters
# ----------------------------
# Cluster 1: +0.2 TO -0.2
CLUSTER_1_ENTRY = 0.2
CLUSTER_1_STOP = -0.2

# Cluster 2: -0.4 TO -0.8
CLUSTER_2_ENTRY = -0.4
CLUSTER_2_STOP = -0.8

# Target: 3.0 SD
TARGET_SD = 2.5

# ----------------------------
# Config & dataclasses
# ----------------------------
@dataclass
class Cluster:
    entry: float
    stop: float
    name: str = ""

@dataclass
class FillResult:
    filled: bool
    fill_time: Optional[pd.Timestamp] = None
    cluster_name: Optional[str] = None
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None

@dataclass
class TradeExit:
    exit_time: pd.Timestamp
    exit_reason: str  # "TARGET", "STOP", "SESSION_CLOSE"
    exit_price: float
    pnl: float
    r_multiple: Optional[float]

# ----------------------------
# Utility functions
# ----------------------------
def load_ohlcv(path: str) -> pd.DataFrame:
    """Load OHLCV data and ensure proper NY timezone handling"""
    print(f"Loading OHLCV data from: {path}")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(NY_TZ, nonexistent="shift_forward", ambiguous="NaT")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(NY_TZ)
    df = df.set_index("timestamp").sort_index()
    print(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")
    return df

# ----------------------------
# Core trading logic
# ----------------------------
def build_defining_range(day_df: pd.DataFrame, dr_start: str = "09:30", dr_end: str = "10:25") -> Dict[str, float]:
    """Build Defining Range - IDR uses max of open/close, min of open/close"""
    start_ts = pd.Timestamp(dr_start).time()
    end_ts = pd.Timestamp(dr_end).time()
    times = sorted({ts.time() for ts in day_df.index})
    dr_times = [t for t in times if (t >= start_ts and t <= end_ts)]
    
    dr_bars = []
    for t in dr_times:
        time_str = f"{t.strftime('%H:%M')}"
        time_bars = day_df.between_time(time_str, time_str)
        if not time_bars.empty:
            dr_bars.append(time_bars.iloc[0])
    
    if len(dr_bars) == 0:
        return {"DR_HIGH": np.nan, "DR_LOW": np.nan, "SR_HIGH_CLOSE": np.nan, "SR_LOW_CLOSE": np.nan}
    
    dr_df = pd.DataFrame(dr_bars)
    high = dr_df["high"].max()
    low = dr_df["low"].min()
    
    # IDR: Max of (highest open, highest close) for high, Min of (lowest open, lowest close) for low
    max_open = dr_df["open"].max()
    max_close = dr_df["close"].max()
    min_open = dr_df["open"].min()
    min_close = dr_df["close"].min()
    
    sr_high = max(max_open, max_close)
    sr_low = min(min_open, min_close)
    
    return {
        "DR_HIGH": float(high),
        "DR_LOW": float(low),
        "SR_HIGH_CLOSE": float(sr_high),
        "SR_LOW_CLOSE": float(sr_low),
    }

def find_first_breakout_bar(day_df: pd.DataFrame, dr_hi: float, dr_lo: float, breakout_start: str = "10:30") -> Optional[pd.Timestamp]:
    """Find first candle whose CLOSE is outside DR"""
    start_t = pd.Timestamp(breakout_start).time()
    times = sorted({ts.time() for ts in day_df.index})
    scan_times = [t for t in times if t >= start_t]
    
    for t in scan_times:
        time_str = f"{t.strftime('%H:%M')}"
        time_bars = day_df.between_time(time_str, time_str)
        if time_bars.empty:
            continue
        bar = time_bars.iloc[0]
        if bar["close"] > dr_hi or bar["close"] < dr_lo:
            return time_bars.index[0]
    return None

def calculate_r_multiple(exit_price: float, entry_price: float, stop_price: float, direction: str) -> float:
    """Calculate R-multiple for a trade exit"""
    if direction == "Long Breakout":
        pnl = exit_price - entry_price
        risk = abs(entry_price - stop_price)
    else:
        pnl = entry_price - exit_price
        risk = abs(stop_price - entry_price)
    return pnl / risk if risk > 0 else 0

def simulate_two_bullet_orders(
    day_df: pd.DataFrame,
    clusters: List[Cluster],
    direction: str,
    start_time: pd.Timestamp,
    session_close_time: pd.Timestamp,
    dr_high: float,
    dr_low: float,
    sr_high_close: float,
    sr_low_close: float,
    use_target_exit: bool = False,
    target_sd: float = 3.0,
) -> List[Tuple[FillResult, TradeExit]]:
    """Simulate two bullet orders - exit at stop, target (if enabled), or end of session"""
    work = day_df.loc[start_time:session_close_time]
    if work.empty or not clusters:
        return []
    
    # Calculate range scale (one standard deviation)
    range_scale = sr_high_close - sr_low_close
    
    # Calculate target price if using target exit
    if use_target_exit:
        if direction == "Long Breakout":
            target_price = sr_high_close + (target_sd * range_scale)
        else:  # Short Breakout
            target_price = sr_low_close - (target_sd * range_scale)
    else:
        target_price = None
    
    filled_trades = []
    
    # Convert cluster entries to absolute prices
    for cluster in clusters:
        if cluster.entry is None or cluster.stop is None:
            continue
        
        # Convert relative entries to absolute prices
        if direction == "Long Breakout":
            absolute_entry = sr_high_close + (cluster.entry * range_scale)
            absolute_stop = sr_high_close + (cluster.stop * range_scale)
        else:
            absolute_entry = sr_low_close - (cluster.entry * range_scale)
            absolute_stop = sr_low_close - (cluster.stop * range_scale)
        
        # Clamp entry to DR bounds
        if direction == "Long Breakout" and absolute_entry > dr_high:
            absolute_entry = dr_high
        if direction == "Short Breakout" and absolute_entry < dr_low:
            absolute_entry = dr_low
        
        # Check if breakout bar's close is above/below first cluster entry
        breakout_bar = work.iloc[0]
        breakout_close = breakout_bar['close']
        
        # Entry logic: If we confirm inside the first cluster, enter at DR_HIGH/DR_LOW (same as breakout level)
        if direction == "Long Breakout":
            if cluster.name == "First":
                # For first cluster: if breakout is inside cluster (between DR_HIGH and entry), enter at DR_HIGH
                if breakout_close >= dr_high and breakout_close < absolute_entry:
                    absolute_entry = dr_high  # Enter at DR_HIGH (range high, same as breakout level)
                    should_place_order = True
                else:
                    should_place_order = breakout_close >= absolute_entry
            else:
                should_place_order = breakout_close >= absolute_entry
        else:  # Short Breakout
            if cluster.name == "First":
                # For first cluster: if breakout is inside cluster (between entry and DR_LOW), enter at DR_LOW
                if breakout_close <= dr_low and breakout_close > absolute_entry:
                    absolute_entry = dr_low  # Enter at DR_LOW (range low, same as breakout level)
                    should_place_order = True
                else:
                    should_place_order = breakout_close <= absolute_entry
            else:
                should_place_order = breakout_close <= absolute_entry
        
        if not should_place_order:
            continue
        
        # Monitor for fill (limit order)
        fill_time = None
        for ts, bar in work.iterrows():
            hi, lo = bar["high"], bar["low"]
            filled = (lo <= absolute_entry <= hi)
            if filled:
                fill_time = ts
                break
        
        if fill_time is None:
            continue  # No fill, move to next cluster
        
        # Manage position until exit (stop or end of session)
        post_fill = work.loc[fill_time:]
        current_stop = absolute_stop
        
        fill = FillResult(
            filled=True,
            fill_time=fill_time,
            cluster_name=cluster.name,
            entry_price=float(absolute_entry),
            stop_price=float(absolute_stop)
        )
        
        for ts2, bar2 in post_fill.iterrows():
            hi2, lo2 = bar2["high"], bar2["low"]
            
            # Check stop first (stop has priority)
            if direction == "Long Breakout" and lo2 <= current_stop:
                exit_px = float(current_stop)
                pnl = exit_px - absolute_entry
                r_multiple = calculate_r_multiple(exit_px, absolute_entry, absolute_stop, direction)
                filled_trades.append((fill, TradeExit(ts2, "STOP", exit_px, pnl, r_multiple)))
                break
            
            if direction == "Short Breakout" and hi2 >= current_stop:
                exit_px = float(current_stop)
                pnl = absolute_entry - exit_px
                r_multiple = calculate_r_multiple(exit_px, absolute_entry, absolute_stop, direction)
                filled_trades.append((fill, TradeExit(ts2, "STOP", exit_px, pnl, r_multiple)))
                break
            
            # Check target if using target exit
            if use_target_exit and target_price is not None:
                if direction == "Long Breakout" and hi2 >= target_price:
                    exit_px = float(target_price)
                    pnl = exit_px - absolute_entry
                    r_multiple = calculate_r_multiple(exit_px, absolute_entry, absolute_stop, direction)
                    filled_trades.append((fill, TradeExit(ts2, "TARGET", exit_px, pnl, r_multiple)))
                    break
                
                if direction == "Short Breakout" and lo2 <= target_price:
                    exit_px = float(target_price)
                    pnl = absolute_entry - exit_px
                    r_multiple = calculate_r_multiple(exit_px, absolute_entry, absolute_stop, direction)
                    filled_trades.append((fill, TradeExit(ts2, "TARGET", exit_px, pnl, r_multiple)))
                    break
            
            # End of session (last bar)
            if ts2 >= session_close_time:
                exit_px = float(bar2["close"])
                if direction == "Long Breakout":
                    pnl = exit_px - absolute_entry
                else:
                    pnl = absolute_entry - exit_px
                r_multiple = calculate_r_multiple(exit_px, absolute_entry, absolute_stop, direction)
                filled_trades.append((fill, TradeExit(ts2, "SESSION_CLOSE", exit_px, pnl, r_multiple)))
                break
    
    return filled_trades

def run_simple_backtest(ohlcv_path: str, asset_name: str, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None, use_target_exit: bool = False, target_sd: float = 3.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run simple two bullet backtest for a single asset"""
    print(f"\nRunning backtest for {asset_name}...")
    
    df = load_ohlcv(ohlcv_path)
    
    # Filter by date range
    if start_date:
        if isinstance(start_date, pd.Timestamp) and start_date.tz is None:
            start_date = start_date.tz_localize(NY_TZ)
        df = df[df.index >= start_date]
    if end_date:
        if isinstance(end_date, pd.Timestamp) and end_date.tz is None:
            end_date = end_date.tz_localize(NY_TZ)
        df = df[df.index <= end_date]
    
    df = df.sort_index()
    df["session_date"] = df.index.tz_convert(NY_TZ).date
    
    # Build hardcoded clusters
    clusters = [
        Cluster(entry=CLUSTER_1_ENTRY, stop=CLUSTER_1_STOP, name="First"),
        Cluster(entry=CLUSTER_2_ENTRY, stop=CLUSTER_2_STOP, name="Second")
    ]
    
    trades = []
    
    for day, g in df.groupby("session_date"):
        day_df = g.tz_convert(NY_TZ)
        if day_df.empty:
            continue
        
        # Build Defining Range
        dr = build_defining_range(day_df)
        if np.isnan(dr["DR_HIGH"]):
            continue
        
        # Find breakout
        breakout_ts = find_first_breakout_bar(day_df, dr["DR_HIGH"], dr["DR_LOW"])
        if breakout_ts is None:
            continue
        
        direction = "Long Breakout" if day_df.loc[breakout_ts, "close"] > dr["DR_HIGH"] else "Short Breakout"
        
        # Session close time
        session_close = day_df.between_time("15:55", "15:55").index
        if len(session_close) > 0:
            session_close_time = session_close[0]
        else:
            session_close_time = day_df.index[-1]
        
        # Simulate trades
        filled_trades = simulate_two_bullet_orders(
            day_df=day_df,
            clusters=clusters,
            direction=direction,
            start_time=breakout_ts,
            session_close_time=session_close_time,
            dr_high=dr["DR_HIGH"],
            dr_low=dr["DR_LOW"],
            sr_high_close=dr["SR_HIGH_CLOSE"],
            sr_low_close=dr["SR_LOW_CLOSE"],
            use_target_exit=use_target_exit,
            target_sd=target_sd
        )
        
        # Record trades
        for fill, exit_res in filled_trades:
            trades.append({
                "asset": asset_name,
                "date": pd.Timestamp(day).date(),
                "dr_high": dr["DR_HIGH"],
                "dr_low": dr["DR_LOW"],
                "sr_high_close": dr["SR_HIGH_CLOSE"],
                "sr_low_close": dr["SR_LOW_CLOSE"],
                "breakout_time": breakout_ts,
                "direction": direction,
                "cluster": fill.cluster_name,
                "entry_time": fill.fill_time,
                "entry_price": fill.entry_price,
                "stop_price": fill.stop_price,
                "exit_time": exit_res.exit_time,
                "exit_reason": exit_res.exit_reason,
                "exit_price": exit_res.exit_price,
                "pnl": exit_res.pnl,
                "r_multiple": exit_res.r_multiple,
            })
    
    trades_df = pd.DataFrame(trades)
    print(f"  Found {len(trades_df)} trades for {asset_name}")
    
    return trades_df

def run_multi_asset_backtest(assets: List[Dict[str, str]], start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None, use_target_exit: bool = False, target_sd: float = 3.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run backtest for multiple assets and combine results"""
    print("\n" + "="*60)
    print("Running Multi-Asset Two Bullet Backtest")
    print("="*60)
    print(f"\nHardcoded Parameters:")
    print(f"  Cluster 1: Entry {CLUSTER_1_ENTRY}, Stop {CLUSTER_1_STOP}")
    print(f"  Cluster 2: Entry {CLUSTER_2_ENTRY}, Stop {CLUSTER_2_STOP}")
    if use_target_exit:
        print(f"  Exit: Stop, {target_sd} SD Target, or End of Session")
    else:
        print(f"  Exit: Stop or End of Session")
    print("="*60)
    
    all_trades = []
    
    for asset in assets:
        asset_name = asset["name"]
        ohlcv_path = asset["ohlcv"]
        
        trades_df = run_simple_backtest(ohlcv_path, asset_name, start_date, end_date, use_target_exit=use_target_exit, target_sd=target_sd)
        if not trades_df.empty:
            all_trades.append(trades_df)
    
    # Combine all trades
    if all_trades:
        combined_trades_df = pd.concat(all_trades, ignore_index=True)
    else:
        combined_trades_df = pd.DataFrame()
    
    # Summary stats
    if not combined_trades_df.empty:
        wins = combined_trades_df["pnl"] > 0
        
        # Calculate Sharpe ratio (pre-commission) from R-multiples
        sharpe_pre_commission = calculate_sharpe_from_r_multiples(combined_trades_df, r_metric="r_multiple")
        
        summary_data = {
            "metric": ["Total Trades", "Win Rate", "Avg PnL", "Total PnL", "Avg R-Multiple", "Sharpe Ratio (Pre-Commission)"],
            "value": [
                len(combined_trades_df),
                f"{wins.mean() * 100:.2f}%" if len(combined_trades_df) > 0 else "0.00%",
                combined_trades_df["pnl"].mean() if len(combined_trades_df) > 0 else 0.0,
                combined_trades_df["pnl"].sum() if len(combined_trades_df) > 0 else 0.0,
                combined_trades_df["r_multiple"].mean() if combined_trades_df["r_multiple"].notna().any() else 0.0,
                f"{sharpe_pre_commission:.3f}" if sharpe_pre_commission is not None else "N/A"
            ]
        }
        
        # Add per-asset breakdown
        for asset_name in combined_trades_df["asset"].unique():
            asset_trades = combined_trades_df[combined_trades_df["asset"] == asset_name]
            asset_wins = asset_trades["pnl"] > 0
            summary_data["metric"].extend([
                f"{asset_name} Trades",
                f"{asset_name} Win Rate",
                f"{asset_name} Avg R"
            ])
            summary_data["value"].extend([
                len(asset_trades),
                f"{asset_wins.mean() * 100:.2f}%" if len(asset_trades) > 0 else "0.00%",
                asset_trades["r_multiple"].mean() if asset_trades["r_multiple"].notna().any() else 0.0
            ])
        
        summary = pd.DataFrame(summary_data)
    else:
        summary = pd.DataFrame({
            "metric": ["Total Trades", "Win Rate", "Avg PnL", "Total PnL", "Avg R-Multiple"],
            "value": [0, "0.00%", 0.0, 0.0, 0.0]
        })
    
    print(f"\nBacktest completed. Found {len(combined_trades_df)} total trades.")
    return combined_trades_df, summary

def plot_equity_curves(trades_df: pd.DataFrame, output_dir: str = "."):
    """Plot equity curves split by asset"""
    if trades_df.empty:
        print("No trades to plot.")
        return
    
    # Sort by date for cumulative calculation
    trades_df = trades_df.sort_values('date').copy()
    
    # Calculate cumulative R-multiples for each asset
    assets = trades_df['asset'].unique()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Different colors for each asset
    asset_colors = {asset: colors[i % len(colors)] for i, asset in enumerate(assets)}
    
    # Plot each asset's equity curve
    for asset in assets:
        asset_trades = trades_df[trades_df['asset'] == asset].copy()
        asset_trades = asset_trades.sort_values('date')
        
        # Calculate cumulative R
        asset_trades['cumulative_r'] = asset_trades['r_multiple'].cumsum()
        
        # Convert dates to datetime for plotting
        asset_trades['date_dt'] = pd.to_datetime(asset_trades['date'])
        
        # Plot this asset's equity curve
        ax.plot(asset_trades['date_dt'], asset_trades['cumulative_r'], 
                label=f'{asset}', linewidth=2, color=asset_colors[asset], alpha=0.8)
    
    # Calculate and plot combined equity curve
    all_trades_sorted = trades_df.sort_values('date').copy()
    all_trades_sorted['cumulative_r'] = all_trades_sorted['r_multiple'].cumsum()
    all_trades_sorted['date_dt'] = pd.to_datetime(all_trades_sorted['date'])
    
    ax.plot(all_trades_sorted['date_dt'], all_trades_sorted['cumulative_r'], 
            label='Combined', linewidth=3, color='black', linestyle='--', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative R-Multiple', fontsize=12, fontweight='bold')
    ax.set_title('Equity Curves by Asset (Cumulative R-Multiple)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    # Format x-axis dates
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    import os
    output_path = os.path.join(output_dir, 'equity_curves_by_asset.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nEquity curve plot saved as '{output_path}'")
    
    # Close plot
    plt.close()

# ----------------------------
# Commission-aware functions
# ----------------------------
def calculate_position_size(
    stop_size: float,
    tick_size: float,
    tick_value: float,
    risk_per_trade: float,
    equity: float,
    commission_per_side: float = 3.0,
    max_dollar_risk: float = None
) -> Tuple[int, float, float]:
    """
    Calculate position size based on stop size and risk per trade.
    Caps max dollar risk if specified.
    
    Returns:
        contracts: Number of contracts to trade
        dollar_risk: Dollar amount at risk per trade
        commission_cost: Total commission cost (round trip)
    """
    # Calculate dollar risk per contract based on stop size
    stop_ticks = stop_size / tick_size
    dollar_risk_per_contract = stop_ticks * tick_value
    
    # Avoid division by zero and infinity
    if dollar_risk_per_contract <= 0 or not np.isfinite(dollar_risk_per_contract):
        return 0, 0.0, 0.0
    
    # Calculate how much dollar risk we want based on equity
    target_dollar_risk = risk_per_trade * equity
    
    # Cap at max dollar risk if specified
    if max_dollar_risk is not None and target_dollar_risk > max_dollar_risk:
        target_dollar_risk = max_dollar_risk
    
    # Calculate number of contracts (round down to avoid over-risking)
    if not np.isfinite(target_dollar_risk / dollar_risk_per_contract):
        return 0, 0.0, 0.0
    
    contracts = int(target_dollar_risk / dollar_risk_per_contract)
    
    # Ensure minimum of 1 contract if we have any risk
    if contracts < 1 and target_dollar_risk > 0:
        contracts = 1
    
    # Calculate actual dollar risk and commission
    actual_dollar_risk = contracts * dollar_risk_per_contract
    commission_cost = contracts * commission_per_side * 2  # Round trip
    
    return contracts, actual_dollar_risk, commission_cost

def build_compounded_equity_with_commissions(
    trades_df: pd.DataFrame, 
    r_metric: str, 
    risk_per_trade: float, 
    start_equity: float = 100000.0,
    asset_config: Dict[str, Dict] = None,
    max_dollar_risk: float = None,
    commission_threshold: float = None
) -> pd.DataFrame:
    """Build compounded equity curve with realistic position sizing and commission costs."""
    if trades_df.empty or r_metric not in trades_df.columns:
        return pd.DataFrame(columns=["exit_time", "equity", "contracts", "commission_cost"])
    
    # Default asset configs if not provided
    if asset_config is None:
        asset_config = {
            "CL": {"tick_size": 0.01, "tick_value": 10.0},
            "ES": {"tick_size": 0.25, "tick_value": 12.50},
            "NQ": {"tick_size": 0.25, "tick_value": 5.0},
            "GC": {"tick_size": 0.1, "tick_value": 10.0},
            "J1": {"tick_size": 0.0000005, "tick_value": 6.25},
        }
    
    df = trades_df[["exit_time", r_metric, "asset", "stop_price", "entry_price"]].dropna().sort_values("exit_time").copy()
    if df.empty:
        return pd.DataFrame(columns=["exit_time", "equity", "contracts", "commission_cost"])
    
    equity = float(start_equity)
    curve = []
    
    for _, row in df.iterrows():
        r = float(row[r_metric])
        asset = row["asset"]
        stop_price = float(row["stop_price"])
        entry_price = float(row["entry_price"])
        
        # Get asset configuration
        config = asset_config.get(asset, {"tick_size": 0.01, "tick_value": 10.0})
        tick_size = config["tick_size"]
        tick_value = config["tick_value"]
        
        # Calculate stop size in price terms
        stop_size = abs(entry_price - stop_price)
        
        # Calculate position size and commission
        contracts, dollar_risk, commission_cost = calculate_position_size(
            stop_size=stop_size,
            tick_size=tick_size,
            tick_value=tick_value,
            risk_per_trade=risk_per_trade,
            equity=equity,
            commission_per_side=3.0,
            max_dollar_risk=max_dollar_risk
        )
        
        # Calculate commission to risk ratio
        commission_ratio = commission_cost / dollar_risk if dollar_risk > 0 else 0.0
        
        # Filter: Skip trade if commission > threshold of risk
        if commission_threshold is not None and dollar_risk > 0:
            if commission_ratio > commission_threshold:
                # Skip this trade - don't update equity, don't add to curve
                # Equity remains unchanged for next trade
                continue
        
        # Calculate P&L in dollars (R-multiple * dollar risk)
        dollar_pnl = r * dollar_risk
        
        # Update equity: P&L - commission (only for trades that pass filter)
        equity += dollar_pnl - commission_cost
        
        curve.append({
            "exit_time": row["exit_time"],
            "equity": equity,
            "contracts": contracts,
            "commission_cost": commission_cost,
            "dollar_pnl": dollar_pnl,
            "dollar_risk": dollar_risk,
            "commission_ratio": commission_ratio
        })
    
    return pd.DataFrame(curve)

def plot_compounded_equity_with_commissions(
    combined_trades: pd.DataFrame, 
    r_metric: str, 
    risk_per_trade: float, 
    start_equity: float = 100000.0,
    asset_config: Dict[str, Dict] = None,
    title_prefix: str = "Compounded Equity (with Commissions)",
    max_dollar_risk: float = None,
    output_dir: str = "."
) -> None:
    """Plot compounded equity curve with realistic position sizing and commission costs.
    Also plots filtered curve skipping trades where commission > 15% of risk."""
    curve = build_compounded_equity_with_commissions(
        combined_trades, 
        r_metric=r_metric, 
        risk_per_trade=risk_per_trade, 
        start_equity=start_equity,
        asset_config=asset_config,
        max_dollar_risk=max_dollar_risk
    )
    if curve.empty:
        return
    
    # Rebuild equity curve from scratch using only filtered trades (commission <= 15% of risk)
    # This filters during the build process, so equity only compounds from filtered trades
    curve_filtered_df = build_compounded_equity_with_commissions(
        combined_trades,
        r_metric=r_metric,
        risk_per_trade=risk_per_trade,
        start_equity=start_equity,
        asset_config=asset_config,
        max_dollar_risk=max_dollar_risk,
        commission_threshold=0.15  # Filter out trades where commission > 15% of risk
    )
    
    # Calculate skipped count (difference between all trades and filtered trades)
    skipped_count = len(curve) - len(curve_filtered_df)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    
    # Top plot: Equity curves (both original and filtered) - LOGARITHMIC SCALE
    ax1.plot(curve["exit_time"], curve["equity"], label=f"All Trades (Risk {risk_per_trade*100:.1f}%)", linewidth=2, color='blue')
    ax1.plot(curve_filtered_df["exit_time"], curve_filtered_df["equity"], label=f"Filtered (Comm < 15% of Risk)", linewidth=2, color='green', linestyle='--')
    ax1.set_title(f"{title_prefix} - Risk {risk_per_trade*100:.1f}%")
    ax1.set_ylabel("Equity ($)")
    ax1.set_yscale('log')  # Logarithmic scale
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend()
    
    # Bottom plot: Commission costs over time (both original and filtered)
    ax2.plot(curve["exit_time"], curve["commission_cost"], color="red", alpha=0.7, label="Commission per trade (All)", linewidth=1.5)
    ax2.plot(curve_filtered_df["exit_time"], curve_filtered_df["commission_cost"], color="orange", alpha=0.7, label="Commission per trade (Filtered)", linewidth=1.5, linestyle='--')
    ax2.set_ylabel("Commission ($)")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    import os
    output_path = os.path.join(output_dir, 'equity_with_commissions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Commission-aware equity plot saved as '{output_path}'")
    plt.close()  # Close instead of show to avoid blocking
    
    # Print summary statistics
    total_commission = curve["commission_cost"].sum()
    total_pnl = curve["dollar_pnl"].sum()
    net_pnl = total_pnl - total_commission
    final_equity = curve["equity"].iloc[-1]
    total_return = (final_equity - start_equity) / start_equity * 100
    
    # Filtered curve statistics
    total_commission_filtered = curve_filtered_df["commission_cost"].sum()
    total_pnl_filtered = curve_filtered_df["dollar_pnl"].sum()
    net_pnl_filtered = total_pnl_filtered - total_commission_filtered
    final_equity_filtered = curve_filtered_df["equity"].iloc[-1]
    total_return_filtered = (final_equity_filtered - start_equity) / start_equity * 100
    
    print(f"\nCommission-Aware Summary (Risk {risk_per_trade*100:.1f}%):")
    print(f"Starting Equity: ${start_equity:,.2f}")
    print(f"Final Equity: ${final_equity:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Total Commissions: ${total_commission:,.2f}")
    print(f"Net P&L: ${net_pnl:,.2f}")
    print(f"Commission as % of P&L: {(total_commission/abs(total_pnl)*100) if total_pnl != 0 else 0:.2f}%")
    
    print(f"\nFiltered Summary (Skipping trades where commission > 15% of risk):")
    print(f"Skipped Trades: {skipped_count}")
    print(f"Starting Equity: ${start_equity:,.2f}")
    print(f"Final Equity: ${final_equity_filtered:,.2f}")
    print(f"Total Return: {total_return_filtered:.2f}%")
    print(f"Total P&L: ${total_pnl_filtered:,.2f}")
    print(f"Total Commissions: ${total_commission_filtered:,.2f}")
    print(f"Net P&L: ${net_pnl_filtered:,.2f}")
    if total_pnl_filtered != 0:
        print(f"Commission as % of P&L: {(total_commission_filtered/abs(total_pnl_filtered)*100):.2f}%")
    
    # Show commission ratio statistics for filtered trades
    if not curve_filtered_df.empty and "commission_ratio" in curve_filtered_df.columns:
        avg_comm_ratio = curve_filtered_df["commission_ratio"].mean() * 100
        max_comm_ratio = curve_filtered_df["commission_ratio"].max() * 100
        print(f"Avg Commission/Risk Ratio (filtered): {avg_comm_ratio:.2f}%")
        print(f"Max Commission/Risk Ratio (filtered): {max_comm_ratio:.2f}%")
    
    # Calculate Sharpe ratios for all three scenarios
    print("\n" + "="*60)
    print("SHARPE RATIO COMPARISON")
    print("="*60)
    
    # 1. No commissions (from R-multiples)
    sharpe_no_comm = calculate_sharpe_from_r_multiples(combined_trades, r_metric=r_metric)
    
    # 2. With commissions (all trades) - calculate from net P&L per day
    if not curve.empty:
        # Create net P&L DataFrame from curve
        net_pnl_df = curve[["exit_time", "dollar_pnl", "commission_cost"]].copy()
        net_pnl_df["net_pnl"] = net_pnl_df["dollar_pnl"] - net_pnl_df["commission_cost"]
        sharpe_with_comm = calculate_sharpe_from_net_pnl(net_pnl_df)
    else:
        sharpe_with_comm = None
    
    # 3. Filtered trades (commission <= 15%) - calculate from net P&L per day
    if not curve_filtered_df.empty:
        net_pnl_filtered_df = curve_filtered_df[["exit_time", "dollar_pnl", "commission_cost"]].copy()
        net_pnl_filtered_df["net_pnl"] = net_pnl_filtered_df["dollar_pnl"] - net_pnl_filtered_df["commission_cost"]
        sharpe_filtered = calculate_sharpe_from_net_pnl(net_pnl_filtered_df)
    else:
        sharpe_filtered = None
    
    print(f"1. No Commissions (R-multiples):           {sharpe_no_comm:.3f}" if sharpe_no_comm is not None else "1. No Commissions (R-multiples):           N/A")
    print(f"2. With Commissions (All Trades):          {sharpe_with_comm:.3f}" if sharpe_with_comm is not None else "2. With Commissions (All Trades):          N/A")
    print(f"3. Filtered (Comm <= 15% of Risk):        {sharpe_filtered:.3f}" if sharpe_filtered is not None else "3. Filtered (Comm <= 15% of Risk):        N/A")

def generate_commission_aware_trade_log(
    trades_df: pd.DataFrame, 
    r_metric: str, 
    risk_per_trade: float, 
    start_equity: float = 100000.0,
    asset_config: Dict[str, Dict] = None,
    output_path: str = "commission_aware_trades.csv",
    max_dollar_risk: float = None
) -> None:
    """Generate detailed trade log with commission costs, position sizing, and results."""
    if trades_df.empty or r_metric not in trades_df.columns:
        print("No trades to process for commission-aware log")
        return
    
    # Default asset configs if not provided
    if asset_config is None:
        asset_config = {
            "CL": {"tick_size": 0.01, "tick_value": 10.0},
            "ES": {"tick_size": 0.25, "tick_value": 12.50},
            "NQ": {"tick_size": 0.25, "tick_value": 5.0},
            "GC": {"tick_size": 0.1, "tick_value": 10.0},
            "J1": {"tick_size": 0.0000005, "tick_value": 6.25},
        }
    
    # Prepare data
    required_cols = ["exit_time", r_metric, "asset", "stop_price", "entry_price", "exit_price"]
    available_cols = [col for col in required_cols if col in trades_df.columns]
    df = trades_df[available_cols].dropna().sort_values("exit_time").copy()
    if df.empty:
        print("No valid trades found for commission-aware log")
        return
    
    equity = float(start_equity)
    detailed_trades = []
    
    for _, row in df.iterrows():
        r = float(row[r_metric])
        asset = row["asset"]
        stop_price = float(row["stop_price"])
        entry_price = float(row["entry_price"])
        exit_price = float(row["exit_price"])
        
        # Get asset configuration
        config = asset_config.get(asset, {"tick_size": 0.01, "tick_value": 10.0})
        tick_size = config["tick_size"]
        tick_value = config["tick_value"]
        
        # Calculate stop size in price terms
        stop_size = abs(entry_price - stop_price)
        
        # Calculate position size and commission
        contracts, dollar_risk, commission_cost = calculate_position_size(
            stop_size=stop_size,
            tick_size=tick_size,
            tick_value=tick_value,
            risk_per_trade=risk_per_trade,
            equity=equity,
            commission_per_side=3.0,
            max_dollar_risk=max_dollar_risk
        )
        
        # Calculate P&L in dollars (R-multiple * dollar risk)
        dollar_pnl = r * dollar_risk
        
        # Calculate net P&L (after commission)
        net_pnl = dollar_pnl - commission_cost
        
        # Update equity for next trade
        equity += net_pnl
        
        # Create detailed trade record
        trade_record = {
            "date": row.get("date", row.get("exit_time", "")),
            "exit_time": row["exit_time"],
            "asset": asset,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "exit_price": exit_price,
            "exit_reason": row.get("exit_reason", ""),
            "cluster": row.get("cluster", ""),
            "direction": row.get("direction", ""),
            "r_multiple": r,
            "stop_size_ticks": stop_size / tick_size,
            "stop_size_dollars": stop_size * tick_value / tick_size,
            "tick_size": tick_size,
            "tick_value": tick_value,
            "contracts": contracts,
            "dollar_risk": dollar_risk,
            "gross_pnl": dollar_pnl,
            "commission_cost": commission_cost,
            "net_pnl": net_pnl,
            "equity_before": equity - net_pnl,
            "equity_after": equity,
            "risk_per_trade_pct": risk_per_trade * 100
        }
        
        detailed_trades.append(trade_record)
    
    # Create DataFrame and save
    detailed_df = pd.DataFrame(detailed_trades)
    detailed_df.to_csv(output_path, index=False)
    
    print(f"Generated detailed commission-aware trade log: {output_path}")
    print(f"Total trades: {len(detailed_df)}")
    print(f"Total contracts traded: {detailed_df['contracts'].sum():,}")
    print(f"Total commission paid: ${detailed_df['commission_cost'].sum():,.2f}")
    print(f"Average commission per trade: ${detailed_df['commission_cost'].mean():.2f}")
    print(f"Average contracts per trade: {detailed_df['contracts'].mean():.1f}")

def calculate_sharpe_from_r_multiples(trades_df: pd.DataFrame, r_metric: str = "r_multiple") -> Optional[float]:
    """Calculate annualized Sharpe ratio from R-multiples using daily returns"""
    if trades_df.empty or r_metric not in trades_df.columns:
        return None
    
    # Group trades by exit date and sum R-multiples per day
    df = trades_df[[r_metric, "exit_time"]].copy()
    # Convert exit_time to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df["exit_time"]):
        df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["date"] = df["exit_time"].dt.date
    
    # Sum R-multiples per day
    daily_r = df.groupby("date")[r_metric].sum().reset_index()
    daily_r.columns = ["date", "daily_R"]
    
    if len(daily_r) < 2:
        return None
    
    # Calculate mean and std of daily R
    mean_daily_r = daily_r["daily_R"].mean()
    std_daily_r = daily_r["daily_R"].std(ddof=1)
    
    if std_daily_r == 0:
        return None
    
    # Annualized Sharpe = (mean / std) * sqrt(252 trading days)
    sharpe = (mean_daily_r / std_daily_r) * np.sqrt(252)
    
    return sharpe

def calculate_sharpe_from_curve(curve_df: pd.DataFrame) -> Optional[float]:
    """Calculate annualized Sharpe ratio from equity curve using daily returns"""
    if curve_df.empty or "exit_time" not in curve_df.columns or "equity" not in curve_df.columns:
        return None
    
    # Convert exit_time to datetime if needed
    df = curve_df[["exit_time", "equity"]].copy()
    if not pd.api.types.is_datetime64_any_dtype(df["exit_time"]):
        df["exit_time"] = pd.to_datetime(df["exit_time"], errors='coerce')
    
    df = df.dropna(subset=["exit_time"])
    if len(df) < 2:
        return None
    
    # Extract date
    df["date"] = df["exit_time"].dt.date
    
    # Calculate daily returns from equity
    df = df.sort_values("date")
    df["daily_return"] = df["equity"].pct_change()
    df = df.dropna(subset=["daily_return"])
    
    if len(df) < 2:
        return None
    
    # Calculate mean and std of daily returns
    mean_daily_return = df["daily_return"].mean()
    std_daily_return = df["daily_return"].std(ddof=1)
    
    if std_daily_return == 0:
        return None
    
    # Annualized Sharpe = (mean / std) * sqrt(252 trading days)
    sharpe = (mean_daily_return / std_daily_return) * np.sqrt(252)
    
    return sharpe

def calculate_sharpe_from_net_pnl(commission_trades_df: pd.DataFrame) -> Optional[float]:
    """Calculate annualized Sharpe ratio from net P&L after commissions using daily returns"""
    if commission_trades_df.empty or "net_pnl" not in commission_trades_df.columns:
        return None
    
    # Determine which date column to use
    date_col = "exit_time" if "exit_time" in commission_trades_df.columns else "date"
    if date_col not in commission_trades_df.columns:
        return None
    
    # Group by date and sum net P&L per day
    df = commission_trades_df[["net_pnl", date_col]].copy()
    # Convert date column to datetime if it's not already
    try:
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            # Try parsing - handle timezone-aware strings like "2019-01-02 10:30:00-05:00"
            # First, try to parse as-is
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            # If that didn't work (still object type), try removing timezone suffix
            if df[date_col].dtype == 'object':
                # Remove timezone offset (e.g., "-05:00" or "+00:00")
                df[date_col] = df[date_col].astype(str).str.replace(r'[+-]\d{2}:\d{2}$', '', regex=True)
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        # Remove any rows where date conversion failed
        df = df.dropna(subset=[date_col])
        # Extract date part
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df["date"] = df[date_col].dt.date
        else:
            # If still not datetime, try extracting date from string
            df["date"] = pd.to_datetime(df[date_col].astype(str).str.split(' ').str[0]).dt.date
    except Exception:
        return None
    
    # Sum net P&L per day
    daily_pnl = df.groupby("date")["net_pnl"].sum().reset_index()
    daily_pnl.columns = ["date", "daily_net_pnl"]
    
    if len(daily_pnl) < 2:
        return None
    
    # Calculate mean and std of daily net P&L
    mean_daily_pnl = daily_pnl["daily_net_pnl"].mean()
    std_daily_pnl = daily_pnl["daily_net_pnl"].std(ddof=1)
    
    if std_daily_pnl == 0:
        return None
    
    # Annualized Sharpe = (mean / std) * sqrt(252 trading days)
    sharpe = (mean_daily_pnl / std_daily_pnl) * np.sqrt(252)
    
    return sharpe

if __name__ == "__main__":
    # Asset configurations (with tick size and tick value for commission calculations)
    assets = [
        {"name": "ES", "ohlcv": "/Users/orlandocantoni/Desktop/RBC auto model/ES_full_5min.csv", "tick_size": 0.25, "tick_value": 12.50},
        {"name": "NQ", "ohlcv": "/Users/orlandocantoni/Desktop/RBC auto model/NQ_full_5min.csv", "tick_size": 0.25, "tick_value": 5.0},
        {"name": "GC", "ohlcv": "/Users/orlandocantoni/Desktop/RBC auto model/GC_full_5min.csv", "tick_size": 0.1, "tick_value": 10.0},
        {"name": "CL", "ohlcv": "/Users/orlandocantoni/Desktop/RBC auto model/CL_full_5min.csv", "tick_size": 0.01, "tick_value": 10.0},
        {"name": "J1", "ohlcv": "/Users/orlandocantoni/Desktop/RBC auto model/J1_full_5min.csv", "tick_size": 0.0000005, "tick_value": 6.25},  # 6J
    ]
    
    # Run from 2019
    start_date = pd.Timestamp('2008-01-10', tz=NY_TZ)
    end_date = None  # Run to end
    
    # Configuration
    target_sd = 2.5
    risk_per_trade = 0.002  # 0.2% risk per trade
    max_dollar_risk = 5000.0  # Cap max risk at $5000 per trade
    start_equity = 100000.0
    
    # Output directory
    import os
    output_dir = "backtest_results_2.5sd"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"BACKTEST: 2.5 SD Target, 0.2% Risk (Max $5,000 per trade) - From 2019")
    print("="*80)
    
    # Run backtest with 2.5 SD target
    trades_df, summary = run_multi_asset_backtest(
        assets,
        start_date=start_date,
        end_date=end_date,
        use_target_exit=True,
        target_sd=target_sd
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))
    
    # Save trades
    if not trades_df.empty:
        trades_csv_path = os.path.join(output_dir, "trades_two_bullet_2.5_sd.csv")
        trades_df.to_csv(trades_csv_path, index=False)
        print(f"\nTrades saved to {trades_csv_path}")
        
        # Show exit reasons breakdown
        print("\n" + "="*60)
        print("EXIT REASONS (All Assets Combined)")
        print("="*60)
        print(trades_df['exit_reason'].value_counts())
        
        # Create asset config lookup for commission calculations
        asset_config = {asset["name"]: {"tick_size": asset["tick_size"], "tick_value": asset["tick_value"]} for asset in assets}
        
        # Plot R-multiple equity curves
        print("\n" + "="*60)
        print("GENERATING R-MULTIPLE EQUITY CURVES")
        print("="*60)
        plot_equity_curves(trades_df, output_dir=output_dir)
        
        # Plot commission-aware equity curve with $5000 cap
        print("\n" + "="*60)
        print("GENERATING EQUITY CURVE (0.2% Risk, Max $5,000 per trade)")
        print("="*60)
        plot_compounded_equity_with_commissions(
            trades_df,
            r_metric="r_multiple",
            risk_per_trade=risk_per_trade,
            start_equity=start_equity,
            asset_config=asset_config,
            title_prefix="Compounded Equity (with Commissions, $5K Max Risk)",
            max_dollar_risk=max_dollar_risk,
            output_dir=output_dir
        )
    
