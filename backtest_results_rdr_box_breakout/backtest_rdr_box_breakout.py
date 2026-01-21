#!/usr/bin/env python3
"""
RDR Overnight Breakout - 5R Strategy
Matches plot style from backtest_results_2.5sd:
- Equity curves by asset (R-multiples)
- Compounded equity with commissions ($500 max risk)
- All trades vs 20% commission filtered
"""

import csv
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ----------------------------
# Configuration
# ----------------------------
BASE_DIR = '/Users/orlandocantoni/Desktop/RBC auto model'
DATA_DIR = f'{BASE_DIR}/5min Jan 2025'
OUTPUT_DIR = f'{BASE_DIR}/backtest_results_overnight'

# 5R Strategy: divisor = 6 (risk 1 to make 5)
DIVISOR = 6.0  
STOP_RATIO = 0.75

ASSETS = {
    'ES': f'{DATA_DIR}/ES_full_5min2008-2025.csv',
    'NQ': f'{DATA_DIR}/NQ_full_5min2008-2025.csv',
    'J1': f'{DATA_DIR}/J1_full_5min2008-2025.csv',
    'CL': f'{DATA_DIR}/CL_full_5min2008-2025.csv',
    'GC': f'{DATA_DIR}/GC_full_5min2008-2025.csv',
}

ASSET_CONFIG = {
    'ES': {'tick_size': 0.25, 'tick_value': 12.50, 'commission': 2.00},
    'NQ': {'tick_size': 0.25, 'tick_value': 5.0, 'commission': 2.00},
    'J1': {'tick_size': 0.0000005, 'tick_value': 6.25, 'commission': 2.00},
    'CL': {'tick_size': 0.01, 'tick_value': 10.0, 'commission': 2.25},
    'GC': {'tick_size': 0.1, 'tick_value': 10.0, 'commission': 2.25},
}

# ----------------------------
# Core Backtest (Row-by-Row)
# ----------------------------
def run_backtest(start_date=None, end_date=None):
    """Run 5R overnight breakout backtest"""
    all_trades = []
    
    for asset_name, file_path in ASSETS.items():
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        print(f"Loading {asset_name}...")
        rows = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        
        # Get tick size for this asset (for 1 tick slippage)
        tick_size = ASSET_CONFIG.get(asset_name, {'tick_size': 0.01})['tick_size']
        
        # State
        range_high = None
        range_low = None
        collecting_range = False
        setup_active = False
        setup_dir = None
        setup_extreme = None
        setup_stop = None
        active_trade = None
        
        for row in rows:
            dt = datetime.datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
            t = dt.time()
            
            if start_date and dt.date() < start_date:
                continue
            if end_date and dt.date() > end_date:
                continue
            
            high = float(row['high'])
            low = float(row['low'])
            close = float(row['close'])
            
            # Trade Management
            if active_trade:
                exit_triggered = False
                exit_reason = None
                exit_price = None
                
                if active_trade['direction'] == 'LONG':
                    if low <= active_trade['stop']:
                        exit_triggered, exit_reason, exit_price = True, 'STOP', active_trade['stop']
                    elif high >= active_trade['target']:
                        exit_triggered, exit_reason, exit_price = True, 'TARGET', active_trade['target']
                else:
                    if high >= active_trade['stop']:
                        exit_triggered, exit_reason, exit_price = True, 'STOP', active_trade['stop']
                    elif low <= active_trade['target']:
                        exit_triggered, exit_reason, exit_price = True, 'TARGET', active_trade['target']
                
                if not exit_triggered and dt >= active_trade['time_exit']:
                    exit_triggered, exit_reason, exit_price = True, 'TIME', close
                
                if exit_triggered:
                    if active_trade['direction'] == 'LONG':
                        pnl = exit_price - active_trade['entry_price']
                    else:
                        pnl = active_trade['entry_price'] - exit_price
                    
                    risk = abs(active_trade['entry_price'] - active_trade['stop'])
                    r_pnl = pnl / risk if risk > 0 else 0
                    
                    all_trades.append({
                        'asset': asset_name,
                        'date': dt.date(),
                        'exit_time': dt,
                        'direction': active_trade['direction'],
                        'entry_price': active_trade['entry_price'],
                        'stop_price': active_trade['stop'],
                        'target_price': active_trade['target'],
                        'exit_price': exit_price,
                        'r_pnl': r_pnl,
                        'reason': exit_reason,
                    })
                    active_trade = None
            
            # Range Collection (09:30 - 15:55)
            if t == datetime.time(9, 30):
                collecting_range = True
                range_high = high
                range_low = low
                setup_active = False
            elif collecting_range:
                if t < datetime.time(15, 55):
                    range_high = max(range_high, high)
                    range_low = min(range_low, low)
                else:
                    collecting_range = False
                    setup_active = True
                    setup_dir = None
                    setup_extreme = None
                    setup_stop = None
            
            # Setup Monitoring
            if setup_active and not active_trade:
                if t == datetime.time(9, 25):
                    setup_active = False
                    continue
                
                if setup_dir is None:
                    if high > range_high:
                        setup_dir = 'LONG'
                        setup_extreme = high
                        box_height = range_high - range_low
                        setup_stop = range_low + (box_height * STOP_RATIO)
                    elif low < range_low:
                        setup_dir = 'SHORT'
                        setup_extreme = low
                        box_height = range_high - range_low
                        setup_stop = range_high - (box_height * STOP_RATIO)
                
                if setup_dir is not None:
                    if setup_dir == 'LONG':
                        setup_extreme = max(setup_extreme, high)
                        dist = setup_extreme - setup_stop
                        if dist <= 0:
                            continue
                        risk = dist / DIVISOR
                        entry_limit = setup_stop + risk
                        
                        if low <= entry_limit:
                            time_exit = datetime.datetime.combine(dt.date(), datetime.time(15, 55))
                            if t > datetime.time(15, 55):
                                time_exit += datetime.timedelta(days=1)
                            # Add 1 tick slippage for LONG entry
                            entry_with_slippage = entry_limit + tick_size
                            active_trade = {
                                'entry_time': dt,
                                'entry_price': entry_with_slippage,
                                'direction': 'LONG',
                                'stop': setup_stop,
                                'target': setup_extreme,
                                'time_exit': time_exit
                            }
                            setup_active = False
                    else:
                        setup_extreme = min(setup_extreme, low)
                        dist = setup_stop - setup_extreme
                        if dist <= 0:
                            continue
                        risk = dist / DIVISOR
                        entry_limit = setup_stop - risk
                        
                        if high >= entry_limit:
                            time_exit = datetime.datetime.combine(dt.date(), datetime.time(15, 55))
                            if t > datetime.time(15, 55):
                                time_exit += datetime.timedelta(days=1)
                            # Subtract 1 tick slippage for SHORT entry
                            entry_with_slippage = entry_limit - tick_size
                            active_trade = {
                                'entry_time': dt,
                                'entry_price': entry_with_slippage,
                                'direction': 'SHORT',
                                'stop': setup_stop,
                                'target': setup_extreme,
                                'time_exit': time_exit
                            }
                            setup_active = False
        
        print(f"  Found {len([t for t in all_trades if t['asset'] == asset_name])} trades")
    
    return all_trades

# ----------------------------
# Plot 1: Equity Curves by Asset (R-Multiple)
# ----------------------------
def plot_equity_by_asset(trades, output_path):
    """Plot cumulative R-multiple by asset"""
    if not trades:
        print("No trades to plot")
        return
    
    trades_sorted = sorted(trades, key=lambda x: x['exit_time'])
    
    # Group by asset
    by_asset = defaultdict(list)
    for t in trades_sorted:
        by_asset[t['asset']].append(t)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {'ES': '#1f77b4', 'NQ': '#ff7f0e', 'J1': '#2ca02c', 'CL': '#17becf', 'GC': '#9467bd'}
    
    # Plot each asset
    for asset in ['ES', 'NQ', 'J1', 'CL', 'GC']:
        if asset not in by_asset:
            continue
        asset_trades = sorted(by_asset[asset], key=lambda x: x['exit_time'])
        cumulative_r = []
        running = 0
        dates = []
        for t in asset_trades:
            running += t['r_pnl']
            cumulative_r.append(running)
            dates.append(t['exit_time'])
        
        ax.plot(dates, cumulative_r, label=asset, color=colors.get(asset, 'gray'), linewidth=1.5, alpha=0.8)
    
    # Plot combined
    combined_r = []
    running = 0
    dates = []
    for t in trades_sorted:
        running += t['r_pnl']
        combined_r.append(running)
        dates.append(t['exit_time'])
    
    ax.plot(dates, combined_r, label='Combined', color='black', linewidth=2.5, linestyle='--')
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative R-Multiple', fontsize=12)
    ax.set_title('Equity Curves by Asset (Cumulative R-Multiple)', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")
    plt.close()

# ----------------------------
# Plot 2: Compounded Equity with Commissions
# ----------------------------
def build_equity_curve(trades, risk_per_trade=0.002, start_equity=100000.0, 
                       max_dollar_risk=500, commission_filter=None):
    """Build compounded equity curve"""
    if not trades:
        return [], {}
    
    equity = start_equity
    curve = []
    stats = {'skipped': 0, 'included': 0, 'total_commission': 0}
    
    for trade in sorted(trades, key=lambda x: x['exit_time']):
        asset = trade['asset']
        config = ASSET_CONFIG.get(asset, {'tick_size': 0.01, 'tick_value': 10.0, 'commission': 2.00})
        
        stop_size = abs(trade['entry_price'] - trade['stop_price'])
        stop_ticks = stop_size / config['tick_size']
        dollar_risk_per_contract = stop_ticks * config['tick_value']
        
        if dollar_risk_per_contract <= 0:
            continue
        
        target_risk = min(risk_per_trade * equity, max_dollar_risk)
        contracts = max(1, int(target_risk / dollar_risk_per_contract))
        actual_risk = contracts * dollar_risk_per_contract
        
        commission = contracts * config['commission'] * 2
        
        # Commission filter
        if commission_filter is not None and actual_risk > 0:
            if commission / actual_risk > commission_filter:
                stats['skipped'] += 1
                continue
        
        dollar_pnl = trade['r_pnl'] * actual_risk
        equity += dollar_pnl - commission
        stats['total_commission'] += commission
        stats['included'] += 1
        
        curve.append({
            'exit_time': trade['exit_time'],
            'equity': equity,
            'dollar_pnl': dollar_pnl,
            'commission': commission,
            'asset': trade['asset']
        })
    
    return curve, stats

def plot_equity_with_commissions(trades, output_path):
    """Plot compounded equity with all trades vs filtered"""
    # Build both curves
    curve_all, stats_all = build_equity_curve(trades, max_dollar_risk=500, commission_filter=None)
    curve_filtered, stats_filtered = build_equity_curve(trades, max_dollar_risk=500, commission_filter=0.20)
    
    if not curve_all:
        print("No data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Top: Equity curves
    dates_all = [c['exit_time'] for c in curve_all]
    equity_all = [c['equity'] for c in curve_all]
    
    dates_filtered = [c['exit_time'] for c in curve_filtered]
    equity_filtered = [c['equity'] for c in curve_filtered]
    
    ax1.plot(dates_all, equity_all, 'b-', linewidth=1.5, label='All Trades (Risk 0.2%)', alpha=0.9)
    ax1.plot(dates_filtered, equity_filtered, 'g--', linewidth=2, label='Filtered (Comm < 20% of Risk)', alpha=0.9)
    
    ax1.set_ylabel('Equity ($)', fontsize=12)
    ax1.set_title('Compounded Equity (with Commissions, $500 Max Risk) - Risk 0.2%', fontsize=14)
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Commission per trade
    comm_all = [c['commission'] for c in curve_all]
    comm_filtered = [c['commission'] for c in curve_filtered]
    
    ax2.plot(dates_all, comm_all, 'r-', linewidth=1, label='Commission per trade (All)', alpha=0.7)
    ax2.plot(dates_filtered, comm_filtered, color='orange', linestyle='--', linewidth=1, label='Commission per trade (Filtered)', alpha=0.7)
    
    ax2.set_ylabel('Commission ($)', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")
    plt.close()
    
    return curve_all, stats_all, curve_filtered, stats_filtered

# ----------------------------
# Sharpe Calculation
# ----------------------------
def calculate_sharpe(trades):
    """Calculate annualized Sharpe from daily R-multiples"""
    if not trades:
        return 0.0
    
    by_date = defaultdict(float)
    for t in trades:
        by_date[t['exit_time'].date()] += t['r_pnl']
    
    returns = list(by_date.values())
    if len(returns) < 2:
        return 0.0
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    
    if std_ret == 0:
        return 0.0
    
    return (mean_ret / std_ret) * np.sqrt(252)

# ----------------------------
# Main
# ----------------------------
def main():
    print("="*70)
    print("RDR Overnight Breakout - 5R Strategy")
    print("="*70)
    print(f"Range: 09:30-15:55 | Breakout: After 15:55 | Exit: 15:55 next day")
    print(f"Stop Ratio: {STOP_RATIO} | Divisor: {DIVISOR} (5R)")
    print("="*70)
    
    start_date = datetime.date(2008, 1, 1)
    end_date = datetime.date(2025, 12, 31)
    
    trades = run_backtest(start_date, end_date)
    
    if not trades:
        print("No trades found!")
        return
    
    # Calculate stats
    total_r = sum(t['r_pnl'] for t in trades)
    wins = len([t for t in trades if t['r_pnl'] > 0])
    win_rate = wins / len(trades)
    sharpe = calculate_sharpe(trades)
    
    print("\n" + "="*70)
    print("SUMMARY (5R Strategy)")
    print("="*70)
    print(f"Total Trades: {len(trades)}")
    print(f"Total R: {total_r:.2f}R")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Avg R/Trade: {total_r / len(trades):.3f}R")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    # Per-asset breakdown
    print("\n--- By Asset ---")
    for asset in ['ES', 'NQ', 'J1', 'CL', 'GC']:
        asset_trades = [t for t in trades if t['asset'] == asset]
        if asset_trades:
            asset_r = sum(t['r_pnl'] for t in asset_trades)
            asset_wr = len([t for t in asset_trades if t['r_pnl'] > 0]) / len(asset_trades)
            print(f"  {asset}: {len(asset_trades)} trades, {asset_r:.2f}R, {asset_wr:.1%} WR")
    
    # Generate plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Plot 1: R-curves by asset
    plot_equity_by_asset(trades, f'{OUTPUT_DIR}/equity_curves_5R_by_asset.png')
    
    # Plot 2: Compounded equity with commissions
    curve_all, stats_all, curve_filtered, stats_filtered = plot_equity_with_commissions(
        trades, f'{OUTPUT_DIR}/equity_with_commissions_5R.png'
    )
    
    # Print commission impact
    print("\n" + "="*70)
    print("COMMISSION ANALYSIS ($500 Max Risk)")
    print("="*70)
    print(f"\nAll Trades:")
    print(f"  Final Equity: ${curve_all[-1]['equity']:,.0f}")
    print(f"  Trades: {stats_all['included']}")
    print(f"  Total Commissions: ${stats_all['total_commission']:,.0f}")
    
    print(f"\n20% Commission Filter:")
    print(f"  Final Equity: ${curve_filtered[-1]['equity']:,.0f}")
    print(f"  Trades: {stats_filtered['included']}, Skipped: {stats_filtered['skipped']}")
    print(f"  Total Commissions: ${stats_filtered['total_commission']:,.0f}")
    
    # Save trades
    trades_file = f'{OUTPUT_DIR}/trades_5R_overnight.csv'
    with open(trades_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=trades[0].keys())
        writer.writeheader()
        writer.writerows(trades)
    print(f"\nTrades saved to {trades_file}")

if __name__ == "__main__":
    main()
