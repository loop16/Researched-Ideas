#!/usr/bin/env python3
"""
Daily R Distribution Analysis
Analyzes daily R results, max loss, and distribution for both strategies
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

BASE_DIR = '/Users/orlandocantoni/Desktop/RBC auto model'

STRATEGIES = {
    'RDR-Box Breakout': {
        'folder': f'{BASE_DIR}/backtest_results_rdr_box_breakout',
        'trades_file': 'trades_rdr_box_breakout_5R.csv'
    },
    'RDR-Session': {
        'folder': f'{BASE_DIR}/backtest_results_rdr_session',
        'trades_file': 'trades_two_bullet_2.5_sd.csv'
    }
}

def load_trades(filepath):
    """Load trades from CSV"""
    trades = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle different column names
            if 'r_pnl' in row:
                r_pnl = float(row['r_pnl'])
            elif 'r_multiple' in row:
                r_pnl = float(row['r_multiple'])
            else:
                continue
            
            # Parse date
            if 'exit_time' in row:
                date_str = row['exit_time'].split()[0]
            elif 'date' in row:
                date_str = row['date']
            else:
                continue
            
            trades.append({
                'date': date_str,
                'r_pnl': r_pnl
            })
    return trades

def analyze_daily_r(trades):
    """Analyze daily R distribution"""
    # Group by date
    daily_r = defaultdict(float)
    for t in trades:
        daily_r[t['date']] += t['r_pnl']
    
    daily_values = list(daily_r.values())
    dates = list(daily_r.keys())
    
    if not daily_values:
        return None
    
    # Statistics
    stats = {
        'total_days': len(daily_values),
        'total_r': sum(daily_values),
        'mean_daily_r': np.mean(daily_values),
        'median_daily_r': np.median(daily_values),
        'std_daily_r': np.std(daily_values),
        'max_daily_r': max(daily_values),
        'min_daily_r': min(daily_values),
        'max_loss_date': dates[daily_values.index(min(daily_values))],
        'max_win_date': dates[daily_values.index(max(daily_values))],
        'positive_days': len([d for d in daily_values if d > 0]),
        'negative_days': len([d for d in daily_values if d < 0]),
        'zero_days': len([d for d in daily_values if d == 0]),
        'daily_values': daily_values,
        'dates': dates
    }
    
    stats['win_rate_days'] = stats['positive_days'] / stats['total_days'] if stats['total_days'] > 0 else 0
    
    # Percentiles
    stats['p5'] = np.percentile(daily_values, 5)
    stats['p25'] = np.percentile(daily_values, 25)
    stats['p75'] = np.percentile(daily_values, 75)
    stats['p95'] = np.percentile(daily_values, 95)
    
    return stats

def plot_daily_distribution(stats, strategy_name, output_path):
    """Plot daily R distribution histogram"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    daily_values = stats['daily_values']
    
    # 1. Histogram
    ax1 = axes[0, 0]
    ax1.hist(daily_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break Even')
    ax1.axvline(x=stats['mean_daily_r'], color='green', linestyle='-', linewidth=2, label=f"Mean: {stats['mean_daily_r']:.2f}R")
    ax1.axvline(x=stats['min_daily_r'], color='darkred', linestyle=':', linewidth=2, label=f"Max Loss: {stats['min_daily_r']:.2f}R")
    ax1.set_xlabel('Daily R-Multiple')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{strategy_name} - Daily R Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative Daily R
    ax2 = axes[0, 1]
    cumulative = np.cumsum(daily_values)
    ax2.plot(cumulative, color='steelblue', linewidth=1)
    ax2.fill_between(range(len(cumulative)), cumulative, alpha=0.3)
    ax2.set_xlabel('Trading Day')
    ax2.set_ylabel('Cumulative R')
    ax2.set_title('Cumulative Daily R Over Time')
    ax2.grid(True, alpha=0.3)
    
    # 3. Box Plot
    ax3 = axes[1, 0]
    bp = ax3.boxplot(daily_values, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax3.set_ylabel('Daily R-Multiple')
    ax3.set_title('Daily R Box Plot')
    ax3.grid(True, alpha=0.3)
    
    # 4. Stats Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    table_data = [
        ['Total Trading Days', f"{stats['total_days']:,}"],
        ['Total R', f"{stats['total_r']:.2f}R"],
        ['Mean Daily R', f"{stats['mean_daily_r']:.3f}R"],
        ['Median Daily R', f"{stats['median_daily_r']:.3f}R"],
        ['Std Dev', f"{stats['std_daily_r']:.3f}R"],
        ['', ''],
        ['Max Daily Win', f"{stats['max_daily_r']:.2f}R"],
        ['Max Daily Loss', f"{stats['min_daily_r']:.2f}R"],
        ['', ''],
        ['5th Percentile', f"{stats['p5']:.2f}R"],
        ['25th Percentile', f"{stats['p25']:.2f}R"],
        ['75th Percentile', f"{stats['p75']:.2f}R"],
        ['95th Percentile', f"{stats['p95']:.2f}R"],
        ['', ''],
        ['Positive Days', f"{stats['positive_days']} ({stats['win_rate_days']:.1%})"],
        ['Negative Days', f"{stats['negative_days']}"],
        ['Zero Days', f"{stats['zero_days']}"],
    ]
    
    table = ax4.table(cellText=table_data, colLabels=['Metric', 'Value'],
                      loc='center', cellLoc='left',
                      colWidths=[0.5, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax4.set_title('Daily R Statistics', fontsize=12, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")
    plt.close()
    
    return stats

def main():
    print("="*70)
    print("Daily R Distribution Analysis")
    print("="*70)
    
    for strategy_name, config in STRATEGIES.items():
        print(f"\n--- {strategy_name} ---")
        
        trades_path = os.path.join(config['folder'], config['trades_file'])
        if not os.path.exists(trades_path):
            print(f"  Trades file not found: {trades_path}")
            continue
        
        # Load and analyze
        trades = load_trades(trades_path)
        print(f"  Loaded {len(trades)} trades")
        
        stats = analyze_daily_r(trades)
        if not stats:
            print("  No data to analyze")
            continue
        
        # Plot
        output_path = os.path.join(config['folder'], 'daily_r_distribution.png')
        plot_daily_distribution(stats, strategy_name, output_path)
        
        # Print summary
        print(f"\n  Daily R Statistics:")
        print(f"    Total Days: {stats['total_days']:,}")
        print(f"    Mean Daily R: {stats['mean_daily_r']:.3f}R")
        print(f"    Max Daily Win: {stats['max_daily_r']:.2f}R (on {stats['max_win_date']})")
        print(f"    Max Daily Loss: {stats['min_daily_r']:.2f}R (on {stats['max_loss_date']})")
        print(f"    Positive Days: {stats['positive_days']} ({stats['win_rate_days']:.1%})")

if __name__ == "__main__":
    main()
