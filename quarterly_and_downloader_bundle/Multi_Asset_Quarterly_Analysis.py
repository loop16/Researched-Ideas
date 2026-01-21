import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Import the quarterly analysis functions
from SPX_Quarterly_Analysis import (
    load_and_clean_data, 
    analyze_quarterly_ranges, 
    calculate_quarterly_statistics,
    generate_quarterly_candles
)

def run_qpp_indicator_levels(analysis_results_path, asset_name):
    """Run the QPP Indicator Levels analysis on the quarterly analysis results."""
    print(f"\n{'='*60}")
    print(f"Running QPP Indicator Levels Analysis for {asset_name}")
    print(f"{'='*60}")
    
    # Load the CSV file
    df = pd.read_csv(analysis_results_path)
    
    # Drop any rows with missing values for critical columns
    df = df.dropna(subset=['breakout_type', 'breakout_outcome', 'high_pct_change', 'low_pct_change'])
    
    if df.empty:
        print(f"No valid data found for {asset_name}")
        return None
    
    print(f"Loaded {len(df)} valid quarterly records for {asset_name}")
    
    # Calculate quantiles for long and short breakouts using all data
    long_data = df[df['breakout_type'] == 'Long']
    short_data = df[df['breakout_type'] == 'Short']
    
    print(f"Long breakouts: {len(long_data)}")
    print(f"Short breakouts: {len(short_data)}")
    
    # Calculate all quantiles from 0.1 to 0.9 for both directions
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    if not long_data.empty:
        print(f"\nQuantiles for {asset_name} Long Breakouts:")
        for q in quantiles:
            low = long_data['low_pct_change'].quantile(q)
            high = long_data['high_pct_change'].quantile(1-q)
            print(f"{int(q*100)}% Level:")
            print(f"Low: {low:.4f}")
            print(f"High: {high:.4f}")
    
    if not short_data.empty:
        print(f"\nQuantiles for {asset_name} Short Breakouts:")
        for q in quantiles:
            low = short_data['low_pct_change'].quantile(q)
            high = short_data['high_pct_change'].quantile(1-q)
            print(f"{int(q*100)}% Level:")
            print(f"Low: {low:.4f}")
            print(f"High: {high:.4f}")
    
    # Save basic quantiles to CSV
    basic_results = []
    for direction, data in [('Long', long_data), ('Short', short_data)]:
        if data.empty:
            continue
        for q in quantiles:
            basic_results.append({
                'Asset': asset_name,
                'Direction': direction,
                'Quantile': f"{int(q*100)}%",
                'Low': data['low_pct_change'].quantile(q) if 'low_pct_change' in data.columns else None,
                'High': data['high_pct_change'].quantile(1-q) if 'high_pct_change' in data.columns else None,
                'Post_Breakout_Low': data['post_breakout_low_pct_change'].quantile(q) if 'post_breakout_low_pct_change' in data.columns else None,
                'Post_Breakout_High': data['post_breakout_high_pct_change'].quantile(1-q) if 'post_breakout_high_pct_change' in data.columns else None
            })
    
    basic_results_df = pd.DataFrame(basic_results)
    basic_output_path = f'/Users/orlandocantoni/Desktop/RBC auto model/{asset_name}_Basic_Quantiles.csv'
    basic_results_df.to_csv(basic_output_path, index=False)
    print(f"Basic quantiles saved to {basic_output_path}")
    
    # ------ Analysis 1: By Breakout Type and Outcome ------
    detailed_results = []
    
    # Calculate 90% levels for long and short breakouts first
    long_low_90 = long_data['low_pct_change'].quantile(0.1) if not long_data.empty else None
    long_high_90 = long_data['high_pct_change'].quantile(0.9) if not long_data.empty else None
    short_low_90 = short_data['low_pct_change'].quantile(0.1) if not short_data.empty else None
    short_high_90 = short_data['high_pct_change'].quantile(0.9) if not short_data.empty else None
    
    for (breakout_type, breakout_outcome), group in df.groupby(['breakout_type', 'breakout_outcome']):
        if group.empty:
            continue
            
        # Calculate percentiles
        low_20 = group['low_pct_change'].quantile(0.2)
        low_50 = group['low_pct_change'].quantile(0.5)
        low_80 = group['low_pct_change'].quantile(0.8)
        low_85 = group['low_pct_change'].quantile(0.85)

        high_15 = group['high_pct_change'].quantile(0.15)
        high_20 = group['high_pct_change'].quantile(0.2)
        high_50 = group['high_pct_change'].quantile(0.5)
        high_80 = group['high_pct_change'].quantile(0.8)

        mid = (low_80 + high_20) / 2
        count = group.shape[0]

        # Assign 90% levels based on breakout type
        if breakout_type == 'Long':
            low_90 = long_low_90
            high_90 = long_high_90
        elif breakout_type == 'Short':
            low_90 = short_low_90
            high_90 = short_high_90
        else:
            low_90 = None
            high_90 = None

        row = {
            'Asset': asset_name,
            'Breakout_Type': breakout_type,
            'Breakout_Outcome': breakout_outcome,
            'Count': count,
            'low_80%': low_20,
            'low_50%': low_50,
            'low_20%': low_80,
            'low_15%': low_85,
            'low_90%': low_90,
            'mid': mid,
            'high_15%': high_15,
            'high_20%': high_20,
            'high_50%': high_50,
            'high_80%': high_80,
            'high_90%': high_90,
        }
        detailed_results.append(row)

    # Create the detailed dataframe
    detailed_df = pd.DataFrame(detailed_results)
    if not detailed_df.empty:
        print(f"\n{asset_name} Detailed Analysis:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(detailed_df.to_string())
        
        detailed_output_path = f'/Users/orlandocantoni/Desktop/RBC auto model/{asset_name}_Quarterly_Percentiles.csv'
        detailed_df.to_csv(detailed_output_path, index=False)
        print(f"Detailed percentiles saved to {detailed_output_path}")

    # ------ Analysis 2: By Quarter, Breakout Type, and Outcome ------
    quarterly_results = []

    for (quarter, breakout_type, breakout_outcome), group in df.groupby(['quarter', 'breakout_type', 'breakout_outcome']):
        if group.empty or len(group) < 5:  # Skip groups with too few samples
            continue
            
        low_20 = group['low_pct_change'].quantile(0.2)
        low_50 = group['low_pct_change'].quantile(0.5)
        low_80 = group['low_pct_change'].quantile(0.8)
        low_85 = group['low_pct_change'].quantile(0.85)

        high_15 = group['high_pct_change'].quantile(0.15)
        high_20 = group['high_pct_change'].quantile(0.2)
        high_50 = group['high_pct_change'].quantile(0.5)
        high_80 = group['high_pct_change'].quantile(0.8)

        mid = (low_80 + high_20) / 2
        count = group.shape[0]

        # Assign 90% levels based on breakout type
        if breakout_type == 'Long':
            low_90 = long_low_90
            high_90 = long_high_90
        elif breakout_type == 'Short':
            low_90 = short_low_90
            high_90 = short_high_90
        else:
            low_90 = None
            high_90 = None

        row = {
            'Asset': asset_name,
            'Quarter': quarter,
            'Breakout_Type': breakout_type,
            'Breakout_Outcome': breakout_outcome,
            'Count': count,
            'low_80%': low_20,
            'low_50%': low_50,
            'low_20%': low_80,
            'low_15%': low_85,
            'low_90%': low_90,
            'mid': mid,
            'high_15%': high_15,
            'high_20%': high_20,
            'high_50%': high_50,
            'high_80%': high_80,
            'high_90%': high_90,
        }
        quarterly_results.append(row)

    # Create the quarterly dataframe
    quarterly_df = pd.DataFrame(quarterly_results)
    if not quarterly_df.empty:
        print(f"\n{asset_name} Quarterly Analysis:")
        print(quarterly_df.to_string())
        
        quarterly_output_path = f'/Users/orlandocantoni/Desktop/RBC auto model/{asset_name}_Quarterly_Percentiles_by_Quarter.csv'
        quarterly_df.to_csv(quarterly_output_path, index=False)
        print(f"Quarterly analysis saved to {quarterly_output_path}")

    # ------ Summary statistics ------
    print(f"\n{asset_name} Sample Counts by Quarter:")
    quarter_counts = df.groupby(['quarter', 'breakout_type', 'breakout_outcome']).size()
    print(quarter_counts)

    # Calculate success rates by quarter
    success_rates = df.groupby(['quarter', 'breakout_type'])['breakout_outcome'].apply(
        lambda x: (x == 'True').mean() * 100
    ).reset_index(name='Success_Rate_Pct')
    success_rates['Asset'] = asset_name

    print(f"\n{asset_name} Success Rates by Quarter:")
    print(success_rates.to_string())
    
    success_output_path = f'/Users/orlandocantoni/Desktop/RBC auto model/{asset_name}_Quarterly_Success_Rates.csv'
    success_rates.to_csv(success_output_path, index=False)
    print(f"Success rates saved to {success_output_path}")
    
    return {
        'basic_quantiles': basic_results_df,
        'detailed_percentiles': detailed_df,
        'quarterly_analysis': quarterly_df,
        'success_rates': success_rates
    }

def generate_levels_json(detailed_percentiles_df, asset_name):
    """Generate levels.json format from detailed percentiles data."""
    
    # Filter data for this asset
    asset_data = detailed_percentiles_df[detailed_percentiles_df['Asset'] == asset_name]
    
    if asset_data.empty:
        return None
    
    lines = []
    
    # Define colors for each breakout type/outcome combination
    colors = {
        ('Long', 'True'): '#2563eb',    # Blue solid
        ('Long', 'False'): '#16a34a',   # Green dashed
        ('Short', 'True'): '#dc2626',   # Dark red solid
        ('Short', 'False'): '#ef4444'   # Red dashed
    }
    
    # Process each breakout type/outcome combination
    for _, row in asset_data.iterrows():
        breakout_type = row['Breakout_Type']
        breakout_outcome = row['Breakout_Outcome']
        
        # Determine style and color
        style = 'solid' if breakout_outcome == 'True' else 'dashed'
        color = colors.get((breakout_type, breakout_outcome), '#666666')
        
        # Create entries for each level (1-9, representing 10%-90%)
        level_columns = ['low_80%', 'low_50%', 'low_20%', 'low_15%', 'mid', 'high_15%', 'high_20%', 'high_50%', 'high_80%']
        
        for i, col in enumerate(level_columns, 1):
            if pd.notna(row[col]):
                lines.append({
                    'name': f"{breakout_type}_{breakout_outcome}_{i}",
                    'value': round(float(row[col]), 6),
                    'style': style,
                    'color': color
                })
    
    # Create the JSON structure
    levels_data = {
        'symbol': asset_name,
        'asof': datetime.now().strftime('%Y-%m-%d'),
        'daily': {
            'lines': lines
        },
        'meta': {
            'model_rev': 'v1.0'
        }
    }
    
    return levels_data

def generate_basic_levels_json(basic_quantiles_df, asset_name):
    """Generate basic levels.json format from basic quantiles data."""
    
    # Filter data for this asset
    asset_data = basic_quantiles_df[basic_quantiles_df['Asset'] == asset_name]
    
    if asset_data.empty:
        return None
    
    lines = []
    
    # Define colors for long and short
    colors = {
        'Long': '#2563eb',    # Blue
        'Short': '#dc2626'    # Red
    }
    
    # Process each direction
    for direction in ['Long', 'Short']:
        direction_data = asset_data[asset_data['Direction'] == direction]
        
        for _, row in direction_data.iterrows():
            quantile = row['Quantile']
            level_num = quantile.replace('%', '')  # e.g., '10%' -> '10'
            
            # Add low and high values if they exist
            if pd.notna(row['Low']):
                lines.append({
                    'name': f"{direction}_Low_{level_num}",
                    'value': round(float(row['Low']), 6),
                    'style': 'solid',
                    'color': colors[direction]
                })
            
            if pd.notna(row['High']):
                lines.append({
                    'name': f"{direction}_High_{level_num}",
                    'value': round(float(row['High']), 6),
                    'style': 'solid',
                    'color': colors[direction]
                })
    
    # Create the JSON structure
    levels_data = {
        'symbol': asset_name,
        'asof': datetime.now().strftime('%Y-%m-%d'),
        'daily': {
            'lines': lines
        },
        'meta': {
            'model_rev': 'v1.0_basic'
        }
    }
    
    return levels_data

def run_quarterly_analysis_for_asset(file_path, asset_name):
    """Run the quarterly analysis for a single asset."""
    print(f"\n{'='*60}")
    print(f"Running Quarterly Analysis for {asset_name}")
    print(f"{'='*60}")
    
    try:
        # Load and clean data
        print("Loading and cleaning data...")
        df = load_and_clean_data(file_path)
        
        # Debug: Print loaded data info
        print(f"Loaded data range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Total data points: {len(df)}")
        
        # Calculate quarterly statistics
        print("\nCalculating quarterly statistics...")
        days_stats, tuesdays_stats, stats_df, days_freq, tuesdays_freq = calculate_quarterly_statistics(df)
        
        # Print summary statistics
        print("\nTrading Days per Quarter (first Friday to first Friday):")
        print(f"Min: {days_stats['min']}")
        print(f"Max: {days_stats['max']}")
        print(f"Average: {days_stats['average']:.2f}")
        print(f"Median: {days_stats['median']:.2f}")
        
        # Analyze quarterly ranges
        print("Analyzing quarterly ranges...")
        results = analyze_quarterly_ranges(df)

        # Save results to CSV
        results_path = f"/Users/orlandocantoni/Desktop/RBC auto model/{asset_name}_Quarterly_Analysis_Results.csv"
        results.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")
        
        # Generate and save normalized quarterly data
        print("\nGenerating normalized quarterly data...")
        normalized_df = generate_quarterly_candles(df, results)
        normalized_path = f"/Users/orlandocantoni/Desktop/RBC auto model/{asset_name}_Quarterly_Normalized.csv"
        normalized_df.to_csv(normalized_path, index=False)
        print(f"Normalized quarterly data saved to {normalized_path}")
        
        # Print summary statistics
        print(f"\n{asset_name} Quarterly Range Analysis Summary:")
        print(f"Total quarters analyzed: {len(results)}")
        
        if len(results) > 0:
            print(f"Long breakouts: {len(results[results['breakout_type'] == 'Long'])}")
            print(f"Short breakouts: {len(results[results['breakout_type'] == 'Short'])}")
            print(f"False breakouts: {len(results[results['breakout_outcome'] == 'False'])}")
            print(f"True breakouts: {len(results[results['breakout_outcome'] == 'True'])}")
            
            # Print range type distribution
            print(f"\n{asset_name} Range Type Distribution:")
            print(results['range_type'].value_counts())
            
        return results_path, normalized_path
        
    except Exception as e:
        print(f"Error processing {asset_name}: {str(e)}")
        return None, None

def process_multiple_assets(asset_configs):
    """
    Process multiple assets through quarterly analysis and QPP indicator levels.
    
    asset_configs: List of dictionaries with 'file_path' and 'asset_name' keys
    """
    all_basic_quantiles = []
    all_detailed_percentiles = []
    all_quarterly_analysis = []
    all_success_rates = []
    
    # Dictionary to store individual asset results for JSON generation
    asset_results = {}
    
    for config in asset_configs:
        file_path = config['file_path']
        asset_name = config['asset_name']
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
            continue
            
        # Step 1: Run quarterly analysis
        results_path, normalized_path = run_quarterly_analysis_for_asset(file_path, asset_name)
        
        if results_path is None:
            print(f"Skipping QPP analysis for {asset_name} due to quarterly analysis failure")
            continue
            
        # Step 2: Run QPP indicator levels analysis
        qpp_results = run_qpp_indicator_levels(results_path, asset_name)
        
        if qpp_results is not None:
            # Store individual asset results
            asset_results[asset_name] = qpp_results
            
            # Collect results for combined output
            all_basic_quantiles.append(qpp_results['basic_quantiles'])
            all_detailed_percentiles.append(qpp_results['detailed_percentiles'])
            all_quarterly_analysis.append(qpp_results['quarterly_analysis'])
            all_success_rates.append(qpp_results['success_rates'])
    
    # Combine all results into master files
    print(f"\n{'='*60}")
    print("Creating Combined Analysis Files")
    print(f"{'='*60}")
    
    if all_basic_quantiles:
        combined_basic = pd.concat(all_basic_quantiles, ignore_index=True)
        combined_basic.to_csv('/Users/orlandocantoni/Desktop/RBC auto model/All_Assets_Basic_Quantiles.csv', index=False)
        print("Combined basic quantiles saved to All_Assets_Basic_Quantiles.csv")
    
    if all_detailed_percentiles:
        combined_detailed = pd.concat(all_detailed_percentiles, ignore_index=True)
        combined_detailed.to_csv('/Users/orlandocantoni/Desktop/RBC auto model/All_Assets_Detailed_Percentiles.csv', index=False)
        print("Combined detailed percentiles saved to All_Assets_Detailed_Percentiles.csv")
    
    if all_quarterly_analysis:
        combined_quarterly = pd.concat(all_quarterly_analysis, ignore_index=True)
        combined_quarterly.to_csv('/Users/orlandocantoni/Desktop/RBC auto model/All_Assets_Quarterly_Analysis.csv', index=False)
        print("Combined quarterly analysis saved to All_Assets_Quarterly_Analysis.csv")
    
    if all_success_rates:
        combined_success = pd.concat(all_success_rates, ignore_index=True)
        combined_success.to_csv('/Users/orlandocantoni/Desktop/RBC auto model/All_Assets_Success_Rates.csv', index=False)
        print("Combined success rates saved to All_Assets_Success_Rates.csv")
    
    # Generate JSON files
    print(f"\n{'='*60}")
    print("Generating JSON Level Files")
    print(f"{'='*60}")
    
    # Generate detailed levels JSON
    if all_detailed_percentiles:
        detailed_levels_json = {}
        for asset_name in asset_results.keys():
            levels_data = generate_levels_json(combined_detailed, asset_name)
            if levels_data:
                detailed_levels_json[asset_name] = levels_data
        
        # Save detailed levels JSON
        detailed_json_path = '/Users/orlandocantoni/Desktop/RBC auto model/detailed_levels.json'
        with open(detailed_json_path, 'w') as f:
            json.dump(detailed_levels_json, f, indent=2)
        print(f"Detailed levels JSON saved to {detailed_json_path}")
    
    # Generate basic levels JSON
    if all_basic_quantiles:
        basic_levels_json = {}
        for asset_name in asset_results.keys():
            levels_data = generate_basic_levels_json(combined_basic, asset_name)
            if levels_data:
                basic_levels_json[asset_name] = levels_data
        
        # Save basic levels JSON
        basic_json_path = '/Users/orlandocantoni/Desktop/RBC auto model/basic_levels.json'
        with open(basic_json_path, 'w') as f:
            json.dump(basic_levels_json, f, indent=2)
        print(f"Basic levels JSON saved to {basic_json_path}")
    
    print(f"\nJSON files generated for {len(asset_results)} assets")

def main():
    """Main function to define assets and run analysis."""
    
    # Define your asset configurations here
    asset_configs = [
        {
            'file_path': '/Users/orlandocantoni/Desktop/RBC auto model/SPX_1D.csv',
            'asset_name': 'SPX'
        },
        {
            'file_path': '/Users/orlandocantoni/Desktop/RBC auto model/NDX_1D.csv',
            'asset_name': 'NDX'
        },
        {
            'file_path': '/Users/orlandocantoni/Desktop/RBC auto model/CL_1D.csv',
            'asset_name': 'CL'
        },
        {
            'file_path': '/Users/orlandocantoni/Desktop/RBC auto model/GC_1D.csv',
            'asset_name': 'GC'
        },
        {
            'file_path': '/Users/orlandocantoni/Desktop/RBC auto model/BTC_1D.csv',
            'asset_name': 'BTC'
        },
        # Add more assets as needed:
        # {
        #     'file_path': '/Users/orlandocantoni/Desktop/RBC auto model/BTC_1D.csv',
        #     'asset_name': 'BTC'
        # },
        # {
        #     'file_path': '/Users/orlandocantoni/Desktop/RBC auto model/Another_Asset_1D.csv',
        #     'asset_name': 'Another_Asset'
        # }
    ]
    
    print("Starting Multi-Asset Quarterly Analysis")
    print(f"Processing {len(asset_configs)} assets...")
    
    process_multiple_assets(asset_configs)
    
    print("\nMulti-Asset Analysis Complete!")

if __name__ == "__main__":
    main()
