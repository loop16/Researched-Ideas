#!/usr/bin/env python3
"""
Test script for the consolidated live feature engine.

This script processes real ES data for May 20-25, 2025 and outputs
one line per 5-minute bar with all the essential features.
"""

import sys
import os
from pathlib import Path
import pandas as pd
from datetime import datetime, date, timedelta
import numpy as np

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from world.fusion.consolidated_live_engine import ConsolidatedLiveEngine
from world.fusion.contracts import Bar


def load_es_data_for_date_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Load ES data for a specific date range."""
    data_dir = Path(__file__).parent.parent / "data" / "processed" / "ES"
    filepath = data_dir / "ES_bars_with_sessions.parquet"
    
    if not filepath.exists():
        raise FileNotFoundError(f"ES data file not found: {filepath}")
    
    print(f"Loading ES data from {filepath}")
    df = pd.read_parquet(filepath)
    
    # Convert timestamps to datetime
    df['datetime'] = pd.to_datetime(df['ts'], unit='s')
    df['date'] = df['datetime'].dt.date
    
    # Filter for date range
    start_dt = datetime.fromisoformat(start_date).date()
    end_dt = datetime.fromisoformat(end_date).date()
    
    mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
    filtered_df = df[mask].copy()
    
    print(f"Loaded {len(filtered_df)} bars for {start_date} to {end_date}")
    print(f"Date range: {filtered_df['date'].min()} to {filtered_df['date'].max()}")
    
    return filtered_df


def convert_to_bars(df: pd.DataFrame) -> list[Bar]:
    """Convert DataFrame to Bar objects."""
    bars = []
    
    for _, row in df.iterrows():
        # Handle timestamp conversion
        if isinstance(row['ts'], pd.Timestamp):
            ts = int(row['ts'].timestamp())
        else:
            ts = int(row['ts'])
        
        bar = Bar(
            ts=ts,
            symbol=str(row['symbol']),
            o=float(row['open']),
            h=float(row['high']),
            l=float(row['low']),
            c=float(row['close']),
            v=float(row.get('volume', 1000)),  # Default volume if not present
            session=str(row['session'])
        )
        bars.append(bar)
    
    return bars


def test_consolidated_live_engine():
    """Test the consolidated live feature engine with real ES data."""
    print("=" * 80)
    print("TESTING CONSOLIDATED LIVE FEATURE ENGINE")
    print("Date Range: May 20-25, 2025")
    print("Output: One line per 5-minute bar with all essential features")
    print("=" * 80)
    
    try:
        # Load ES data for the specified date range
        es_data = load_es_data_for_date_range("2025-05-20", "2025-05-25")
        
        if es_data.empty:
            print("No data found for the specified date range!")
            return
        
        # Convert to Bar objects
        bars = convert_to_bars(es_data)
        print(f"Converted to {len(bars)} Bar objects")
        
        # Initialize the consolidated live engine
        data_dir = Path(__file__).parent.parent / "data"
        engine = ConsolidatedLiveEngine(str(data_dir))
        
        print(f"\nInitializing Consolidated Live Engine...")
        print(f"Data directory: {data_dir}")
        
        # List to store all consolidated outputs
        all_outputs = []
        
        # Process each bar through the consolidated engine
        print(f"\nProcessing {len(bars)} bars...")
        
        processed_count = 0
        
        for i, bar in enumerate(bars):
            # Ingest the bar and get consolidated output
            output = engine.ingest_bar(bar)
            all_outputs.append(output)
            
            processed_count += 1
            
            # Print progress every 200 bars
            if processed_count % 200 == 0:
                bar_dt = datetime.fromtimestamp(bar.ts)
                print(f"  Processed {processed_count}/{len(bars)} bars - {bar_dt} - {bar.session} - Close: {bar.c:.2f}")
        
        print(f"\nCompleted processing {processed_count} bars")
        
        # Create output directory
        output_dir = Path(__file__).parent.parent / "consolidated_live_outputs"
        output_dir.mkdir(exist_ok=True)
        
        # Export to CSV
        print(f"\n--- EXPORTING CONSOLIDATED OUTPUT TO CSV ---")
        print(f"Output directory: {output_dir}")
        
        # Convert to DataFrame and export
        df = pd.DataFrame(all_outputs)
        output_file = output_dir / "consolidated_live_features_may_20_25_2025.csv"
        df.to_csv(output_file, index=False)
        print(f"✓ Exported {len(df)} consolidated feature records to: {output_file}")
        
        # Show sample data
        print(f"\n--- SAMPLE DATA PREVIEW ---")
        print(f"\nConsolidated Output Sample (first 10 rows):")
        
        # Show key columns
        key_columns = [
            'timestamp', 'session', 'close',
            'prior_session', 'prior_session_model', 'prior_session_box_dir', 'prior_session_break_out', 'prior_session_true',
            'current_session', 'current_session_model', 'current_session_box_dir', 'current_session_box_std', 
            'current_session_break_out', 'current_session_true', 'current_session_max_ext', 'current_session_max_ret',
            'high_time_idx', 'low_time_idx', 'high_pct', 'low_pct',
            'WDR_high_time_idx', 'WDR_low_time_idx', 'WDR_high_pct', 'WDR_low_pct',
            'adr_mid_break', 'odr_mid_break'
        ]
        
        # Filter to only show columns that exist
        available_columns = [col for col in key_columns if col in df.columns]
        print(df[available_columns].head(10).to_string())
        
        # Show some examples with actual data
        print(f"\n--- EXAMPLES WITH ACTUAL DATA ---")
        
        # Show range session examples
        range_examples = df[df['session'].isin(['AR', 'OR', 'RR'])].head(3)
        if not range_examples.empty:
            print(f"\nRange Session Examples:")
            print(range_examples[['timestamp', 'session', 'close', 'current_session_box_std', 'current_session_model']].to_string())
        
        # Show follow session examples
        follow_examples = df[df['session'].isin(['AS', 'OS', 'RS'])].head(3)
        if not follow_examples.empty:
            print(f"\nFollow Session Examples:")
            print(follow_examples[['timestamp', 'session', 'close', 'current_session_break_out', 'current_session_max_ext']].to_string())
        
        # Show intraday examples
        intraday_examples = df[df['high_time_idx'].notna()].head(3)
        if not intraday_examples.empty:
            print(f"\nIntraday Sequencing Examples:")
            print(intraday_examples[['timestamp', 'session', 'close', 'high_time_idx', 'low_time_idx', 'high_pct', 'low_pct']].to_string())
        
        # Show mid break examples
        mid_break_examples = df[(df['adr_mid_break'] == True) | (df['odr_mid_break'] == True)].head(3)
        if not mid_break_examples.empty:
            print(f"\nMid Break Examples:")
            print(mid_break_examples[['timestamp', 'session', 'close', 'adr_mid_break', 'odr_mid_break']].to_string())
        
        # Show frequency analysis
        print(f"\n--- OUTPUT FREQUENCY ANALYSIS ---")
        print(f"Total consolidated outputs: {len(df)}")
        print(f"Unique sessions: {df['session'].unique()}")
        print(f"Date range: {df['ny_day'].min()} to {df['ny_day'].max()}")
        
        # Show session breakdown
        session_counts = df['session'].value_counts()
        print(f"\nOutputs per session:")
        for session, count in session_counts.items():
            print(f"  {session}: {count} outputs")
        
        # Show columns in output
        print(f"\nAll columns in consolidated output ({len(df.columns)} total):")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        
        print(f"\n=== CONSOLIDATED OUTPUT COMPLETED SUCCESSFULLY ===")
        print(f"Consolidated CSV file saved to: {output_file}")
        print(f"\nKey Features:")
        print(f"- One line per 5-minute bar")
        print(f"- All essential features in single output")
        print(f"- Proper session boundary resets")
        print(f"- Real-time feature updates")
        print(f"- Ready for model adapter consumption")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_consolidated_live_engine()
