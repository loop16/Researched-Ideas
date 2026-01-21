import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def load_and_clean_data(file_path):
    """Load and clean the SPX data file."""
    # Read the CSV file directly with pandas
    df = pd.read_csv(file_path)
    
    # Rename columns if needed to match expected format
    if 'time' in df.columns:
        df = df.rename(columns={
            'time': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close'
        })
    
    # Convert types
    df['Date'] = pd.to_datetime(df['Date'])
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by date and filter for 1962 onwards
    df = df.sort_values('Date')
    df = df[df['Date'] >= '1962-01-01']
    
    return df

def get_range_type(current_low, current_high, prior_low, prior_high, prior_mid):
    """Determine the range type based on the provided conditions."""
    # Keep UXP and DXP categories as they are
    if current_low >= prior_high and current_high >= prior_high:
        return "Above"
    elif current_low <= prior_low and current_high <= prior_low:
        return "Below"
    else:
        # Group all other range types into RX
        return "Touching"

def get_first_friday_and_previous_day(df, year, quarter):
    """Find the first Friday of the quarter and the trading day before it."""
    # Determine the first day of the quarter
    if quarter == 1:
        first_day = f"{year}-01-01"
    elif quarter == 2:
        first_day = f"{year}-04-01"
    elif quarter == 3:
        first_day = f"{year}-07-01"
    else:
        first_day = f"{year}-10-01"
    
    # Get days in the quarter
    quarter_days = df[(df['Date'] >= first_day) & 
                     (df['Date'] < pd.to_datetime(first_day) + pd.DateOffset(months=3))]
    
    # Find the first Friday (weekday=4 in pandas)
    first_fridays = quarter_days[quarter_days['Date'].dt.weekday == 4]
    if first_fridays.empty:
        return None, None
    
    first_friday = first_fridays.iloc[0]
    
    # Find the previous trading day (not necessarily Thursday)
    previous_days = df[df['Date'] < first_friday['Date']]
    if previous_days.empty:
        return first_friday, None
    
    previous_day = previous_days.iloc[-1]  # Get the most recent day before Friday
    
    return first_friday, previous_day

def determine_bucket(value, thresholds):
    """Determine which bucket a value falls into based on thresholds."""
    # Convert value to float if it's not None
    if value is None:
        return "N/A"
    
    value = float(value)
    
    # Debug print
    print(f"\nDetermining bucket for value: {value}")
    print("Thresholds:")
    print(f"low_80%: {thresholds[0]}")
    print(f"low_50%: {thresholds[1]}")
    print(f"low_20%: {thresholds[2]}")
    print(f"mid: {thresholds[3]}")
    print(f"high_20%: {thresholds[4]}")
    print(f"high_50%: {thresholds[5]}")
    print(f"high_80%: {thresholds[6]}")
    
    # Check if value is beyond extreme thresholds
    if value < thresholds[0]:  # Below low_80%
        return "Below Low 20"
    if value > thresholds[6]:  # Above high_80%
        return "Above High 20"
    
    # Check which two thresholds the value falls between
    if value < thresholds[1]:  # Between low_80% and low_50%
        return "Low 20-Low 50"
    if value < thresholds[2]:  # Between low_50% and low_20%
        return "Low 50-Low 80"
    if value < thresholds[3]:  # Between low_20% and mid
        return "Low 80-Mid"
    if value < thresholds[4]:  # Between mid and high_20%
        return "Mid-High 80"
    if value < thresholds[5]:  # Between high_20% and high_50%
        return "High 80-High 50"
    if value < thresholds[6]:  # Between high_50% and high_80%
        return "High 50-High 20"
    
    return "N/A"  # Should never reach here due to initial checks

def analyze_quarterly_ranges(df):
    """Analyze quarterly ranges from 1962 onwards."""
    # Load percentiles data
    percentiles_df = pd.read_csv("/Users/orlandocantoni/Desktop/RBC auto model/SPX_Quarterly_Percentiles.csv")
    
    # Create a dictionary of thresholds for quick lookup
    thresholds_dict = {}
    for _, row in percentiles_df.iterrows():
        # Use exact strings from CSV
        key = (str(row['Breakout_Type']).strip(), str(row['Breakout_Outcome']).strip())
        
        # Create thresholds list in correct order
        thresholds = [
            float(row['low_80%']),    # Below Low 80
            float(row['low_50%']),    # Below Low 50
            float(row['low_20%']),    # Below Low 20
            float(row['mid']),        # Below Mid
            float(row['high_20%']),   # Above Mid
            float(row['high_50%']),   # Above High 20
            float(row['high_80%']),   # Above High 50
            float(row['high_80%'])    # Above High 80 (same threshold as Above High 50)
        ]
        
        thresholds_dict[key] = thresholds
        
        # Debug print for this specific key
        print(f"\nThresholds for {key}:")
        print("Thresholds:", thresholds)
        print("Bucket names:")
        print("Below Low 80:", thresholds[0])
        print("Below Low 50:", thresholds[1])
        print("Below Low 20:", thresholds[2])
        print("Below Mid:", thresholds[3])
        print("Above Mid:", thresholds[4])
        print("Above High 20:", thresholds[5])
        print("Above High 50:", thresholds[6])
        print("Above High 80:", thresholds[7])
    
    # Debug print the dictionary keys and thresholds
    print("\nAvailable threshold keys and their values:")
    for key, thresholds in thresholds_dict.items():
        print(f"\nKey: {key}")
        print("Thresholds:", thresholds)
        print("Bucket names and thresholds:")
        print("Below Low 80:", thresholds[0])
        print("Below Low 50:", thresholds[1])
        print("Below Low 20:", thresholds[2])
        print("Below Mid:", thresholds[3])
        print("Above Mid:", thresholds[4])
        print("Above High 20:", thresholds[5])
        print("Above High 50:", thresholds[6])
        print("Above High 80:", thresholds[7] if len(thresholds) > 7 else "N/A")
    
    results = []
    quarters = []
    prior_range = None
    
    # Initialize tracking variables for prior quarter's information
    prior_breakout_type = None
    prior_breakout_outcome = None
    prior_range_low = None
    prior_range_high = None
    prior_quarter_high = None
    prior_quarter_low = None
    
    # Initialize triple false tracking counters
    total_false_breakouts = 0
    triple_false_count = 0
    
    # Debug: Print date range of data
    print(f"Data range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Total trading days: {len(df)}")
    
    # Iterate through years and quarters
    for year in range(1962, 2025):  # Adjust end year based on your data
        for quarter in range(1, 5):
            # Use the new function name
            friday, previous_day = get_first_friday_and_previous_day(df, year, quarter)
            
            if friday is None or previous_day is None:
                continue
                
            # Debug output to see what data is being found
            print(f"Analyzing {year} Q{quarter}: Found Friday {friday['Date'].strftime('%Y-%m-%d')} and previous day {previous_day['Date'].strftime('%Y-%m-%d')}")
            
            # Define quarterly range
            range_high = max(friday['High'], previous_day['High'])
            range_low = min(friday['Low'], previous_day['Low'])
            range_mid = (range_high + range_low) / 2
            
            # Get data after the range
            after_range = df[df['Date'] > friday['Date']]
            after_range_quarter = after_range[
                after_range['Date'] < pd.to_datetime(friday['Date']) + pd.DateOffset(months=3)
            ]
            
            if after_range_quarter.empty:
                continue

            # Find the first Tuesday on or after the quarter start
            first_tuesday = friday['Date']
            while first_tuesday.weekday() != 1:  # 1 = Tuesday
                first_tuesday += pd.Timedelta(days=1)
            
            # Initialize weekly tracking dictionaries
            weekly_highs = {}  # Dictionary to store high for each week
            weekly_lows = {}   # Dictionary to store low for each week
            weekly_highs_pct = {}  # Dictionary to store high percentage changes
            weekly_lows_pct = {}   # Dictionary to store low percentage changes
            
            # Calculate weekly highs and lows using the same Tuesday week logic
            for _, row in after_range_quarter.iterrows():
                days_since_first_tuesday = (row['Date'] - first_tuesday).days
                week_number = (days_since_first_tuesday // 7) + 1
                
                # Initialize week in dictionaries if not exists
                if week_number not in weekly_highs:
                    weekly_highs[week_number] = float('-inf')
                    weekly_lows[week_number] = float('inf')
                
                # Update weekly high/low
                weekly_highs[week_number] = max(weekly_highs[week_number], row['High'])
                weekly_lows[week_number] = min(weekly_lows[week_number], row['Low'])
            
            # Convert weekly highs/lows to percentage changes from range mid
            weekly_highs_pct = {week: ((high - range_mid) / range_mid) * 100 
                              for week, high in weekly_highs.items()}
            weekly_lows_pct = {week: ((low - range_mid) / range_mid) * 100 
                             for week, low in weekly_lows.items()}
            
            # Find quarter high and low
            quarter_high = after_range_quarter['High'].max()
            quarter_low = after_range_quarter['Low'].min()
            
            # Find when high and low occurred
            high_date = after_range_quarter[after_range_quarter['High'] == quarter_high].iloc[0]['Date']
            low_date = after_range_quarter[after_range_quarter['Low'] == quarter_low].iloc[0]['Date']
            
            # Calculate which Tuesday week the high and low occurred in
            days_since_first_tuesday_high = (high_date - first_tuesday).days
            days_since_first_tuesday_low = (low_date - first_tuesday).days
            quarter_high_week = (days_since_first_tuesday_high // 7) + 1
            quarter_low_week = (days_since_first_tuesday_low // 7) + 1
            
            # Calculate normalized positions (1, 2, or 3)
            total_days = len(after_range_quarter)
            high_position = after_range_quarter[after_range_quarter['Date'] == high_date].index[0] - after_range_quarter.index[0]
            low_position = after_range_quarter[after_range_quarter['Date'] == low_date].index[0] - after_range_quarter.index[0]
            
            # Convert positions to buckets (1, 2, or 3)
            high_bucket = min(3, max(1, int(np.ceil((high_position + 1) / total_days * 3))))
            low_bucket = min(3, max(1, int(np.ceil((low_position + 1) / total_days * 3))))
            
            # Initialize prior level hold checks
            prior_low_held = "N/A"  # Default if no prior range
            prior_high_held = "N/A"  # Default if no prior range
            prior_quarter_high_held = "N/A"  # Default if no prior quarter
            prior_quarter_low_held = "N/A"  # Default if no prior quarter
            
            # Initialize quarter close position
            quarter_close_position = "N/A"
            
            # Check if prior range levels held
            if prior_range_low is not None and prior_range_high is not None:
                # Check if current range low is above prior range low
                if range_low > prior_range_low:
                    # Check if any close was below prior range low during the quarter
                    if any(after_range_quarter['Close'] < prior_range_low):
                        prior_low_held = "No"
                    else:
                        prior_low_held = "Yes"
                
                # Check if current range high is below prior range high
                if range_high < prior_range_high:
                    # Check if any close was above prior range high during the quarter
                    if any(after_range_quarter['Close'] > prior_range_high):
                        prior_high_held = "No"
                    else:
                        prior_high_held = "Yes"
            
            # Check if prior quarter levels held
            if prior_quarter_high is not None and prior_quarter_low is not None:
                # Get the high and low of the current quarter
                current_quarter_high = after_range_quarter['High'].max()
                current_quarter_low = after_range_quarter['Low'].min()
                
                # Check if prior quarter high held
                if current_quarter_high > prior_quarter_high:
                    prior_quarter_high_held = "No"
                else:
                    prior_quarter_high_held = "Yes"
                
                # Check if prior quarter low held
                if current_quarter_low < prior_quarter_low:
                    prior_quarter_low_held = "No"
                else:
                    prior_quarter_low_held = "Yes"
            
            # Initialize return to range tracking
            returned_to_range = "No"  # Default to No, will be set to Yes if we return to range
            
            # Check for breakouts
            breakout_type = None
            breakout_outcome = "True"  # Default to True (successful breakout)
            wick_session_true = "True"  # Default to True (no wick failure)
            breakout_date = None
            days_to_breakout = None
            tuesday_week = None
            qpp_retrace = None
            max_extension = None
            max_retracement = None
            qpp_retrace_pct = None
            max_extension_pct = None
            max_retracement_pct = None
            
            # New time metrics for the events
            qpp_retrace_days = None
            max_extension_days = None
            max_retracement_days = None
            qpp_retrace_tuesday_week = None
            max_extension_tuesday_week = None
            max_retracement_tuesday_week = None
            second_breakout_days = None
            second_breakout_tuesday_week = None
            
            # Track subsequent prices
            high_pct_change = None
            low_pct_change = None
            
            # Initial high and low after range
            high_after = after_range_quarter['High'].max()
            low_after = after_range_quarter['Low'].min()
            
            # Track subsequent prices (original calculation using all data after range)
            high_pct_change = ((high_after - range_mid) / range_mid) * 100
            low_pct_change = ((low_after - range_mid) / range_mid) * 100
            
            # Initialize post-breakout percent changes
            post_breakout_high_pct_change = None
            post_breakout_low_pct_change = None
            
            # Initialize m7_retrace
            m7_retrace = None
            
            # New variables for tracking triple false and quarter close
            triple_false = "N/A"  # Default if no false breakout occurs
            quarter_close_price = None
            quarter_close_pct = None
            
            # Look for first closes above or below range
            closes_above = after_range_quarter[after_range_quarter['Close'] > range_high]
            closes_below = after_range_quarter[after_range_quarter['Close'] < range_low]
            
            # Check if we have any breakouts
            if not closes_above.empty and not closes_below.empty:
                # Both types of breakouts occurred, determine which came first
                first_close_above = closes_above.iloc[0]['Date']
                first_close_below = closes_below.iloc[0]['Date']
                
                if first_close_above < first_close_below:
                    # Long breakout occurred first
                    breakout_type = "Long"
                    breakout_date = first_close_above
                    
                    # Calculate post-breakout percent changes
                    after_breakout = after_range_quarter[after_range_quarter['Date'] > breakout_date]
                    if not after_breakout.empty:
                        post_breakout_high = after_breakout['High'].max()
                        post_breakout_low = after_breakout['Low'].min()
                        post_breakout_high_pct_change = ((post_breakout_high - range_mid) / range_mid) * 100
                        post_breakout_low_pct_change = ((post_breakout_low - range_mid) / range_mid) * 100
                    
                    # Calculate trading days from quarter start to breakout
                    trading_days_to_breakout = len(after_range_quarter[after_range_quarter['Date'] <= breakout_date])
                    days_to_breakout = trading_days_to_breakout
                    
                    # Calculate which Tuesday week the breakout falls into
                    # Find the first Tuesday on or after the quarter start
                    first_tuesday = friday['Date']
                    while first_tuesday.weekday() != 1:  # 1 = Tuesday
                        first_tuesday += pd.Timedelta(days=1)
                    
                    # Calculate which Tuesday-to-Tuesday week the breakout falls into
                    days_since_first_tuesday = (breakout_date - first_tuesday).days
                    tuesday_week = (days_since_first_tuesday // 7) + 1
                    
                    # Get data after breakout
                    after_breakout = after_range_quarter[after_range_quarter['Date'] > breakout_date]
                    
                    # Check if price returned to range after breakout
                    if not after_breakout.empty:
                        # For long breakouts, check if any low was below range_high
                        if any(after_breakout['Low'] < range_high):
                            returned_to_range = "Yes"
                        else:
                            returned_to_range = "No"
                    
                    # Get low before high (QPP retracement)
                    if not after_breakout.empty:
                        # Find high point after breakout
                        high_after_breakout = after_breakout['High'].max()
                        high_date = after_breakout[after_breakout['High'] == high_after_breakout].iloc[0]['Date']
                        
                        # Find lowest point between breakout and high
                        data_before_high = after_breakout[after_breakout['Date'] < high_date]
                        if not data_before_high.empty:
                            qpp_retrace = data_before_high['Low'].min()
                            qpp_retrace_pct = ((qpp_retrace - range_mid) / range_mid) * 100
                            
                            # Find the date when QPP retracement occurred
                            qpp_retrace_date = data_before_high[data_before_high['Low'] == qpp_retrace].iloc[0]['Date']
                            
                            # Calculate trading days from breakout to QPP retracement
                            qpp_retrace_days = len(data_before_high[data_before_high['Date'] <= qpp_retrace_date])
                            
                            # Calculate which Tuesday week the QPP retracement falls into
                            days_since_first_tuesday = (qpp_retrace_date - first_tuesday).days
                            qpp_retrace_tuesday_week = (days_since_first_tuesday // 7) + 1
                        
                        # Max extension is the highest high after breakout
                        max_extension = high_after_breakout
                        max_extension_pct = ((max_extension - range_mid) / range_mid) * 100
                        
                        # Calculate trading days from breakout to max extension
                        max_extension_days = len(after_breakout[after_breakout['Date'] <= high_date])
                        
                        # Calculate which Tuesday week the max extension falls into
                        days_since_first_tuesday = (high_date - first_tuesday).days
                        max_extension_tuesday_week = (days_since_first_tuesday // 7) + 1
                        
                        # Max retracement is the lowest low after breakout
                        max_retracement = after_breakout['Low'].min()
                        max_retracement_pct = ((max_retracement - range_mid) / range_mid) * 100
                        
                        # Find the date when max retracement occurred
                        max_retracement_date = after_breakout[after_breakout['Low'] == max_retracement].iloc[0]['Date']
                        
                        # Calculate trading days from breakout to max retracement
                        max_retracement_days = len(after_breakout[after_breakout['Date'] <= max_retracement_date])
                        
                        # Calculate which Tuesday week the max retracement falls into
                        days_since_first_tuesday = (max_retracement_date - first_tuesday).days
                        max_retracement_tuesday_week = (days_since_first_tuesday // 7) + 1
                    
                    # Check for false breakout (a close below range_low after the breakout)
                    if not after_breakout.empty and any(after_breakout['Close'] < range_low):
                        breakout_outcome = "False"
                        total_false_breakouts += 1
                        
                        # Find first close below range_low after the long breakout
                        false_date = after_breakout[after_breakout['Close'] < range_low].iloc[0]['Date']
                        
                        # Calculate timing metrics for second breakout
                        second_breakout_days = len(after_range_quarter[after_range_quarter['Date'] <= false_date])
                        days_since_first_tuesday = (false_date - first_tuesday).days
                        second_breakout_tuesday_week = (days_since_first_tuesday // 7) + 1
                        
                        # Look for a SUBSEQUENT LONG breakout (close above range_high again)
                        data_after_false = after_range_quarter[after_range_quarter['Date'] > false_date]
                        if not data_after_false.empty and any(data_after_false['Close'] > range_high):
                            triple_false = "True"
                            triple_false_count += 1
                        else:
                            triple_false = "False"
                    
                    # Check for wick session separately - any low below range_low
                    if not after_breakout.empty and any(after_breakout['Low'] < range_low):
                        wick_session_true = "False"  # Session failed due to a low below range_low
                    
                    # Calculate low before high
                    before_high = after_range_quarter[after_range_quarter['Date'] < breakout_date]
                    low_before_high = before_high['Low'].min() if not before_high.empty else None
                    m7_retrace = ((low_before_high - range_mid) / range_mid) * 100 if low_before_high is not None else None
                else:
                    # Short breakout occurred first
                    breakout_type = "Short"
                    breakout_date = first_close_below
                    
                    # Calculate post-breakout percent changes
                    after_breakout = after_range_quarter[after_range_quarter['Date'] > breakout_date]
                    if not after_breakout.empty:
                        post_breakout_high = after_breakout['High'].max()
                        post_breakout_low = after_breakout['Low'].min()
                        post_breakout_high_pct_change = ((post_breakout_high - range_mid) / range_mid) * 100
                        post_breakout_low_pct_change = ((post_breakout_low - range_mid) / range_mid) * 100
                    
                    # Calculate trading days from quarter start to breakout
                    trading_days_to_breakout = len(after_range_quarter[after_range_quarter['Date'] <= breakout_date])
                    days_to_breakout = trading_days_to_breakout
                    
                    # Calculate which Tuesday week the breakout falls into
                    # Find the first Tuesday on or after the quarter start
                    first_tuesday = friday['Date']
                    while first_tuesday.weekday() != 1:  # 1 = Tuesday
                        first_tuesday += pd.Timedelta(days=1)
                    
                    # Calculate which Tuesday-to-Tuesday week the breakout falls into
                    days_since_first_tuesday = (breakout_date - first_tuesday).days
                    tuesday_week = (days_since_first_tuesday // 7) + 1
                    
                    # Get data after breakout
                    after_breakout = after_range_quarter[after_range_quarter['Date'] > breakout_date]
                    
                    # Check if price returned to range after breakout
                    if not after_breakout.empty:
                        # For short breakouts, check if any high was above range_low
                        if any(after_breakout['High'] > range_low):
                            returned_to_range = "Yes"
                        else:
                            returned_to_range = "No"
                    
                    # Get high before low (QPP retracement)
                    if not after_breakout.empty:
                        # Find low point after breakout
                        low_after_breakout = after_breakout['Low'].min()
                        low_date = after_breakout[after_breakout['Low'] == low_after_breakout].iloc[0]['Date']
                        
                        # Find highest point between breakout and low
                        data_before_low = after_breakout[after_breakout['Date'] < low_date]
                        if not data_before_low.empty:
                            qpp_retrace = data_before_low['High'].max()
                            qpp_retrace_pct = ((qpp_retrace - range_mid) / range_mid) * 100
                            
                            # Find the date when QPP retracement occurred
                            qpp_retrace_date = data_before_low[data_before_low['High'] == qpp_retrace].iloc[0]['Date']
                            
                            # Calculate trading days from breakout to QPP retracement
                            qpp_retrace_days = len(data_before_low[data_before_low['Date'] <= qpp_retrace_date])
                            
                            # Calculate which Tuesday week the QPP retracement falls into
                            days_since_first_tuesday = (qpp_retrace_date - first_tuesday).days
                            qpp_retrace_tuesday_week = (days_since_first_tuesday // 7) + 1
                        
                        # Max extension is the lowest low after breakout
                        max_extension = low_after_breakout
                        max_extension_pct = ((max_extension - range_mid) / range_mid) * 100
                        
                        # Calculate trading days from breakout to max extension
                        max_extension_days = len(after_breakout[after_breakout['Date'] <= low_date])
                        
                        # Calculate which Tuesday week the max extension falls into
                        days_since_first_tuesday = (low_date - first_tuesday).days
                        max_extension_tuesday_week = (days_since_first_tuesday // 7) + 1
                        
                        # Max retracement is the highest high after breakout
                        max_retracement = after_breakout['High'].max()
                        max_retracement_pct = ((max_retracement - range_mid) / range_mid) * 100
                        
                        # Find the date when max retracement occurred
                        max_retracement_date = after_breakout[after_breakout['High'] == max_retracement].iloc[0]['Date']
                        
                        # Calculate trading days from breakout to max retracement
                        max_retracement_days = len(after_breakout[after_breakout['Date'] <= max_retracement_date])
                        
                        # Calculate which Tuesday week the max retracement falls into
                        days_since_first_tuesday = (max_retracement_date - first_tuesday).days
                        max_retracement_tuesday_week = (days_since_first_tuesday // 7) + 1
                    
                    # Check for false breakout (a close above range_high after the breakout)
                    if not after_breakout.empty and any(after_breakout['Close'] > range_high):
                        breakout_outcome = "False"
                        total_false_breakouts += 1
                        
                        # Find first close above range_high after the short breakout
                        false_date = after_breakout[after_breakout['Close'] > range_high].iloc[0]['Date']
                        
                        # Calculate timing metrics for second breakout
                        second_breakout_days = len(after_range_quarter[after_range_quarter['Date'] <= false_date])
                        days_since_first_tuesday = (false_date - first_tuesday).days
                        second_breakout_tuesday_week = (days_since_first_tuesday // 7) + 1
                        
                        # Look for a SUBSEQUENT SHORT breakout (close below range_low again)
                        data_after_false = after_range_quarter[after_range_quarter['Date'] > false_date]
                        if not data_after_false.empty and any(data_after_false['Close'] < range_low):
                            triple_false = "True"
                            triple_false_count += 1
                        else:
                            triple_false = "False"
                    
                    # Check for wick session separately - any high above range_high
                    if not after_breakout.empty and any(after_breakout['High'] > range_high):
                        wick_session_true = "False"  # Session failed due to a high above range_high
                    
                    # Calculate high before low
                    before_low = after_range_quarter[after_range_quarter['Date'] < breakout_date]
                    high_before_low = before_low['High'].max() if not before_low.empty else None
                    m7_retrace = ((high_before_low - range_mid) / range_mid) * 100 if high_before_low is not None else None
            
            elif not closes_above.empty:
                # Only long breakout occurred
                breakout_type = "Long"
                breakout_date = closes_above.iloc[0]['Date']
                
                # Calculate post-breakout percent changes
                after_breakout = after_range_quarter[after_range_quarter['Date'] > breakout_date]
                if not after_breakout.empty:
                    post_breakout_high = after_breakout['High'].max()
                    post_breakout_low = after_breakout['Low'].min()
                    post_breakout_high_pct_change = ((post_breakout_high - range_mid) / range_mid) * 100
                    post_breakout_low_pct_change = ((post_breakout_low - range_mid) / range_mid) * 100
                
                # Calculate trading days from quarter start to breakout
                trading_days_to_breakout = len(after_range_quarter[after_range_quarter['Date'] <= breakout_date])
                days_to_breakout = trading_days_to_breakout
                
                # Calculate which Tuesday week the breakout falls into
                # Find the first Tuesday on or after the quarter start
                first_tuesday = friday['Date']
                while first_tuesday.weekday() != 1:  # 1 = Tuesday
                    first_tuesday += pd.Timedelta(days=1)
                
                # Calculate which Tuesday-to-Tuesday week the breakout falls into
                days_since_first_tuesday = (breakout_date - first_tuesday).days
                tuesday_week = (days_since_first_tuesday // 7) + 1
                
                # Get data after breakout
                after_breakout = after_range_quarter[after_range_quarter['Date'] > breakout_date]
                
                # Check if price returned to range after breakout
                if not after_breakout.empty:
                    # For long breakouts, check if any low was below range_high
                    if any(after_breakout['Low'] < range_high):
                        returned_to_range = "Yes"
                    else:
                        returned_to_range = "No"
                
                # Get low before high (QPP retracement)
                if not after_breakout.empty:
                    # Find high point after breakout
                    high_after_breakout = after_breakout['High'].max()
                    high_date = after_breakout[after_breakout['High'] == high_after_breakout].iloc[0]['Date']
                    
                    # Find lowest point between breakout and high
                    data_before_high = after_breakout[after_breakout['Date'] < high_date]
                    if not data_before_high.empty:
                        qpp_retrace = data_before_high['Low'].min()
                        qpp_retrace_pct = ((qpp_retrace - range_mid) / range_mid) * 100
                        
                        # Find the date when QPP retracement occurred
                        qpp_retrace_date = data_before_high[data_before_high['Low'] == qpp_retrace].iloc[0]['Date']
                        
                        # Calculate trading days from breakout to QPP retracement
                        qpp_retrace_days = len(data_before_high[data_before_high['Date'] <= qpp_retrace_date])
                        
                        # Calculate which Tuesday week the QPP retracement falls into
                        days_since_first_tuesday = (qpp_retrace_date - first_tuesday).days
                        qpp_retrace_tuesday_week = (days_since_first_tuesday // 7) + 1
                    
                    # Max extension is the highest high after breakout
                    max_extension = high_after_breakout
                    max_extension_pct = ((max_extension - range_mid) / range_mid) * 100
                    
                    # Calculate trading days from breakout to max extension
                    max_extension_days = len(after_breakout[after_breakout['Date'] <= high_date])
                    
                    # Calculate which Tuesday week the max extension falls into
                    days_since_first_tuesday = (high_date - first_tuesday).days
                    max_extension_tuesday_week = (days_since_first_tuesday // 7) + 1
                    
                    # Max retracement is the lowest low after breakout
                    max_retracement = after_breakout['Low'].min()
                    max_retracement_pct = ((max_retracement - range_mid) / range_mid) * 100
                    
                    # Find the date when max retracement occurred
                    max_retracement_date = after_breakout[after_breakout['Low'] == max_retracement].iloc[0]['Date']
                    
                    # Calculate trading days from breakout to max retracement
                    max_retracement_days = len(after_breakout[after_breakout['Date'] <= max_retracement_date])
                    
                    # Calculate which Tuesday week the max retracement falls into
                    days_since_first_tuesday = (max_retracement_date - first_tuesday).days
                    max_retracement_tuesday_week = (days_since_first_tuesday // 7) + 1
                
                # Check for false breakout (close-based)
                if not after_breakout.empty and any(after_breakout['Close'] < range_low):
                    breakout_outcome = "False"
                    total_false_breakouts += 1
                    
                    # Find first close below range_low after the long breakout
                    false_date = after_breakout[after_breakout['Close'] < range_low].iloc[0]['Date']
                    
                    # Calculate timing metrics for second breakout
                    second_breakout_days = len(after_range_quarter[after_range_quarter['Date'] <= false_date])
                    days_since_first_tuesday = (false_date - first_tuesday).days
                    second_breakout_tuesday_week = (days_since_first_tuesday // 7) + 1
                    
                    # Look for a SUBSEQUENT LONG breakout (close above range_high again)
                    data_after_false = after_range_quarter[after_range_quarter['Date'] > false_date]
                    if not data_after_false.empty and any(data_after_false['Close'] > range_high):
                        triple_false = "True"
                        triple_false_count += 1
                    else:
                        triple_false = "False"
                
                # Check for wick session separately (wick-based)
                if not after_breakout.empty and any(after_breakout['Low'] < range_low):
                    wick_session_true = "False"  # Session failed due to a low below range_low
                
                # Calculate low before high
                before_high = after_range_quarter[after_range_quarter['Date'] < breakout_date]
                low_before_high = before_high['Low'].min() if not before_high.empty else None
                m7_retrace = ((low_before_high - range_mid) / range_mid) * 100 if low_before_high is not None else None
                
            elif not closes_below.empty:
                # Only short breakout occurred
                breakout_type = "Short"
                breakout_date = closes_below.iloc[0]['Date']
                
                # Calculate post-breakout percent changes
                after_breakout = after_range_quarter[after_range_quarter['Date'] > breakout_date]
                if not after_breakout.empty:
                    post_breakout_high = after_breakout['High'].max()
                    post_breakout_low = after_breakout['Low'].min()
                    post_breakout_high_pct_change = ((post_breakout_high - range_mid) / range_mid) * 100
                    post_breakout_low_pct_change = ((post_breakout_low - range_mid) / range_mid) * 100
                
                # Calculate trading days from quarter start to breakout
                trading_days_to_breakout = len(after_range_quarter[after_range_quarter['Date'] <= breakout_date])
                days_to_breakout = trading_days_to_breakout
                
                # Calculate which Tuesday week the breakout falls into
                # Find the first Tuesday on or after the quarter start
                first_tuesday = friday['Date']
                while first_tuesday.weekday() != 1:  # 1 = Tuesday
                    first_tuesday += pd.Timedelta(days=1)
                
                # Calculate which Tuesday-to-Tuesday week the breakout falls into
                days_since_first_tuesday = (breakout_date - first_tuesday).days
                tuesday_week = (days_since_first_tuesday // 7) + 1
                
                # Get data after breakout
                after_breakout = after_range_quarter[after_range_quarter['Date'] > breakout_date]
                
                # Check if price returned to range after breakout
                if not after_breakout.empty:
                    # For short breakouts, check if any high was above range_low
                    if any(after_breakout['High'] > range_low):
                        returned_to_range = "Yes"
                    else:
                        returned_to_range = "No"
                
                # Get high before low (QPP retracement)
                if not after_breakout.empty:
                    # Find low point after breakout
                    low_after_breakout = after_breakout['Low'].min()
                    low_date = after_breakout[after_breakout['Low'] == low_after_breakout].iloc[0]['Date']
                    
                    # Find highest point between breakout and low
                    data_before_low = after_breakout[after_breakout['Date'] < low_date]
                    if not data_before_low.empty:
                        qpp_retrace = data_before_low['High'].max()
                        qpp_retrace_pct = ((qpp_retrace - range_mid) / range_mid) * 100
                        
                        # Find the date when QPP retracement occurred
                        qpp_retrace_date = data_before_low[data_before_low['High'] == qpp_retrace].iloc[0]['Date']
                        
                        # Calculate trading days from breakout to QPP retracement
                        qpp_retrace_days = len(data_before_low[data_before_low['Date'] <= qpp_retrace_date])
                        
                        # Calculate which Tuesday week the QPP retracement falls into
                        days_since_first_tuesday = (qpp_retrace_date - first_tuesday).days
                        qpp_retrace_tuesday_week = (days_since_first_tuesday // 7) + 1
                    
                    # Max extension is the lowest low after breakout
                    max_extension = low_after_breakout
                    max_extension_pct = ((max_extension - range_mid) / range_mid) * 100
                    
                    # Calculate trading days from breakout to max extension
                    max_extension_days = len(after_breakout[after_breakout['Date'] <= low_date])
                    
                    # Calculate which Tuesday week the max extension falls into
                    days_since_first_tuesday = (low_date - first_tuesday).days
                    max_extension_tuesday_week = (days_since_first_tuesday // 7) + 1
                    
                    # Max retracement is the highest high after breakout
                    max_retracement = after_breakout['High'].max()
                    max_retracement_pct = ((max_retracement - range_mid) / range_mid) * 100
                    
                    # Find the date when max retracement occurred
                    max_retracement_date = after_breakout[after_breakout['High'] == max_retracement].iloc[0]['Date']
                    
                    # Calculate trading days from breakout to max retracement
                    max_retracement_days = len(after_breakout[after_breakout['Date'] <= max_retracement_date])
                    
                    # Calculate which Tuesday week the max retracement falls into
                    days_since_first_tuesday = (max_retracement_date - first_tuesday).days
                    max_retracement_tuesday_week = (days_since_first_tuesday // 7) + 1
                
                # Check for false breakout (close-based)
                if not after_breakout.empty and any(after_breakout['Close'] > range_high):
                    breakout_outcome = "False"
                    total_false_breakouts += 1
                    
                    # Find first close above range_high after the short breakout
                    false_date = after_breakout[after_breakout['Close'] > range_high].iloc[0]['Date']
                    
                    # Calculate timing metrics for second breakout
                    second_breakout_days = len(after_range_quarter[after_range_quarter['Date'] <= false_date])
                    days_since_first_tuesday = (false_date - first_tuesday).days
                    second_breakout_tuesday_week = (days_since_first_tuesday // 7) + 1
                    
                    # Look for a SUBSEQUENT SHORT breakout (close below range_low again)
                    data_after_false = after_range_quarter[after_range_quarter['Date'] > false_date]
                    if not data_after_false.empty and any(data_after_false['Close'] < range_low):
                        triple_false = "True"
                        triple_false_count += 1
                    else:
                        triple_false = "False"
                
                # Check for wick session separately (wick-based)
                if not after_breakout.empty and any(after_breakout['High'] > range_high):
                    wick_session_true = "False"  # Session failed due to a high above range_high
                
                # Calculate high before low
                before_low = after_range_quarter[after_range_quarter['Date'] < breakout_date]
                high_before_low = before_low['High'].max() if not before_low.empty else None
                m7_retrace = ((high_before_low - range_mid) / range_mid) * 100 if high_before_low is not None else None
            
            # Calculate where quarter closes in % from mid
            if not after_range_quarter.empty:
                # Get the last price in the quarter
                quarter_close_price = after_range_quarter.iloc[-1]['Close']
                quarter_close_pct = ((quarter_close_price - range_mid) / range_mid) * 100
                
                # Determine where the quarter closes relative to the range
                if quarter_close_price > range_high:
                    quarter_close_position = "Above"
                elif quarter_close_price < range_low:
                    quarter_close_position = "Below"
                else:
                    quarter_close_position = "Inside"
            
            # Determine range type compared to prior range
            range_type = "N/A"  # First range has no prior
            if prior_range:
                prior_low, prior_high, prior_mid = prior_range
                range_type = get_range_type(range_low, range_high, prior_low, prior_high, prior_mid)
            
            # Save the current range for next comparison
            prior_range = (range_low, range_high, range_mid)
            prior_range_low = range_low
            prior_range_high = range_high
            prior_quarter_high = high_after
            prior_quarter_low = low_after
            
            # Quarter label (1-4)
            quarter_num = quarter
            
            # Get thresholds for bucketing based on breakout type and outcome
            if breakout_type is not None and breakout_outcome is not None:
                # Debug prints
                print(f"\nProcessing quarter {year} Q{quarter}")
                print(f"Breakout Type: {breakout_type}, Outcome: {breakout_outcome}")
                print(f"High PCT Change: {high_pct_change}")
                print(f"Low PCT Change: {low_pct_change}")
                print(f"Close PCT Change: {quarter_close_pct}")
                
                # Look up thresholds using exact string matching
                key = (str(breakout_type).strip(), str(breakout_outcome).strip())
                print(f"Looking for key: {key}")
                
                if key in thresholds_dict:
                    thresholds = thresholds_dict[key]
                    print(f"Using thresholds: {thresholds}")
                    
                    # Bucket the high and low percentage changes
                    high_pct_bucket = determine_bucket(high_pct_change, thresholds)
                    low_pct_bucket = determine_bucket(low_pct_change, thresholds)
                    close_pct_bucket = determine_bucket(quarter_close_pct, thresholds)
                    
                    print(f"Buckets assigned:")
                    print(f"High PCT Bucket: {high_pct_bucket}")
                    print(f"Low PCT Bucket: {low_pct_bucket}")
                    print(f"Close PCT Bucket: {close_pct_bucket}")
                else:
                    print(f"No thresholds found for {key}")
                    high_pct_bucket = "N/A"
                    low_pct_bucket = "N/A"
                    close_pct_bucket = "N/A"
            else:
                print(f"No breakout type/outcome for {year} Q{quarter}")
                high_pct_bucket = "N/A"
                low_pct_bucket = "N/A"
                close_pct_bucket = "N/A"
            
            # Record results
            quarter_info = {
                'year': year,
                'quarter': quarter_num,
                'range_low': range_low,
                'range_high': range_high,
                'range_mid': range_mid,
                'breakout_type': breakout_type,
                'breakout_outcome': breakout_outcome,
                'wick_session_true': wick_session_true,
                'days_to_breakout': days_to_breakout,
                'tuesday_week': tuesday_week,
                'qpp_retrace': qpp_retrace,
                'qpp_retrace_pct': qpp_retrace_pct,
                'qpp_retrace_days': qpp_retrace_days,
                'qpp_retrace_tuesday_week': qpp_retrace_tuesday_week,
                'max_extension': max_extension,
                'max_extension_pct': max_extension_pct,
                'max_extension_days': max_extension_days,
                'max_extension_tuesday_week': max_extension_tuesday_week,
                'max_retracement': max_retracement,
                'max_retracement_pct': max_retracement_pct,
                'max_retracement_days': max_retracement_days,
                'max_retracement_tuesday_week': max_retracement_tuesday_week,
                'second_breakout_days': second_breakout_days,
                'second_breakout_tuesday_week': second_breakout_tuesday_week,
                'quarter_high_week': quarter_high_week,
                'quarter_low_week': quarter_low_week,
                'high_pct_change': high_pct_change,
                'low_pct_change': low_pct_change,
                'post_breakout_high_pct_change': post_breakout_high_pct_change,
                'post_breakout_low_pct_change': post_breakout_low_pct_change,
                'm7_retrace': m7_retrace,
                'range_type': range_type,
                'triple_false': triple_false,
                'quarter_close_price': quarter_close_price,
                'quarter_close_pct': quarter_close_pct,
                'prior_breakout_type': prior_breakout_type,
                'prior_breakout_outcome': prior_breakout_outcome,
                'returned_to_range': returned_to_range,
                'prior_range_low_held': prior_low_held,
                'prior_range_high_held': prior_high_held,
                'prior_quarter_high_held': prior_quarter_high_held,
                'prior_quarter_low_held': prior_quarter_low_held,
                'quarter_close_position': quarter_close_position,
                'quarter_high_time': high_bucket,
                'quarter_low_time': low_bucket,
                'high_pct_bucket': high_pct_bucket,
                'low_pct_bucket': low_pct_bucket,
                'close_pct_bucket': close_pct_bucket
            }
            
            # Add weekly statistics as separate columns
            for week in weekly_highs.keys():
                quarter_info[f'week_{week}_high'] = weekly_highs[week]
                quarter_info[f'week_{week}_low'] = weekly_lows[week]
                quarter_info[f'week_{week}_high_pct'] = weekly_highs_pct[week]
                quarter_info[f'week_{week}_low_pct'] = weekly_lows_pct[week]
                quarter_info[f'week_{week}_range'] = weekly_highs_pct[week] - weekly_lows_pct[week]
            
            results.append(quarter_info)
            
            # Update the prior breakout information for the next iteration
            prior_breakout_type = breakout_type
            prior_breakout_outcome = breakout_outcome
    
    # Calculate and print the triple false rate
    if total_false_breakouts > 0:
        triple_false_rate = (triple_false_count / total_false_breakouts) * 100
        print(f"\nTriple False Rate: {triple_false_rate:.2f}% ({triple_false_count} out of {total_false_breakouts} false breakouts)")
    else:
        print("\nNo false breakouts detected, cannot calculate triple false rate or second breakout statistics.")
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Calculate and print second breakout timing statistics if we have false breakouts
    if total_false_breakouts > 0:
        second_breakout_days = results_df[results_df['second_breakout_days'].notna()]['second_breakout_days']
        second_breakout_weeks = results_df[results_df['second_breakout_tuesday_week'].notna()]['second_breakout_tuesday_week']
        
        print("\nSecond Breakout Timing Statistics:")
        print("Days from Quarter Start to Second Breakout:")
        print(f"Min: {second_breakout_days.min()}")
        print(f"Max: {second_breakout_days.max()}")
        print(f"Average: {second_breakout_days.mean():.2f}")
        print(f"Median: {second_breakout_days.median():.2f}")
        print(f"20th percentile: {second_breakout_days.quantile(0.2):.2f}")
        print(f"80th percentile: {second_breakout_days.quantile(0.8):.2f}")
        
        print("\nTuesday Week Distribution of Second Breakouts:")
        week_distribution = second_breakout_weeks.value_counts().sort_index()
        for week, count in week_distribution.items():
            print(f"Week {week}: {count} occurrences ({count/len(second_breakout_weeks)*100:.1f}%)")
    
    # Calculate and print weekly high/low statistics
    print("\nWeekly High/Low Statistics by Tuesday Week:")
    
    # Get all unique weeks from both high and low weeks
    all_weeks = set(results_df['quarter_high_week'].dropna().unique())
    all_weeks.update(results_df['quarter_low_week'].dropna().unique())
    
    for week in sorted(all_weeks):
        # Get all quarters where high occurred in this week
        week_highs = results_df[results_df['quarter_high_week'] == week]['high_pct_change'].dropna()
        # Get all quarters where low occurred in this week
        week_lows = results_df[results_df['quarter_low_week'] == week]['low_pct_change'].dropna()
        
        if not week_highs.empty or not week_lows.empty:
            print(f"\nWeek {week}:")
            if not week_highs.empty:
                print(f"Highs - Avg: {week_highs.mean():.2f}%, Median: {week_highs.median():.2f}%, Max: {week_highs.max():.2f}%")
            if not week_lows.empty:
                print(f"Lows - Avg: {week_lows.mean():.2f}%, Median: {week_lows.median():.2f}%, Min: {week_lows.min():.2f}%")
            
            # Calculate range statistics for quarters where both high and low occurred in this week
            week_ranges = []
            for _, row in results_df[results_df['quarter_high_week'] == week].iterrows():
                if row['quarter_low_week'] == week:
                    week_ranges.append(row['high_pct_change'] - row['low_pct_change'])
            
            if week_ranges:
                print(f"Range - Avg: {np.mean(week_ranges):.2f}%, Median: {np.median(week_ranges):.2f}%")
    
    return results_df

def calculate_quarterly_statistics(df):
    """Calculate statistics about trading days and Tuesdays per quarter.
    Quarters are defined from first Friday to first Friday (next quarter)."""
    quarter_stats = []
    
    # Find all first Fridays for each quarter
    first_fridays = []
    for year in range(1962, 2025):
        for quarter in range(1, 5):
            friday, _ = get_first_friday_and_previous_day(df, year, quarter)
            if friday is not None:
                first_fridays.append((year, quarter, friday['Date']))
    
    # Sort by date
    first_fridays.sort(key=lambda x: x[2])
    
    # Calculate stats for each quarter (from one first Friday to the next)
    for i in range(len(first_fridays) - 1):
        current_yr, current_q, current_friday = first_fridays[i]
        next_yr, next_q, next_friday = first_fridays[i+1]
        
        # Get days in this period (first Friday to day before next first Friday)
        quarter_data = df[(df['Date'] >= current_friday) & (df['Date'] < next_friday)]
        days_count = len(quarter_data)
        
        # Count Tuesdays (weekday=1 in pandas)
        date_range = pd.date_range(current_friday, next_friday - pd.Timedelta(days=1))
        tuesdays_count = sum(date.weekday() == 1 for date in date_range)
        
        quarter_stats.append({
            'year': current_yr,
            'quarter': current_q,
            'start_date': current_friday,
            'end_date': next_friday - pd.Timedelta(days=1),
            'days_count': days_count,
            'tuesdays_count': tuesdays_count
        })
    
    stats_df = pd.DataFrame(quarter_stats)
    
    # Calculate statistics
    days_stats = {
        'min': stats_df['days_count'].min(),
        'max': stats_df['days_count'].max(),
        'average': stats_df['days_count'].mean(),
        'median': stats_df['days_count'].median(),
        'p20': stats_df['days_count'].quantile(0.2),
        'p80': stats_df['days_count'].quantile(0.8)
    }
    
    tuesdays_stats = {
        'min': stats_df['tuesdays_count'].min(),
        'max': stats_df['tuesdays_count'].max(),
        'average': stats_df['tuesdays_count'].mean(),
        'median': stats_df['tuesdays_count'].median(),
        'p20': stats_df['tuesdays_count'].quantile(0.2),
        'p80': stats_df['tuesdays_count'].quantile(0.8)
    }
    
    # Get frequency distributions
    days_freq = stats_df['days_count'].value_counts().sort_index()
    tuesdays_freq = stats_df['tuesdays_count'].value_counts().sort_index()
    
    return days_stats, tuesdays_stats, stats_df, days_freq, tuesdays_freq

def generate_quarterly_candles(df, quarterly_ranges):
    """Generate normalized quarterly data with 3 time buckets using linear regression."""
    normalized_data = []
    
    # Iterate through each quarter in the quarterly ranges
    for _, quarter_row in quarterly_ranges.iterrows():
        year = quarter_row['year']
        quarter = quarter_row['quarter']
        range_mid = quarter_row['range_mid']
        
        # Get the first Friday of the quarter
        friday, _ = get_first_friday_and_previous_day(df, year, quarter)
        if friday is None:
            continue
            
        # Get the date range for this quarter (from day after first Friday to end of quarter)
        if quarter == 1:
            end_date = f"{year}-03-31"
        elif quarter == 2:
            end_date = f"{year}-06-30"
        elif quarter == 3:
            end_date = f"{year}-09-30"
        else:
            end_date = f"{year}-12-31"
        
        # Get all candles for this quarter starting from the day after the first Friday
        start_date = friday['Date'] + pd.Timedelta(days=1)
        quarter_candles = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        if len(quarter_candles) == 0:
            continue
            
        # Calculate percentage changes from range mid
        highs = ((quarter_candles['High'] - range_mid) / range_mid) * 100
        lows = ((quarter_candles['Low'] - range_mid) / range_mid) * 100
        
        # Create x-axis (1 to number of days)
        x = np.arange(1, len(quarter_candles) + 1)
        
        # Fit linear regression for highs
        high_slope, high_intercept = np.polyfit(x, highs, 1)
        high_line = np.poly1d([high_slope, high_intercept])
        
        # Fit linear regression for lows
        low_slope, low_intercept = np.polyfit(x, lows, 1)
        low_line = np.poly1d([low_slope, low_intercept])
        
        # Calculate 3 evenly spaced points
        points = np.linspace(1, len(quarter_candles), 3)
        
        # Get predicted values at these points
        high_predictions = high_line(points)
        low_predictions = low_line(points)
        
        # Create normalized data points
        for i, (high_pct, low_pct) in enumerate(zip(high_predictions, low_predictions)):
            normalized_data.append({
                'Year': year,
                'Quarter': quarter,
                'TimeBucket': i + 1,  # 1 = start, 2 = middle, 3 = end
                'High': high_pct,
                'Low': low_pct,
                'HighSlope': high_slope,
                'LowSlope': low_slope,
                'HighIntercept': high_intercept,
                'LowIntercept': low_intercept
            })
    
    return pd.DataFrame(normalized_data)

def main():
    file_path = "/Users/orlandocantoni/Desktop/RBC auto model/SPX_1D.csv"
    
    # Load and clean data
    print("Loading and cleaning data...")
    df = load_and_clean_data(file_path)
    
    # Debug: Print loaded data info
    print(f"Loaded data range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Total data points: {len(df)}")
    
    # Calculate quarterly statistics (first Friday to first Friday)
    print("\nCalculating quarterly statistics (first Friday to first Friday)...")
    days_stats, tuesdays_stats, stats_df, days_freq, tuesdays_freq = calculate_quarterly_statistics(df)
    
    # Print quarterly statistics
    print("\nTrading Days per Quarter (first Friday to first Friday):")
    print(f"Min: {days_stats['min']}")
    print(f"Max: {days_stats['max']}")
    print(f"Average: {days_stats['average']:.2f}")
    print(f"Median: {days_stats['median']:.2f}")
    print(f"20th percentile: {days_stats['p20']:.2f}")
    print(f"80th percentile: {days_stats['p80']:.2f}")
    
    # Print frequency distribution of trading days
    print("\nFrequency of Trading Days per Quarter:")
    for days, count in days_freq.items():
        print(f"{days} trading days occurred {count} times")
    
    print("\nTuesdays per Quarter (first Friday to first Friday):")
    print(f"Min: {tuesdays_stats['min']}")
    print(f"Max: {tuesdays_stats['max']}")
    print(f"Average: {tuesdays_stats['average']:.2f}")
    print(f"Median: {tuesdays_stats['median']:.2f}")
    print(f"20th percentile: {tuesdays_stats['p20']:.2f}")
    print(f"80th percentile: {tuesdays_stats['p80']:.2f}")
    
    # Print frequency distribution of Tuesdays
    print("\nFrequency of Tuesdays per Quarter:")
    for tuesdays, count in tuesdays_freq.items():
        print(f"{tuesdays} Tuesdays occurred {count} times")
    
    # Analyze quarterly ranges
    print("Analyzing quarterly ranges...")
    results = analyze_quarterly_ranges(df)

    # Save results to CSV
    results.to_csv("SPX_Quarterly_Analysis_Results.csv", index=False)
    print(f"Results saved to SPX_Quarterly_Analysis_Results.csv")
    
    # Generate and save normalized quarterly data
    print("\nGenerating normalized quarterly data...")
    normalized_df = generate_quarterly_candles(df, results)
    normalized_df.to_csv("SPX_Quarterly_Normalized.csv", index=False)
    print(f"Normalized quarterly data saved to SPX_Quarterly_Normalized.csv")
    
    # Print summary statistics
    print("\nQuarterly Range Analysis Summary:")
    print(f"Total quarters analyzed: {len(results)}")
    
    # Only proceed with analysis if we have results
    if len(results) > 0:
        print(f"Long breakouts: {len(results[results['breakout_type'] == 'Long'])}")
        print(f"Short breakouts: {len(results[results['breakout_type'] == 'Short'])}")
        print(f"False breakouts: {len(results[results['breakout_outcome'] == 'False'])}")
        print(f"True breakouts: {len(results[results['breakout_outcome'] == 'True'])}")
        print(f"Wick session true: {len(results[results['wick_session_true'] == 'True'])}")
        
        # Print range type distribution
        print("\nRange Type Distribution:")
        print(results['range_type'].value_counts())
        
        # Print time bucket distribution
        print("\nQuarter High Time Distribution:")
        print(results['quarter_high_time'].value_counts().sort_index())
        print("\nQuarter Low Time Distribution:")
        print(results['quarter_low_time'].value_counts().sort_index())
    
    else:
        print("No quarterly data was found to analyze. Please check your data file and date ranges.")
        
    return results

if __name__ == "__main__":
    results = main() 