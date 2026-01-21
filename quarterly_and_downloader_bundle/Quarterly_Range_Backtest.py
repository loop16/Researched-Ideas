import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

class QuarterlyRangeBacktest:
    """
    Backtesting engine for quarterly range breakout strategy.
    
    Strategy:
    1. Wait for first breakout (long or short) from quarterly range
    2. Enter position in direction of breakout
    3. Hold until end of quarter or stop out if opposite side of range is breached
    4. Track performance metrics
    """
    
    def __init__(self, data_file, daily_file='SPX_1D.csv', start_year=1962, end_year=2024):
        """
        Initialize the backtesting engine.
        
        Args:
            data_file (str): Path to the quarterly analysis results CSV
            daily_file (str): Path to the daily data CSV
            start_year (int): Start year for backtesting
            end_year (int): End year for backtesting
        """
        self.data_file = data_file
        self.daily_file = daily_file
        self.start_year = start_year
        self.end_year = end_year
        self.results = None
        self.trades = []
        self.flipping_trades = []
        self.performance_metrics = {}
        self.flipping_performance_metrics = {}
        self.daily_df = None
        self.trades_df = None
        self.flipping_trades_df = None
        self.equity_curve = None
        self.flipping_equity_curve = None
        self.long_only_equity_curve = None
        self.long_only_trades_df = None
        self.qpp_levels = None
        
    def load_data(self):
        """Load and prepare the quarterly analysis data."""
        print("Loading quarterly analysis data...")
        self.results = pd.read_csv(self.data_file)
        
        # Normalize breakout_outcome column for robust logic
        if 'breakout_outcome' in self.results.columns:
            self.results['breakout_outcome'] = (
                self.results['breakout_outcome']
                .astype(str)
                .str.strip()
                .str.lower()
            )
        
        # Print unique values for debugging
        print("Unique breakout_outcome values:", self.results['breakout_outcome'].unique())
        
        # Convert year and quarter to datetime for easier filtering
        self.results['date'] = pd.to_datetime(self.results['year'].astype(str) + '-' + 
                                            (self.results['quarter'] * 3).astype(str) + '-01')
        
        # Filter by year range
        self.results = self.results[
            (self.results['year'] >= self.start_year) & 
            (self.results['year'] <= self.end_year)
        ]
        
        # Only include quarters with valid breakouts
        self.results = self.results[
            (self.results['breakout_type'].notna()) & 
            (self.results['breakout_type'].isin(['Long', 'Short']))
        ]
        
        print(f"Loaded {len(self.results)} quarters with valid breakouts")
        print(f"Date range: {self.results['year'].min()} to {self.results['year'].max()}")
        
        # Load daily data for day-by-day simulation
        print("Loading daily data for day-by-day simulation...")
        self.daily_df = pd.read_csv(self.daily_file)
        
        # Rename columns to match expected format
        self.daily_df = self.daily_df.rename(columns={
            'time': 'Date',
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close'
        })
        
        self.daily_df['Date'] = pd.to_datetime(self.daily_df['Date'])
        self.daily_df = self.daily_df.sort_values('Date')
        print(f"Loaded {len(self.daily_df)} daily bars.")
        
        # Load QPP indicator levels
        print("Loading QPP indicator levels...")
        try:
            self.qpp_levels = pd.read_csv('SPX_Quarterly_Percentiles.csv')
            print(f"Loaded QPP levels for {len(self.qpp_levels)} breakout types")
        except FileNotFoundError:
            print("Warning: QPP levels file not found. Short breakout filtering will be disabled.")
            self.qpp_levels = None
        
        return self.results
    
    def get_quarter_range_dates(self, year, quarter):
        """Helper to get first Friday and previous day for a quarter."""
        if quarter == 1:
            first_day = f"{year}-01-01"
        elif quarter == 2:
            first_day = f"{year}-04-01"
        elif quarter == 3:
            first_day = f"{year}-07-01"
        else:
            first_day = f"{year}-10-01"
        
        quarter_days = self.daily_df[(self.daily_df['Date'] >= first_day) &
                                     (self.daily_df['Date'] < pd.to_datetime(first_day) + pd.DateOffset(months=3))]
        first_fridays = quarter_days[quarter_days['Date'].dt.weekday == 4]
        if first_fridays.empty:
            return None, None
        first_friday = first_fridays.iloc[0]
        previous_days = self.daily_df[self.daily_df['Date'] < first_friday['Date']]
        previous_day = previous_days.iloc[-1] if not previous_days.empty else None
        return first_friday, previous_day

    def simulate_flipping_strategy_for_quarter(self, year, quarter):
        """Simulate flipping strategy with quarterly expiration exits."""
        # Get range for the quarter
        first_friday, previous_day = self.get_quarter_range_dates(year, quarter)
        if first_friday is None or previous_day is None:
            return []
        
        range_high = max(first_friday['High'], previous_day['High'])
        range_low = min(first_friday['Low'], previous_day['Low'])
        range_mid = (range_high + range_low) / 2
        
        # Get prior range for comparison
        prior_range_high = None
        prior_range_low = None
        
        # Find the prior quarter's range
        if quarter == 1:
            prior_year = year - 1
            prior_quarter = 4
        else:
            prior_year = year
            prior_quarter = quarter - 1
        
        prior_friday, prior_previous_day = self.get_quarter_range_dates(prior_year, prior_quarter)
        if prior_friday is not None and prior_previous_day is not None:
            prior_range_high = max(prior_friday['High'], prior_previous_day['High'])
            prior_range_low = min(prior_friday['Low'], prior_previous_day['Low'])
        
        # Get all days in the quarter after the range is set
        quarter_start = first_friday['Date']
        quarter_end = quarter_start + pd.DateOffset(months=3)
        
        quarter_days = self.daily_df[(self.daily_df['Date'] > quarter_start) & 
                                   (self.daily_df['Date'] < quarter_end)].copy()
        
        if quarter_days.empty:
            return []
        
        # Get the quarter expiration date (last trading day of the quarter)
        quarter_expiration_date = quarter_days.iloc[-1]['Date']
        quarter_expiration_price = quarter_days.iloc[-1]['Close']
        
        # Determine if we should allow short trades based on prior range
        allow_shorts = True
        if prior_range_high is not None and prior_range_low is not None:
            # If prior range is below current range (uptrend), skip short trades
            if prior_range_high < range_low:
                allow_shorts = False
                print(f"Q{year} Q{quarter}: Shorts disabled - prior range below current range (uptrend)")
        
        # Simulate flipping strategy with quarterly expiration
        trades = []
        position = None
        entry_price = None
        entry_date = None
        direction = None
        
        for i, row in quarter_days.iterrows():
            date = row['Date']
            close = row['Close']
            
            # If no position, look for breakout
            if position is None:
                if close > range_high:
                    position = 'Long'
                    entry_price = close
                    entry_date = date
                    direction = 'Long'
                elif close < range_low and allow_shorts:
                    position = 'Short'
                    entry_price = close
                    entry_date = date
                    direction = 'Short'
                continue
            
            # If in a position, check for flip
            if position == 'Long':
                if close < range_low:
                    # Record the long trade (exit for risk management)
                    trades.append({
                        'year': year,
                        'quarter': quarter,
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': date,
                        'exit_price': close,
                        'direction': 'Long',
                        'pct_return': ((close - entry_price) / entry_price) * 100,
                        'exit_reason': 'Risk Management Exit' if not allow_shorts else 'Flip to Short',
                        'range_high': range_high,
                        'range_low': range_low
                    })
                    
                    # Only start new short position if shorts are allowed
                    if allow_shorts:
                        position = 'Short'
                        entry_price = close
                        entry_date = date
                        direction = 'Short'
                    else:
                        # Exit position and wait for next opportunity
                        position = None
                        entry_price = None
                        entry_date = None
                        direction = None
            elif position == 'Short':
                if close > range_high:
                    # Record the short trade
                    trades.append({
                        'year': year,
                        'quarter': quarter,
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': date,
                        'exit_price': close,
                        'direction': 'Short',
                        'pct_return': ((entry_price - close) / entry_price) * 100,
                        'exit_reason': 'Flip to Long',
                        'range_high': range_high,
                        'range_low': range_low
                    })
                    
                    # Start new long position
                    position = 'Long'
                    entry_price = close
                    entry_date = date
                    direction = 'Long'
        
        # Record the final position at quarter expiration
        if position is not None:
            if direction == 'Long':
                pct_return = ((quarter_expiration_price - entry_price) / entry_price) * 100
            else:  # Short
                pct_return = ((entry_price - quarter_expiration_price) / entry_price) * 100
            
            trades.append({
                'year': year,
                'quarter': quarter,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': quarter_expiration_date,
                'exit_price': quarter_expiration_price,
                'direction': direction,
                'pct_return': pct_return,
                'exit_reason': 'Quarter Expiration',
                'range_high': range_high,
                'range_low': range_low
            })
        
        return trades

    def run_backtest(self):
        """Run quarterly expiration strategy."""
        print("Running quarterly expiration strategy (flip on breakouts, exit at expiration)...")
        if self.results is None:
            self.load_data()
        
        # Run quarterly expiration strategy
        self.flipping_trades = []
        for _, row in self.results.iterrows():
            year = row['year']
            quarter = row['quarter']
            trades = self.simulate_flipping_strategy_for_quarter(year, quarter)
            self.flipping_trades.extend(trades)
        
        self.flipping_trades_df = pd.DataFrame(self.flipping_trades)
        # Filter for trades from start_year onwards
        filter_date = pd.to_datetime(f'{self.start_year}-01-01')
        self.flipping_trades_df = self.flipping_trades_df[self.flipping_trades_df['entry_date'] >= filter_date].reset_index(drop=True)
        self.calculate_flipping_performance_metrics()
        
        print(f"Quarterly expiration strategy completed. {len(self.flipping_trades_df)} trades simulated.")
        
        return self.flipping_trades_df
    
    def calculate_flipping_performance_metrics(self):
        """Calculate comprehensive performance metrics for flipping strategy."""
        if self.flipping_trades_df is None or len(self.flipping_trades_df) == 0:
            print("No flipping trades to analyze")
            return
        
        trades = self.flipping_trades_df.copy()
        
        # Compounded equity curve
        trades['equity'] = (1 + trades['pct_return'] / 100).cumprod()
        self.flipping_equity_curve = trades['equity']
        
        # Long-only equity curve
        long_trades = trades[trades['direction'] == 'Long'].copy()
        if len(long_trades) > 0:
            long_trades['long_equity'] = (1 + long_trades['pct_return'] / 100).cumprod()
            self.long_only_equity_curve = long_trades['long_equity']
            self.long_only_trades_df = long_trades
        else:
            self.long_only_equity_curve = None
            self.long_only_trades_df = None
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(trades[trades['pct_return'] > 0])
        losing_trades = len(trades[trades['pct_return'] < 0])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Return metrics
        total_return = (trades['equity'].iloc[-1] - 1) * 100 if total_trades > 0 else 0
        avg_return = trades['pct_return'].mean()
        median_return = trades['pct_return'].median()
        
        # Risk metrics
        max_drawdown = self.calculate_max_drawdown(trades['equity'])
        std_return = trades['pct_return'].std()
        
        # Calculate annualized metrics for Sharpe ratio using daily returns
        start_date = trades['entry_date'].min()
        end_date = trades['exit_date'].max()
        years_covered = (end_date - start_date).days / 365.25
        
        # Create daily equity curve for proper Sharpe ratio calculation
        daily_equity = self.create_daily_equity_curve(trades, start_date, end_date)
        
        # Calculate daily returns (including zero-return days between trades)
        daily_returns = daily_equity.pct_change().dropna()
        
        # Annualized return from CAGR
        if years_covered > 0:
            annualized_return = ((trades['equity'].iloc[-1]) ** (1/years_covered) - 1) * 100
        else:
            annualized_return = 0
        
        # Annualized volatility from daily returns (standard approach)
        # This includes both trading days and flat days between trades
        daily_volatility = daily_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252) * 100  # 252 trading days per year
        
        # Risk-free rate assumption (using 3% as a reasonable historical average)
        risk_free_rate = 3.0
        
        # Sharpe ratio using annualized metrics
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        # Long-only metrics
        long_total_trades = len(long_trades) if len(long_trades) > 0 else 0
        long_winning_trades = len(long_trades[long_trades['pct_return'] > 0]) if len(long_trades) > 0 else 0
        long_losing_trades = len(long_trades[long_trades['pct_return'] < 0]) if len(long_trades) > 0 else 0
        long_win_rate = (long_winning_trades / long_total_trades) * 100 if long_total_trades > 0 else 0
        long_total_return = (long_trades['long_equity'].iloc[-1] - 1) * 100 if len(long_trades) > 0 else 0
        long_avg_return = long_trades['pct_return'].mean() if len(long_trades) > 0 else 0
        long_median_return = long_trades['pct_return'].median() if len(long_trades) > 0 else 0
        long_max_drawdown = self.calculate_max_drawdown(long_trades['long_equity']) if len(long_trades) > 0 else 0
        long_std_return = long_trades['pct_return'].std() if len(long_trades) > 0 else 0
        
        # Long-only annualized metrics using daily returns
        if len(long_trades) > 0 and years_covered > 0:
            long_annualized_return = ((long_trades['long_equity'].iloc[-1]) ** (1/years_covered) - 1) * 100
            
            # Create daily equity curve for long-only strategy
            long_start_date = long_trades['entry_date'].min()
            long_end_date = long_trades['exit_date'].max()
            long_daily_equity = self.create_daily_equity_curve(long_trades, long_start_date, long_end_date)
            
            # Calculate daily returns for long-only (including flat days)
            long_daily_returns = long_daily_equity.pct_change().dropna()
            long_daily_volatility = long_daily_returns.std()
            long_annualized_volatility = long_daily_volatility * np.sqrt(252) * 100
            
            long_sharpe_ratio = (long_annualized_return - risk_free_rate) / long_annualized_volatility if long_annualized_volatility > 0 else 0
        else:
            long_annualized_return = 0
            long_annualized_volatility = 0
            long_sharpe_ratio = 0
        
        # Store metrics
        self.flipping_performance_metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_return': avg_return,
            'median_return': median_return,
            'max_drawdown': max_drawdown,
            'std_return': std_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'years_covered': years_covered,
            'long_total_trades': long_total_trades,
            'long_winning_trades': long_winning_trades,
            'long_losing_trades': long_losing_trades,
            'long_win_rate': long_win_rate,
            'long_total_return': long_total_return,
            'long_avg_return': long_avg_return,
            'long_median_return': long_median_return,
            'long_max_drawdown': long_max_drawdown,
            'long_std_return': long_std_return,
            'long_annualized_return': long_annualized_return,
            'long_annualized_volatility': long_annualized_volatility,
            'long_sharpe_ratio': long_sharpe_ratio
        }
    
    def create_daily_equity_curve(self, trades, start_date, end_date):
        """Create a daily equity curve from trades."""
        # Get all trading days in the period
        trading_days = self.daily_df[
            (self.daily_df['Date'] >= start_date) & 
            (self.daily_df['Date'] <= end_date)
        ]['Date'].values
        
        # Initialize daily equity series with NaN
        daily_equity = pd.Series(index=pd.DatetimeIndex(trading_days), data=np.nan)
        
        # Track which trade is active on each day
        current_equity = 1.0
        
        for _, trade in trades.iterrows():
            entry_date = trade['entry_date']
            exit_date = trade['exit_date']
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            direction = trade['direction']
            
            # Get daily prices during the trade
            trade_days = self.daily_df[
                (self.daily_df['Date'] >= entry_date) & 
                (self.daily_df['Date'] <= exit_date)
            ].copy()
            
            # Equity at the start of this trade (from previous trade)
            start_equity = current_equity
            
            for _, day in trade_days.iterrows():
                date = day['Date']
                close_price = day['Close']
                
                # Calculate the unrealized return from entry to this day
                if direction == 'Long':
                    trade_return = (close_price - entry_price) / entry_price
                else:  # Short
                    trade_return = (entry_price - close_price) / entry_price
                
                # Update daily equity with mark-to-market
                daily_equity.loc[date] = start_equity * (1 + trade_return)
            
            # Update current_equity to the realized exit value
            final_return = (exit_price - entry_price) / entry_price if direction == 'Long' else (entry_price - exit_price) / entry_price
            current_equity = start_equity * (1 + final_return)
        
        # Forward fill for days between trades (flat equity during no position)
        daily_equity = daily_equity.sort_index()
        daily_equity = daily_equity.ffill().fillna(1.0)
        
        return daily_equity
    
    def calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown from a series of returns."""
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max * 100
        return drawdown.min()
    
    def print_performance_summary(self):
        """Print a comprehensive performance summary for quarterly expiration strategy."""
        print("\n" + "="*80)
        print("QUARTERLY EXPIRATION STRATEGY - PERFORMANCE SUMMARY")
        print("="*80)
        
        # Flipping strategy results
        if self.flipping_performance_metrics:
            metrics = self.flipping_performance_metrics
            print(f"\nOVERALL STRATEGY (flip on breakouts, exit at expiration):")
            print(f"Total Trades: {metrics['total_trades']}")
            print(f"Winning Trades: {metrics['winning_trades']}")
            print(f"Losing Trades: {metrics['losing_trades']}")
            print(f"Win Rate: {metrics['win_rate']:.2f}%")
            print(f"Total Compounded Return: {metrics['total_return']:.2f}%")
            print(f"Annualized Return: {metrics['annualized_return']:.2f}%")
            print(f"Average Return per Trade: {metrics['avg_return']:.2f}%")
            print(f"Median Return per Trade: {metrics['median_return']:.2f}%")
            print(f"Annualized Volatility: {metrics['annualized_volatility']:.2f}%")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
            print(f"Time Period: {metrics['years_covered']:.1f} years")
            
            print(f"\nLONG-ONLY STRATEGY (only long trades):")
            print(f"Total Long Trades: {metrics['long_total_trades']}")
            print(f"Winning Long Trades: {metrics['long_winning_trades']}")
            print(f"Losing Long Trades: {metrics['long_losing_trades']}")
            print(f"Long Win Rate: {metrics['long_win_rate']:.2f}%")
            print(f"Long Total Compounded Return: {metrics['long_total_return']:.2f}%")
            print(f"Long Annualized Return: {metrics['long_annualized_return']:.2f}%")
            print(f"Long Average Return per Trade: {metrics['long_avg_return']:.2f}%")
            print(f"Long Median Return per Trade: {metrics['long_median_return']:.2f}%")
            print(f"Long Annualized Volatility: {metrics['long_annualized_volatility']:.2f}%")
            print(f"Long Sharpe Ratio: {metrics['long_sharpe_ratio']:.2f}")
            print(f"Long Maximum Drawdown: {metrics['long_max_drawdown']:.2f}%")
        
        print("="*80)
    
    def plot_performance(self):
        """Create performance visualization plots for quarterly expiration strategy."""
        if self.flipping_trades_df is None or len(self.flipping_trades_df) == 0:
            print("No quarterly expiration trades to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Plot equity curves
        if self.flipping_equity_curve is not None:
            ax1.plot(self.flipping_trades_df['exit_date'], self.flipping_equity_curve, 
                    label='Overall Strategy', linewidth=2, color='blue')
        
        if self.long_only_equity_curve is not None:
            ax1.plot(self.long_only_trades_df['exit_date'], self.long_only_equity_curve, 
                    label='Long-Only Strategy', linewidth=2, color='green')
        
        ax1.set_title('Compounded Equity Curves - Quarterly Expiration Strategy')
        ax1.set_xlabel('Exit Date')
        ax1.set_ylabel('Equity (Start=1.0)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot drawdowns
        if self.flipping_equity_curve is not None:
            running_max = self.flipping_equity_curve.cummax()
            drawdown = (self.flipping_equity_curve - running_max) / running_max * 100
            ax2.plot(self.flipping_trades_df['exit_date'], drawdown, 
                    label='Overall Strategy Drawdown', linewidth=2, color='blue')
        
        if self.long_only_equity_curve is not None:
            long_running_max = self.long_only_equity_curve.cummax()
            long_drawdown = (self.long_only_equity_curve - long_running_max) / long_running_max * 100
            ax2.plot(self.long_only_trades_df['exit_date'], long_drawdown, 
                    label='Long-Only Strategy Drawdown', linewidth=2, color='green')
        
        ax2.set_title('Drawdown Analysis')
        ax2.set_xlabel('Exit Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quarterly_expiration_strategy_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filename_prefix='quarterly_expiration_strategy'):
        """Save backtest results to CSV files."""
        if self.flipping_trades_df is not None:
            self.flipping_trades_df.to_csv(f'{filename_prefix}_results.csv', index=False)
            print(f"Quarterly expiration strategy results saved to {filename_prefix}_results.csv")
        
        if self.long_only_trades_df is not None:
            self.long_only_trades_df.to_csv(f'{filename_prefix}_long_only_results.csv', index=False)
            print(f"Long-only strategy results saved to {filename_prefix}_long_only_results.csv")
        
        # Save performance metrics
        if self.flipping_performance_metrics:
            metrics_df = pd.DataFrame([self.flipping_performance_metrics])
            metrics_df.to_csv(f'{filename_prefix}_metrics.csv', index=False)
            print(f"Quarterly expiration strategy metrics saved to {filename_prefix}_metrics.csv")

def main():
    """Run the complete backtest analysis."""
    # Initialize backtest
    backtest = QuarterlyRangeBacktest(
        data_file='SPX_Quarterly_Analysis_Results.csv',
        daily_file='SPX_1D.csv',
        start_year=1970,
        end_year=2024
    )
    
    # Load data
    backtest.load_data()
    
    # Run backtest
    flipping_results = backtest.run_backtest()
    
    # Print performance summary
    backtest.print_performance_summary()
    
    # Create performance plots
    backtest.plot_performance()
    
    # Save results
    backtest.save_results()
    
    return backtest

if __name__ == "__main__":
    backtest = main() 