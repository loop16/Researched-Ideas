import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from scipy.stats import norm
import math

class OptionsQuarterlyBacktest:
    """
    Options-based backtesting engine for quarterly range breakout strategy.
    
    Strategy:
    1. Wait for first breakout (long or short) from quarterly range
    2. Enter ATM/OTM/ITM option position in direction of breakout
    3. Hold until end of quarter or exit if opposite side of range is breached
    4. Track performance metrics for different moneyness levels
    """
    
    def __init__(self, data_file, daily_file='SPX_1D.csv', vix_file='CBOE_VIX3M, 1D.csv', 
                 fed_funds_file='FRED_FEDFUNDS, 1M.csv', start_year=2008, end_year=2024):
        """
        Initialize the options backtesting engine.
        
        Args:
            data_file (str): Path to the quarterly analysis results CSV
            daily_file (str): Path to the daily data CSV
            vix_file (str): Path to the VIX3M data CSV
            fed_funds_file (str): Path to the Fed Funds rate data CSV
            start_year (int): Start year for backtesting
            end_year (int): End year for backtesting
        """
        self.data_file = data_file
        self.daily_file = daily_file
        self.vix_file = vix_file
        self.fed_funds_file = fed_funds_file
        self.start_year = start_year
        self.end_year = end_year
        self.results = None
        self.daily_df = None
        self.vix_df = None
        self.fed_funds_df = None
        self.options_trades = []
        self.performance_metrics = {}
        
    def load_data(self):
        """Load and prepare all required data."""
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
        
        # Load daily data
        print("Loading daily data...")
        self.daily_df = pd.read_csv(self.daily_file)
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
        
        # Load VIX data
        print("Loading VIX3M data...")
        self.vix_df = pd.read_csv(self.vix_file)
        self.vix_df['Date'] = pd.to_datetime(self.vix_df['time'])
        self.vix_df = self.vix_df.sort_values('Date')
        print(f"Loaded {len(self.vix_df)} VIX3M data points.")
        
        # Load Fed Funds rate data
        print("Loading Fed Funds rate data...")
        self.fed_funds_df = pd.read_csv(self.fed_funds_file)
        self.fed_funds_df['Date'] = pd.to_datetime(self.fed_funds_df['time'])
        self.fed_funds_df = self.fed_funds_df.sort_values('Date')
        print(f"Loaded {len(self.fed_funds_df)} Fed Funds rate data points.")
        
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
    
    def get_quarterly_expiration_date(self, year, quarter):
        """Get the last trading day of the quarter."""
        if quarter == 1:
            month = 3
        elif quarter == 2:
            month = 6
        elif quarter == 3:
            month = 9
        else:
            month = 12
        
        # Get the last day of the month
        if month == 12:
            last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = datetime(year, month + 1, 1) - timedelta(days=1)
        
        return last_day
    
    def get_nearest_strike(self, underlying_price, moneyness='ATM', option_type='call'):
        """Get the strike price based on 2.5% ITM/OTM logic, rounded to nearest 5."""
        if moneyness == 'ATM':
            strike = underlying_price
        elif moneyness == 'ITM':
            if option_type == 'call':
                strike = underlying_price * 0.975  # 2.5% below for ITM call
            else:  # put
                strike = underlying_price * 1.025  # 2.5% above for ITM put
        elif moneyness == 'OTM':
            if option_type == 'call':
                strike = underlying_price * 1.025  # 2.5% above for OTM call
            else:  # put
                strike = underlying_price * 0.975  # 2.5% below for OTM put
        else:
            strike = underlying_price
        # Round to nearest 5
        return round(strike / 5) * 5
    
    def black_scholes(self, S, K, T, r, sigma, option_type='call', q=0.014):
        """
        Black-Scholes option pricing model.
        If T <= 0, return intrinsic value.
        """
        if T <= 0:
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:  # put
                return max(K - S, 0)
        
        d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        return price
    
    def get_market_data(self, date):
        """Get VIX and Fed Funds rate for a given date."""
        # Get VIX (use forward fill for missing values)
        vix_data = self.vix_df[self.vix_df['Date'] <= date]
        if not vix_data.empty:
            vix = vix_data.iloc[-1]['close'] / 100  # Convert to decimal
        else:
            vix = 0.20  # Default 20% volatility
        
        # Get Fed Funds rate (use forward fill for missing values)
        fed_data = self.fed_funds_df[self.fed_funds_df['Date'] <= date]
        if not fed_data.empty:
            fed_rate = fed_data.iloc[-1]['close'] / 100  # Convert to decimal
        else:
            fed_rate = 0.05  # Default 5% rate
        
        return vix, fed_rate
    
    def simulate_options_strategy_for_quarter(self, year, quarter):
        """Simulate options strategy for a quarter."""
        # Get range for the quarter
        first_friday, previous_day = self.get_quarter_range_dates(year, quarter)
        if first_friday is None or previous_day is None:
            return []
        
        range_high = max(first_friday['High'], previous_day['High'])
        range_low = min(first_friday['Low'], previous_day['Low'])
        range_mid = (range_high + range_low) / 2
        
        # Get prior range for market structure filter
        prior_range_high = None
        prior_range_low = None
        
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
        
        # Simulate options strategy with quarterly expiration
        trades = []
        position = None
        entry_price = None
        entry_date = None
        direction = None
        option_type = None
        
        # Moneyness levels to test
        moneyness_levels = ['OTM', 'ATM', 'ITM']
        
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
                    option_type = 'call'
                elif close < range_low and allow_shorts:
                    position = 'Short'
                    entry_price = close
                    entry_date = date
                    direction = 'Short'
                    option_type = 'put'
                continue
            
            # If in a position, check for flip
            if position == 'Long':
                if close < range_low:
                    # Record the long trade (exit for risk management)
                    trades.extend(self.calculate_options_pnl(
                        year, quarter, entry_date, date, entry_price, close,
                        direction, option_type, moneyness_levels, 'Risk Management Exit' if not allow_shorts else 'Flip to Short'
                    ))
                    
                    # Only start new short position if shorts are allowed
                    if allow_shorts:
                        position = 'Short'
                        entry_price = close
                        entry_date = date
                        direction = 'Short'
                        option_type = 'put'
                    else:
                        # Exit position and wait for next opportunity
                        position = None
                        entry_price = None
                        entry_date = None
                        direction = None
                        option_type = None
            elif position == 'Short':
                if close > range_high:
                    # Record the short trade
                    trades.extend(self.calculate_options_pnl(
                        year, quarter, entry_date, date, entry_price, close,
                        direction, option_type, moneyness_levels, 'Flip to Long'
                    ))
                    
                    # Start new long position
                    position = 'Long'
                    entry_price = close
                    entry_date = date
                    direction = 'Long'
                    option_type = 'call'
        
        # Record the final position at quarter expiration
        if position is not None:
            trades.extend(self.calculate_options_pnl(
                year, quarter, entry_date, quarter_expiration_date, entry_price, quarter_expiration_price,
                direction, option_type, moneyness_levels, 'Quarter Expiration'
            ))
        
        return trades
    
    def calculate_options_pnl(self, year, quarter, entry_date, exit_date, entry_underlying, 
                             exit_underlying, direction, option_type, moneyness_levels, exit_reason):
        """Calculate entry and exit option prices for each moneyness level."""
        trades = []
        
        # Calculate time to expiration
        expiration_date = self.get_quarterly_expiration_date(year, quarter)
        entry_T = (expiration_date - entry_date).days / 365.25
        exit_T = (expiration_date - exit_date).days / 365.25
        
        # Get market data
        entry_vix, entry_rate = self.get_market_data(entry_date)
        exit_vix, exit_rate = self.get_market_data(exit_date)
        
        for moneyness in moneyness_levels:
            # Determine strike price based on moneyness and option type
            strike = self.get_nearest_strike(entry_underlying, moneyness, option_type)
            
            # Price entry option
            entry_option_price = self.black_scholes(
                entry_underlying, strike, entry_T, entry_rate, entry_vix, option_type
            )
            
            # Price exit option
            exit_option_price = self.black_scholes(
                exit_underlying, strike, exit_T, exit_rate, exit_vix, option_type
            )
            
            trades.append({
                'year': year,
                'quarter': quarter,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_underlying': entry_underlying,
                'exit_underlying': exit_underlying,
                'direction': direction,
                'option_type': option_type,
                'moneyness': moneyness,
                'strike': strike,
                'entry_option_price': entry_option_price,
                'exit_option_price': exit_option_price,
                'exit_reason': exit_reason,
                'entry_vix': entry_vix * 100,
                'exit_vix': exit_vix * 100,
                'entry_rate': entry_rate * 100,
                'exit_rate': exit_rate * 100
            })
        
        return trades
    
    def run_backtest(self):
        """Run options strategy backtest."""
        print("Running options strategy backtest...")
        if self.results is None:
            self.load_data()
        
        # Run options strategy
        self.options_trades = []
        for _, row in self.results.iterrows():
            year = row['year']
            quarter = row['quarter']
            trades = self.simulate_options_strategy_for_quarter(year, quarter)
            self.options_trades.extend(trades)
        
        self.options_trades_df = pd.DataFrame(self.options_trades)
        # Filter for trades from 1990 onwards
        self.options_trades_df = self.options_trades_df[self.options_trades_df['entry_date'] >= pd.to_datetime('1990-01-01')].reset_index(drop=True)
        # Add option_pnl column
        self.options_trades_df['option_pnl'] = self.options_trades_df['exit_option_price'] - self.options_trades_df['entry_option_price']
        self.calculate_options_performance_metrics()
        
        print(f"Options strategy completed. {len(self.options_trades_df)} trades simulated.")
        
        return self.options_trades_df
    
    def calculate_options_performance_metrics(self):
        """Calculate performance metrics for options strategy."""
        if self.options_trades_df is None or len(self.options_trades_df) == 0:
            print("No options trades to analyze")
            return
        
        trades = self.options_trades_df.copy()
        
        # Calculate percentage return based on option_pnl
        trades['pct_return'] = (trades['option_pnl'] / 500) * 100
        
        # Calculate metrics by moneyness
        self.performance_metrics = {}
        
        for moneyness in ['OTM', 'ATM', 'ITM']:
            moneyness_trades = trades[trades['moneyness'] == moneyness].copy()
            
            if len(moneyness_trades) == 0:
                continue
            
            # Compounded equity curve
            moneyness_trades['equity'] = (1 + moneyness_trades['pct_return'] / 100).cumprod()
            
            # Basic metrics
            total_trades = len(moneyness_trades)
            winning_trades = len(moneyness_trades[moneyness_trades['pct_return'] > 0])
            losing_trades = len(moneyness_trades[moneyness_trades['pct_return'] < 0])
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Return metrics
            total_return = (moneyness_trades['equity'].iloc[-1] - 1) * 100 if total_trades > 0 else 0
            avg_return = moneyness_trades['pct_return'].mean()
            median_return = moneyness_trades['pct_return'].median()
            
            # Risk metrics
            max_drawdown = self.calculate_max_drawdown(moneyness_trades['equity'])
            std_return = moneyness_trades['pct_return'].std()
            
            # Annualized metrics
            start_date = moneyness_trades['entry_date'].min()
            end_date = moneyness_trades['exit_date'].max()
            years_covered = (end_date - start_date).days / 365.25
            
            if years_covered > 0:
                annualized_return = ((moneyness_trades['equity'].iloc[-1]) ** (1/years_covered) - 1) * 100
                trades_per_year = total_trades / years_covered
                annualized_volatility = std_return * (trades_per_year ** 0.5) if trades_per_year > 0 else 0
            else:
                annualized_return = 0
                annualized_volatility = 0
            
            # Sharpe ratio
            risk_free_rate = 3.0
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
            
            self.performance_metrics[moneyness] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'avg_return': avg_return,
                'median_return': median_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'years_covered': years_covered
            }
    
    def calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown from a series of returns."""
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max * 100
        return drawdown.min()
    
    def print_performance_summary(self):
        """Print performance summary for options strategy."""
        print("\n" + "="*80)
        print("OPTIONS QUARTERLY RANGE BREAKOUT STRATEGY - PERFORMANCE SUMMARY")
        print("="*80)
        
        for moneyness, metrics in self.performance_metrics.items():
            print(f"\n{moneyness} OPTIONS:")
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
        
        print("="*80)
    
    def plot_performance(self):
        """Create performance visualization plots for options strategy."""
        if self.options_trades_df is None or len(self.options_trades_df) == 0:
            print("No options trades to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Plot cumulative PnL by moneyness
        for moneyness in ['OTM', 'ATM', 'ITM']:
            moneyness_trades = self.options_trades_df[self.options_trades_df['moneyness'] == moneyness].copy()
            if len(moneyness_trades) > 0:
                # Sort by exit date and calculate cumulative PnL
                moneyness_trades = moneyness_trades.sort_values('exit_date')
                moneyness_trades['cumulative_pnl'] = moneyness_trades['option_pnl'].cumsum()
                ax1.plot(moneyness_trades['exit_date'], moneyness_trades['cumulative_pnl'], 
                        label=f'{moneyness} Options', linewidth=2)
        
        ax1.set_title('Cumulative PnL by Moneyness ($500 Risk per Trade)')
        ax1.set_xlabel('Exit Date')
        ax1.set_ylabel('Cumulative PnL ($)')
        ax1.legend()
        ax1.grid(True)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot individual trade PnL scatter plot
        for moneyness in ['OTM', 'ATM', 'ITM']:
            moneyness_trades = self.options_trades_df[self.options_trades_df['moneyness'] == moneyness]
            if len(moneyness_trades) > 0:
                ax2.scatter(moneyness_trades['exit_date'], moneyness_trades['option_pnl'], 
                           label=f'{moneyness} Options', alpha=0.6, s=20)
        
        ax2.set_title('Individual Trade PnL by Moneyness ($500 Risk per Trade)')
        ax2.set_xlabel('Exit Date')
        ax2.set_ylabel('Trade PnL ($)')
        ax2.legend()
        ax2.grid(True)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('options_strategy_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filename_prefix='options_strategy'):
        """Save options backtest results to CSV files."""
        if self.options_trades_df is not None:
            self.options_trades_df.to_csv(f'{filename_prefix}_results.csv', index=False)
            print(f"Options strategy results saved to {filename_prefix}_results.csv")
        
        # Save performance metrics
        if self.performance_metrics:
            metrics_list = []
            for moneyness, metrics in self.performance_metrics.items():
                metrics['moneyness'] = moneyness
                metrics_list.append(metrics)
            
            metrics_df = pd.DataFrame(metrics_list)
            metrics_df.to_csv(f'{filename_prefix}_metrics.csv', index=False)
            print(f"Options strategy metrics saved to {filename_prefix}_metrics.csv")

def main():
    """Run the complete options backtest analysis."""
    # Initialize options backtest
    backtest = OptionsQuarterlyBacktest(
        data_file='SPX_Quarterly_Analysis_Results.csv',
        daily_file='SPX_1D.csv',
        vix_file='CBOE_VIX3M, 1D.csv',
        fed_funds_file='FRED_FEDFUNDS, 1M.csv',
        start_year=2008,
        end_year=2024
    )
    
    # Load data
    backtest.load_data()
    
    # Run backtest
    options_results = backtest.run_backtest()
    
    # Print performance summary
    backtest.print_performance_summary()
    
    # Create performance plots
    backtest.plot_performance()
    
    # Save results
    backtest.save_results()
    
    return backtest

if __name__ == "__main__":
    backtest = main() 