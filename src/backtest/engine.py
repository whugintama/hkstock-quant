import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os

from src.utils.logger import get_logger
from src.config.config import backtest_config, LOGS_DIR
from src.data.data_fetcher import HKStockDataFetcher, DataProcessor

class BacktestEngine:
    """
    Backtesting engine for trading strategies
    """
    def __init__(self, strategy, start_date, end_date, symbols, initial_capital=None, 
                 commission=None, slippage=None, data_source='auto'):
        """
        Initialize the backtesting engine
        
        Args:
            strategy: Strategy class (not instance)
            start_date (str): Start date in format YYYY-MM-DD
            end_date (str): End date in format YYYY-MM-DD
            symbols (list): List of symbols to backtest
            initial_capital (float): Initial capital
            commission (float): Commission rate
            slippage (float): Slippage rate
            data_source (str): Data source ('yfinance', 'akshare', 'auto')
        """
        self.logger = get_logger("BacktestEngine")
        self.strategy_class = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.initial_capital = initial_capital or backtest_config.INITIAL_CAPITAL
        self.commission = commission or backtest_config.COMMISSION_RATE
        self.slippage = slippage or backtest_config.SLIPPAGE
        self.data_source = data_source
        
        # Initialize components
        self.data_fetcher = HKStockDataFetcher()
        self.data_processor = DataProcessor()
        
        # Initialize data containers
        self.data = {}
        self.results = {}
        self.portfolio_history = None
        self.trades = []
        self.metrics = {}
        
        self.logger.info(f"Initialized backtesting engine for {len(self.symbols)} symbols "
                        f"from {start_date} to {end_date}")
    
    def fetch_data(self):
        """
        Fetch historical data for all symbols
        
        Returns:
            dict: Dictionary of dataframes for each symbol
        """
        self.logger.info("Fetching historical data")
        
        for symbol in self.symbols:
            # Fetch raw data
            df = self.data_fetcher.get_data(
                symbol=symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                source=self.data_source
            )
            
            # Process data
            df = self.data_processor.add_technical_indicators(df)
            
            # Store data
            self.data[symbol] = df
            
            self.logger.info(f"Fetched {len(df)} data points for {symbol}")
        
        return self.data
    
    def run(self, strategy_params=None):
        """
        Run the backtest
        
        Args:
            strategy_params (dict): Strategy parameters
            
        Returns:
            dict: Backtest results
        """
        # Fetch data if not already fetched
        if not self.data:
            self.fetch_data()
        
        # Initialize strategy
        self.strategy = self.strategy_class(self.symbols, strategy_params)
        self.strategy.set_data(self.data)
        
        # Generate signals
        signals = self.strategy.generate_signals()
        
        # Run simulation
        self._run_simulation(signals)
        
        # Calculate performance metrics
        self._calculate_metrics()
        
        # Log results
        self._log_results()
        
        return self.results
    
    def _run_simulation(self, signals):
        """
        Run the trading simulation
        
        Args:
            signals (dict): Dictionary of signal dataframes for each symbol
        """
        self.logger.info("Running trading simulation")
        
        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'positions': {symbol: 0 for symbol in self.symbols},
            'equity': self.initial_capital,
            'timestamp': signals[self.symbols[0]].index[0]
        }
        
        # Initialize portfolio history
        portfolio_history = []
        
        # Track trades
        trades = []
        
        # Iterate through each time step
        dates = signals[self.symbols[0]].index
        for date in dates:
            # Update portfolio value
            portfolio_value = portfolio['cash']
            for symbol in self.symbols:
                if date in signals[symbol].index:
                    price = signals[symbol].loc[date, 'Close']
                    portfolio_value += portfolio['positions'][symbol] * price
            
            # Record portfolio state
            portfolio_snapshot = {
                'timestamp': date,
                'cash': portfolio['cash'],
                'positions': portfolio['positions'].copy(),
                'equity': portfolio_value
            }
            portfolio_history.append(portfolio_snapshot)
            
            # Check for signals and execute trades
            for symbol in self.symbols:
                if date in signals[symbol].index:
                    signal_row = signals[symbol].loc[date]
                    
                    # Check for buy/sell signals
                    if 'Position' in signal_row and signal_row['Position'] != 0:
                        # Determine trade direction and size
                        if signal_row['Position'] > 0:  # Buy signal
                            # Calculate position size
                            price = signal_row['Close']
                            available_capital = portfolio['cash'] * 0.95  # Keep some cash reserve
                            
                            # Apply slippage to buy price
                            execution_price = price * (1 + self.slippage)
                            
                            # Calculate shares to buy
                            max_shares = int(available_capital / execution_price)
                            shares_to_buy = min(max_shares, 100)  # Limit position size
                            
                            if shares_to_buy > 0:
                                # Calculate transaction cost
                                transaction_cost = execution_price * shares_to_buy * self.commission
                                
                                # Update portfolio
                                portfolio['cash'] -= (execution_price * shares_to_buy + transaction_cost)
                                portfolio['positions'][symbol] += shares_to_buy
                                
                                # Record trade
                                trade = {
                                    'timestamp': date,
                                    'symbol': symbol,
                                    'type': 'buy',
                                    'price': execution_price,
                                    'shares': shares_to_buy,
                                    'cost': execution_price * shares_to_buy,
                                    'commission': transaction_cost
                                }
                                trades.append(trade)
                                
                                self.logger.info(f"Buy {shares_to_buy} shares of {symbol} at {execution_price}")
                        
                        elif signal_row['Position'] < 0:  # Sell signal
                            if portfolio['positions'][symbol] > 0:
                                # Apply slippage to sell price
                                price = signal_row['Close']
                                execution_price = price * (1 - self.slippage)
                                
                                # Sell all shares
                                shares_to_sell = portfolio['positions'][symbol]
                                
                                # Calculate transaction cost
                                transaction_cost = execution_price * shares_to_sell * self.commission
                                
                                # Update portfolio
                                portfolio['cash'] += (execution_price * shares_to_sell - transaction_cost)
                                portfolio['positions'][symbol] = 0
                                
                                # Record trade
                                trade = {
                                    'timestamp': date,
                                    'symbol': symbol,
                                    'type': 'sell',
                                    'price': execution_price,
                                    'shares': shares_to_sell,
                                    'cost': execution_price * shares_to_sell,
                                    'commission': transaction_cost
                                }
                                trades.append(trade)
                                
                                self.logger.info(f"Sell {shares_to_sell} shares of {symbol} at {execution_price}")
        
        # Store results
        self.portfolio_history = pd.DataFrame(portfolio_history)
        self.trades = pd.DataFrame(trades)
        
        # Convert timestamp to datetime if it's not already
        if self.portfolio_history is not None and 'timestamp' in self.portfolio_history.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.portfolio_history['timestamp']):
                self.portfolio_history['timestamp'] = pd.to_datetime(self.portfolio_history['timestamp'])
        
        if self.trades is not None and 'timestamp' in self.trades.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.trades['timestamp']):
                self.trades['timestamp'] = pd.to_datetime(self.trades['timestamp'])
    
    def _calculate_metrics(self):
        """Calculate performance metrics"""
        self.logger.info("Calculating performance metrics")
        
        if self.portfolio_history is None or len(self.portfolio_history) == 0:
            self.logger.warning("No portfolio history available")
            return
        
        # Calculate returns
        self.portfolio_history['returns'] = self.portfolio_history['equity'].pct_change()
        
        # Calculate metrics
        total_days = (self.portfolio_history['timestamp'].iloc[-1] - self.portfolio_history['timestamp'].iloc[0]).days
        years = total_days / 365.25
        
        # Total return
        initial_equity = self.portfolio_history['equity'].iloc[0]
        final_equity = self.portfolio_history['equity'].iloc[-1]
        total_return = (final_equity / initial_equity) - 1
        
        # Annualized return
        annual_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1
        
        # Volatility
        daily_returns = self.portfolio_history['returns'].dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_rate = backtest_config.RISK_FREE_RATE
        sharpe_ratio = (annual_return - risk_free_rate) / max(volatility, 0.0001)
        
        # Maximum drawdown
        cumulative_returns = (1 + self.portfolio_history['returns']).cumprod()
        drawdown = 1 - cumulative_returns / cumulative_returns.cummax()
        max_drawdown = drawdown.max()
        
        # Win rate
        if len(self.trades) > 0:
            # Calculate profit/loss for each trade
            self.trades['profit'] = 0
            
            # Group trades by symbol and calculate profit
            for symbol in self.symbols:
                symbol_trades = self.trades[self.trades['symbol'] == symbol].copy()
                
                if len(symbol_trades) > 0:
                    # Sort trades by timestamp
                    symbol_trades = symbol_trades.sort_values('timestamp')
                    
                    # Calculate profit for each trade
                    buy_price = None
                    buy_shares = 0
                    
                    for idx, trade in symbol_trades.iterrows():
                        if trade['type'] == 'buy':
                            buy_price = trade['price']
                            buy_shares = trade['shares']
                        elif trade['type'] == 'sell' and buy_price is not None:
                            sell_price = trade['price']
                            sell_shares = trade['shares']
                            profit = (sell_price - buy_price) * sell_shares
                            self.trades.loc[idx, 'profit'] = profit
                            buy_price = None
                            buy_shares = 0
            
            # Calculate win rate
            winning_trades = len(self.trades[self.trades['profit'] > 0])
            total_trades = len(self.trades[self.trades['profit'] != 0])
            win_rate = winning_trades / max(total_trades, 1)
            
            # Calculate profit factor
            gross_profit = self.trades[self.trades['profit'] > 0]['profit'].sum()
            gross_loss = abs(self.trades[self.trades['profit'] < 0]['profit'].sum())
            profit_factor = gross_profit / max(gross_loss, 0.0001)
        else:
            win_rate = 0
            profit_factor = 0
        
        # Store metrics
        self.metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'final_equity': final_equity
        }
        
        # Store results
        self.results = {
            'metrics': self.metrics,
            'portfolio_history': self.portfolio_history,
            'trades': self.trades
        }
    
    def _log_results(self):
        """Log backtest results"""
        self.logger.info("Backtest results:")
        self.logger.info(f"Total Return: {self.metrics['total_return']:.2%}")
        self.logger.info(f"Annual Return: {self.metrics['annual_return']:.2%}")
        self.logger.info(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        self.logger.info(f"Max Drawdown: {self.metrics['max_drawdown']:.2%}")
        self.logger.info(f"Win Rate: {self.metrics['win_rate']:.2%}")
        self.logger.info(f"Profit Factor: {self.metrics['profit_factor']:.2f}")
        self.logger.info(f"Total Trades: {self.metrics['total_trades']}")
    
    def save_results(self, filename=None):
        """
        Save backtest results to file
        
        Args:
            filename (str): Filename to save results
            
        Returns:
            str: Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = self.strategy.__class__.__name__
            filename = f"backtest_{strategy_name}_{timestamp}.json"
        
        # Create logs directory if it doesn't exist
        Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_path = os.path.join(LOGS_DIR, filename)
        
        # Convert DataFrames to JSON-serializable format
        results = {
            'metrics': self.metrics,
            'portfolio_history': self.portfolio_history.to_dict(orient='records') if self.portfolio_history is not None else None,
            'trades': self.trades.to_dict(orient='records') if self.trades is not None else None,
            'parameters': {
                'strategy': self.strategy.__class__.__name__,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'symbols': self.symbols,
                'initial_capital': self.initial_capital,
                'commission': self.commission,
                'slippage': self.slippage,
                'strategy_params': self.strategy.get_parameters()
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        self.logger.info(f"Results saved to {results_path}")
        return results_path
    
    def visualize(self, show_trades=True, save_path=None):
        """
        Visualize backtest results
        
        Args:
            show_trades (bool): Whether to show trades on the chart
            save_path (str): Path to save the visualization
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.portfolio_history is None or len(self.portfolio_history) == 0:
            self.logger.warning("No portfolio history available for visualization")
            return None
        
        # Set up the figure
        plt.figure(figsize=(14, 10))
        
        # Set style
        sns.set_style('whitegrid')
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(self.portfolio_history['timestamp'], self.portfolio_history['equity'], label='Portfolio Value')
        
        # Plot buy/sell signals if requested
        if show_trades and len(self.trades) > 0:
            # Plot buy signals
            buy_trades = self.trades[self.trades['type'] == 'buy']
            if len(buy_trades) > 0:
                plt.scatter(buy_trades['timestamp'], buy_trades['price'], 
                           marker='^', color='green', s=100, label='Buy')
            
            # Plot sell signals
            sell_trades = self.trades[self.trades['type'] == 'sell']
            if len(sell_trades) > 0:
                plt.scatter(sell_trades['timestamp'], sell_trades['price'], 
                           marker='v', color='red', s=100, label='Sell')
        
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        cumulative_returns = (1 + self.portfolio_history['returns'].fillna(0)).cumprod()
        drawdown = 1 - cumulative_returns / cumulative_returns.cummax()
        plt.fill_between(self.portfolio_history['timestamp'], drawdown, 0, color='red', alpha=0.3)
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        
        # Add performance metrics as text
        metrics_text = (
            f"Total Return: {self.metrics['total_return']:.2%}\n"
            f"Annual Return: {self.metrics['annual_return']:.2%}\n"
            f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {self.metrics['max_drawdown']:.2%}\n"
            f"Win Rate: {self.metrics['win_rate']:.2%}\n"
            f"Profit Factor: {self.metrics['profit_factor']:.2f}\n"
            f"Total Trades: {self.metrics['total_trades']}"
        )
        
        plt.figtext(0.01, 0.01, metrics_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Visualization saved to {save_path}")
        
        return plt.gcf()
    
    def optimize_strategy(self, param_grid, target_metric='sharpe_ratio'):
        """
        Optimize strategy parameters
        
        Args:
            param_grid (dict): Dictionary of parameter ranges for grid search
            target_metric (str): Metric to optimize
            
        Returns:
            dict: Optimized parameters and results
        """
        self.logger.info(f"Optimizing strategy parameters for {target_metric}")
        
        # Fetch data if not already fetched
        if not self.data:
            self.fetch_data()
        
        # Initialize strategy
        strategy = self.strategy_class(self.symbols)
        
        # Optimize parameters
        best_params = strategy.optimize_parameters(
            data=self.data,
            target_metric=target_metric,
            parameter_grid=param_grid
        )
        
        # Run backtest with optimized parameters
        self.run(strategy_params=best_params)
        
        # Return optimized parameters and results
        optimization_results = {
            'best_parameters': best_params,
            'metrics': self.metrics
        }
        
        self.logger.info(f"Optimization results: {optimization_results}")
        return optimization_results
