import pandas as pd
import numpy as np
from src.strategies.base_strategy import BaseStrategy
from src.utils.logger import get_logger

class MovingAverageStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy
    
    This strategy generates buy signals when the short-term moving average crosses above
    the long-term moving average, and sell signals when the short-term moving average
    crosses below the long-term moving average.
    """
    
    def __init__(self, symbols, parameters=None):
        """
        Initialize the Moving Average Crossover strategy.
        
        Args:
            symbols (list): List of symbols to trade
            parameters (dict): Strategy parameters
        """
        default_params = {
            'short_window': 20,
            'long_window': 50,
            'ma_type': 'SMA',  # 'SMA' or 'EMA'
            'signal_threshold': 0  # Additional threshold for signal confirmation
        }
        
        # Update default parameters with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(symbols, default_params)
        self.logger = get_logger("MovingAverageStrategy")
    
    def generate_signals(self):
        """
        Generate trading signals based on moving average crossovers.
        
        Returns:
            dict: Dictionary of signal dataframes for each symbol
        """
        self.signals = {}
        
        for symbol, df in self.data.items():
            self.logger.info(f"Generating signals for {symbol}")
            
            # Create a copy of the dataframe
            signals = df.copy()
            
            # Get parameters
            short_window = self.parameters['short_window']
            long_window = self.parameters['long_window']
            ma_type = self.parameters['ma_type']
            
            # Calculate moving averages
            if ma_type == 'SMA':
                signals[f'Short_MA'] = signals['Close'].rolling(window=short_window, min_periods=1).mean()
                signals[f'Long_MA'] = signals['Close'].rolling(window=long_window, min_periods=1).mean()
            else:  # EMA
                signals[f'Short_MA'] = signals['Close'].ewm(span=short_window, adjust=False).mean()
                signals[f'Long_MA'] = signals['Close'].ewm(span=long_window, adjust=False).mean()
            
            # Initialize signal column
            signals['Signal'] = 0
            
            # Generate signals
            # 1 = Buy, -1 = Sell, 0 = Hold
            signals['Signal'] = np.where(signals['Short_MA'] > signals['Long_MA'], 1, -1)
            
            # Generate crossover signals (signal changes)
            signals['Position'] = signals['Signal'].diff()
            
            # Store the signals
            self.signals[symbol] = signals
            
            self.logger.info(f"Generated {signals['Position'].abs().sum()} signals for {symbol}")
        
        return self.signals
    
    def calculate_target_position(self, symbol, row, capital):
        """
        Calculate the target position based on the signal.
        
        Args:
            symbol (str): Symbol
            row (pd.Series): Current data row
            capital (float): Available capital
            
        Returns:
            int: Target position (positive for long, negative for short, 0 for no position)
        """
        if row['Position'] > 0:  # Buy signal
            # Calculate position size
            price = row['Close']
            position_size = self.calculate_position_size(capital, price)
            return position_size
        
        elif row['Position'] < 0:  # Sell signal
            return 0
        
        else:  # Hold
            return self.positions.get(symbol, 0)
    
    def optimize_parameters(self, data, target_metric='sharpe_ratio', parameter_grid=None):
        """
        Optimize strategy parameters using grid search.
        
        Args:
            data (dict): Dictionary of dataframes for each symbol
            target_metric (str): Metric to optimize
            parameter_grid (dict): Dictionary of parameter ranges for grid search
            
        Returns:
            dict: Optimized parameters
        """
        if parameter_grid is None:
            parameter_grid = {
                'short_window': range(5, 30, 5),
                'long_window': range(30, 100, 10),
                'ma_type': ['SMA', 'EMA']
            }
        
        best_params = {}
        best_metric_value = -np.inf if target_metric in ['sharpe_ratio', 'return'] else np.inf
        
        # Store original parameters
        original_params = self.parameters.copy()
        
        # Generate all parameter combinations
        import itertools
        param_keys = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        self.logger.info(f"Optimizing parameters with {len(param_combinations)} combinations")
        
        # Test each parameter combination
        for params in param_combinations:
            # Update parameters
            test_params = dict(zip(param_keys, params))
            self.update_parameters(test_params)
            
            # Set data and generate signals
            self.set_data(data)
            self.generate_signals()
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics()
            
            # Check if this is the best combination
            metric_value = metrics.get(target_metric)
            if metric_value is not None:
                if (target_metric in ['sharpe_ratio', 'return'] and metric_value > best_metric_value) or \
                   (target_metric not in ['sharpe_ratio', 'return'] and metric_value < best_metric_value):
                    best_metric_value = metric_value
                    best_params = test_params.copy()
        
        # Restore original parameters
        self.update_parameters(original_params)
        
        self.logger.info(f"Optimized parameters: {best_params}, {target_metric}: {best_metric_value}")
        return best_params
    
    def _calculate_performance_metrics(self):
        """
        Calculate performance metrics for the strategy.
        
        Returns:
            dict: Dictionary of performance metrics
        """
        # This is a simplified implementation
        # In a real system, this would calculate returns, Sharpe ratio, etc.
        metrics = {
            'sharpe_ratio': 0,
            'return': 0,
            'drawdown': 0
        }
        
        # Calculate metrics based on signals
        for symbol, signals in self.signals.items():
            # Calculate returns
            signals['Returns'] = signals['Close'].pct_change()
            signals['Strategy_Returns'] = signals['Signal'].shift(1) * signals['Returns']
            
            # Calculate cumulative returns
            signals['Cumulative_Returns'] = (1 + signals['Returns']).cumprod()
            signals['Strategy_Cumulative_Returns'] = (1 + signals['Strategy_Returns']).cumprod()
            
            # Calculate metrics
            total_return = signals['Strategy_Cumulative_Returns'].iloc[-1] - 1
            annual_return = total_return / (len(signals) / 252)  # Assuming 252 trading days per year
            
            # Calculate Sharpe ratio
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            excess_returns = signals['Strategy_Returns'] - risk_free_rate
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            # Calculate maximum drawdown
            cumulative_max = signals['Strategy_Cumulative_Returns'].cummax()
            drawdown = (signals['Strategy_Cumulative_Returns'] / cumulative_max) - 1
            max_drawdown = drawdown.min()
            
            # Update metrics
            metrics['return'] = annual_return
            metrics['sharpe_ratio'] = sharpe_ratio
            metrics['drawdown'] = max_drawdown
        
        return metrics
