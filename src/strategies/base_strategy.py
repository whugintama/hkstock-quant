import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from src.utils.logger import get_logger

logger = get_logger("BaseStrategy")

class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    
    This abstract class defines the interface that all strategies must implement.
    It provides common functionality for strategy initialization, parameter management,
    and signal generation.
    """
    
    def __init__(self, symbols, parameters=None):
        """
        Initialize the strategy.
        
        Args:
            symbols (list): List of symbols to trade
            parameters (dict): Strategy parameters
        """
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.parameters = parameters or {}
        self.data = {}
        self.signals = {}
        self.positions = {}
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize positions
        for symbol in self.symbols:
            self.positions[symbol] = 0
    
    def set_data(self, data):
        """
        Set the data for the strategy.
        
        Args:
            data (dict): Dictionary of dataframes for each symbol
        """
        self.data = data
        self.logger.info(f"Data set for {len(data)} symbols")
    
    @abstractmethod
    def generate_signals(self):
        """
        Generate trading signals.
        
        This method must be implemented by all strategy subclasses.
        It should analyze the data and generate buy/sell signals.
        
        Returns:
            dict: Dictionary of signal dataframes for each symbol
        """
        pass
    
    def get_signals(self):
        """
        Get the generated signals.
        
        Returns:
            dict: Dictionary of signal dataframes for each symbol
        """
        return self.signals
    
    def update_parameters(self, parameters):
        """
        Update strategy parameters.
        
        Args:
            parameters (dict): New parameters
        """
        self.parameters.update(parameters)
        self.logger.info(f"Parameters updated: {parameters}")
    
    def get_parameters(self):
        """
        Get current strategy parameters.
        
        Returns:
            dict: Strategy parameters
        """
        return self.parameters
    
    def calculate_position_size(self, capital, price, risk_per_trade=0.02):
        """
        Calculate position size based on available capital and risk.
        
        Args:
            capital (float): Available capital
            price (float): Current price
            risk_per_trade (float): Risk per trade as a fraction of capital
            
        Returns:
            int: Number of shares to trade
        """
        risk_amount = capital * risk_per_trade
        position_size = int(risk_amount / price)
        return position_size
    
    def calculate_stop_loss(self, entry_price, direction, atr_multiple=2, atr_value=None):
        """
        Calculate stop loss price based on ATR.
        
        Args:
            entry_price (float): Entry price
            direction (str): 'long' or 'short'
            atr_multiple (float): Multiple of ATR
            atr_value (float): ATR value, if None will use ATR from data
            
        Returns:
            float: Stop loss price
        """
        if atr_value is None and 'ATR_14' in self.data:
            atr_value = self.data['ATR_14'].iloc[-1]
        else:
            atr_value = entry_price * 0.02  # Default to 2% if ATR not available
        
        if direction == 'long':
            stop_loss = entry_price - (atr_value * atr_multiple)
        else:  # short
            stop_loss = entry_price + (atr_value * atr_multiple)
        
        return stop_loss
    
    def __str__(self):
        """String representation of the strategy"""
        return f"{self.__class__.__name__}(symbols={self.symbols}, parameters={self.parameters})"
