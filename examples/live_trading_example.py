#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script for live trading with Hong Kong stocks using Futu API.
This script demonstrates how to use the trading system with a moving average strategy.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import time
import argparse

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.futu_trader import FutuTrader
from src.strategies.moving_average import MovingAverageStrategy
from src.utils.logger import get_logger
from src.config.config import trading_config

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Live trading example for Hong Kong stocks')
    parser.add_argument('--symbols', type=str, nargs='+', default=['0700.HK', '9988.HK'],
                        help='List of symbols to trade')
    parser.add_argument('--capital', type=float, default=100000,
                        help='Initial capital')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--paper', action='store_true',
                        help='Use paper trading')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Initialize logger
    logger = get_logger("LiveTradingExample")
    logger.info("Starting live trading example")
    
    # Define trading parameters
    symbols = args.symbols
    initial_capital = args.capital
    config_path = args.config
    
    # Define strategy parameters
    strategy_params = {
        'short_window': 20,
        'long_window': 50,
        'ma_type': 'EMA'
    }
    
    # Initialize strategy
    strategy = MovingAverageStrategy(symbols, strategy_params)
    
    # Initialize trader
    trader = FutuTrader(
        strategy=strategy,
        symbols=symbols,
        initial_capital=initial_capital,
        config_path=config_path
    )
    
    try:
        # Start trading
        logger.info("Starting trading")
        trader.start()
        
        # Display initial account information
        account_info = trader.get_account_info()
        logger.info(f"Account Information:")
        logger.info(f"  Total Assets: {account_info.get('total_assets', 0)}")
        logger.info(f"  Cash: {account_info.get('cash', 0)}")
        logger.info(f"  Market Value: {account_info.get('market_value', 0)}")
        
        # Display initial positions
        positions = trader.get_positions()
        logger.info(f"Initial Positions:")
        for symbol, quantity in positions.items():
            if quantity > 0:
                logger.info(f"  {symbol}: {quantity}")
        
        # Keep the script running
        logger.info("Trading is running. Press Ctrl+C to stop.")
        
        # Main loop
        while True:
            # Display current performance every hour
            performance = trader.get_performance()
            logger.info(f"Current Performance:")
            logger.info(f"  Portfolio Value: {performance.get('current_value', 0)}")
            logger.info(f"  Total Return: {performance.get('total_return', 0):.2%}")
            logger.info(f"  Cash: {performance.get('cash', 0)}")
            
            # Sleep for an hour
            time.sleep(3600)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping trading.")
    
    except Exception as e:
        logger.error(f"Error in trading: {e}")
    
    finally:
        # Stop trading
        logger.info("Stopping trading")
        trader.stop()
        
        # Display final performance
        performance = trader.get_performance()
        logger.info(f"Final Performance:")
        logger.info(f"  Portfolio Value: {performance.get('current_value', 0)}")
        logger.info(f"  Total Return: {performance.get('total_return', 0):.2%}")
        logger.info(f"  Cash: {performance.get('cash', 0)}")
        
        # Display final positions
        positions = trader.get_positions()
        logger.info(f"Final Positions:")
        for symbol, quantity in positions.items():
            if quantity > 0:
                logger.info(f"  {symbol}: {quantity}")
        
        logger.info("Trading stopped")

if __name__ == "__main__":
    main()
