#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script for backtesting a strategy on Hong Kong stocks.
This script demonstrates how to use the backtesting engine with a moving average strategy.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.engine import BacktestEngine
from src.strategies.moving_average import MovingAverageStrategy
from src.data.data_fetcher import HKStockDataFetcher
from src.visualization.visualizer import Visualizer
from src.utils.logger import get_logger

def main():
    # Initialize logger
    logger = get_logger("BacktestExample")
    logger.info("Starting backtest example")
    
    # Define backtest parameters
    symbols = ['0700.HK', '9988.HK', '0388.HK']  # Tencent, Alibaba, HKEX
    start_date = '2022-01-01'
    end_date = '2023-01-01'
    initial_capital = 100000
    
    # Define strategy parameters
    strategy_params = {
        'short_window': 20,
        'long_window': 50,
        'ma_type': 'EMA'
    }
    
    # Initialize backtest engine
    engine = BacktestEngine(
        strategy=MovingAverageStrategy,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        initial_capital=initial_capital,
        data_source='auto'
    )
    
    # Run backtest
    logger.info("Running backtest")
    results = engine.run(strategy_params=strategy_params)
    
    # Print results
    logger.info("Backtest completed")
    logger.info(f"Total Return: {results['metrics']['total_return']:.2%}")
    logger.info(f"Annual Return: {results['metrics']['annual_return']:.2%}")
    logger.info(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
    logger.info(f"Win Rate: {results['metrics']['win_rate']:.2%}")
    logger.info(f"Profit Factor: {results['metrics']['profit_factor']:.2f}")
    logger.info(f"Total Trades: {results['metrics']['total_trades']}")
    
    # Visualize results
    logger.info("Visualizing results")
    visualizer = Visualizer()
    
    # Create interactive equity curve
    equity_curve_path = os.path.join('logs', 'visualizations', 'equity_curve.html')
    visualizer.plot_equity_curve(
        portfolio_history=results['portfolio_history'],
        trades=results['trades'],
        title="Moving Average Strategy Backtest",
        show_trades=True,
        interactive=True,
        save_path=equity_curve_path
    )
    
    # Create performance metrics chart
    metrics_path = os.path.join('logs', 'visualizations', 'performance_metrics.html')
    visualizer.plot_performance_metrics(
        metrics=results['metrics'],
        title="Performance Metrics",
        save_path=metrics_path
    )
    
    # Create trade analysis chart
    if len(results['trades']) > 0:
        trades_path = os.path.join('logs', 'visualizations', 'trade_analysis.html')
        visualizer.plot_trade_analysis(
            trades=results['trades'],
            title="Trade Analysis",
            save_path=trades_path
        )
    
    # Create comprehensive dashboard
    dashboard_path = os.path.join('logs', 'visualizations', 'dashboard.html')
    visualizer.create_dashboard(
        backtest_results=results,
        title="Moving Average Strategy Dashboard",
        save_path=dashboard_path
    )
    
    # Save backtest results
    results_path = engine.save_results()
    
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Visualizations saved to logs/visualizations/")
    
    # Optimize strategy parameters
    logger.info("Optimizing strategy parameters")
    param_grid = {
        'short_window': range(10, 30, 5),
        'long_window': range(40, 100, 10),
        'ma_type': ['SMA', 'EMA']
    }
    
    optimization_results = engine.optimize_strategy(
        param_grid=param_grid,
        target_metric='sharpe_ratio'
    )
    
    logger.info("Optimization completed")
    logger.info(f"Best parameters: {optimization_results['best_parameters']}")
    logger.info(f"Sharpe Ratio with optimized parameters: {optimization_results['metrics']['sharpe_ratio']:.2f}")
    
    # Run backtest with optimized parameters
    logger.info("Running backtest with optimized parameters")
    optimized_results = engine.run(strategy_params=optimization_results['best_parameters'])
    
    # Visualize optimized results
    logger.info("Visualizing optimized results")
    optimized_dashboard_path = os.path.join('logs', 'visualizations', 'optimized_dashboard.html')
    visualizer.create_dashboard(
        backtest_results=optimized_results,
        title="Optimized Moving Average Strategy Dashboard",
        save_path=optimized_dashboard_path
    )
    
    logger.info(f"Optimized results dashboard saved to {optimized_dashboard_path}")
    
    return results

if __name__ == "__main__":
    main()
