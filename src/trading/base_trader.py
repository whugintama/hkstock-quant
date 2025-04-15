import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import time
import threading
from datetime import datetime, timedelta
import schedule
import os
import json

from src.utils.logger import get_logger
from src.config.config import trading_config, LOGS_DIR

class BaseTrader(ABC):
    """
    Base class for all traders.
    
    This abstract class defines the interface that all trader implementations must follow.
    It provides common functionality for trading execution, order management,
    and position tracking.
    """
    
    def __init__(self, strategy, symbols, initial_capital=None, config_path=None):
        """
        Initialize the trader.
        
        Args:
            strategy: Strategy instance or class
            symbols (list): List of symbols to trade
            initial_capital (float): Initial capital
            config_path (str): Path to configuration file
        """
        self.logger = get_logger(self.__class__.__name__)
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.initial_capital = initial_capital or trading_config.INITIAL_CAPITAL
        
        # Load configuration
        self.config = trading_config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                for key, value in custom_config.items():
                    setattr(self.config, key, value)
        
        # Initialize strategy
        if isinstance(strategy, type):
            self.strategy = strategy(self.symbols)
        else:
            self.strategy = strategy
        
        # Initialize data containers
        self.data = {}
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.orders = []
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        
        # Trading state
        self.is_running = False
        self.trading_thread = None
        self.last_update_time = None
        
        self.logger.info(f"Initialized {self.__class__.__name__} for {len(self.symbols)} symbols")
    
    @abstractmethod
    def connect(self):
        """
        Connect to the trading API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """
        Disconnect from the trading API.
        
        Returns:
            bool: True if disconnection is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_market_data(self, symbol, timeframe='1d', bars=100):
        """
        Get market data for a symbol.
        
        Args:
            symbol (str): Symbol
            timeframe (str): Timeframe
            bars (int): Number of bars
            
        Returns:
            pd.DataFrame: DataFrame with market data
        """
        pass
    
    @abstractmethod
    def get_account_info(self):
        """
        Get account information.
        
        Returns:
            dict: Account information
        """
        pass
    
    @abstractmethod
    def get_positions(self):
        """
        Get current positions.
        
        Returns:
            dict: Dictionary of positions
        """
        pass
    
    @abstractmethod
    def place_order(self, symbol, order_type, quantity, price=None, stop_price=None, time_in_force='day'):
        """
        Place an order.
        
        Args:
            symbol (str): Symbol
            order_type (str): Order type (market, limit, stop, stop_limit)
            quantity (int): Quantity
            price (float): Price for limit orders
            stop_price (float): Stop price for stop orders
            time_in_force (str): Time in force
            
        Returns:
            str: Order ID
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id):
        """
        Cancel an order.
        
        Args:
            order_id (str): Order ID
            
        Returns:
            bool: True if cancellation is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id):
        """
        Get order status.
        
        Args:
            order_id (str): Order ID
            
        Returns:
            dict: Order status
        """
        pass
    
    def start(self):
        """
        Start the trading process.
        
        Returns:
            bool: True if trading is started, False otherwise
        """
        if self.is_running:
            self.logger.warning("Trading is already running")
            return False
        
        # Connect to the trading API
        if not self.connect():
            self.logger.error("Failed to connect to trading API")
            return False
        
        # Set trading state
        self.is_running = True
        
        # Start trading thread
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        self.logger.info("Trading started")
        return True
    
    def stop(self):
        """
        Stop the trading process.
        
        Returns:
            bool: True if trading is stopped, False otherwise
        """
        if not self.is_running:
            self.logger.warning("Trading is not running")
            return False
        
        # Set trading state
        self.is_running = False
        
        # Wait for trading thread to stop
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=10)
        
        # Disconnect from the trading API
        self.disconnect()
        
        self.logger.info("Trading stopped")
        return True
    
    def _trading_loop(self):
        """
        Main trading loop.
        """
        self.logger.info("Trading loop started")
        
        # Schedule data update and trading
        schedule.every(1).minutes.do(self._update_data)
        schedule.every(1).minutes.do(self._check_signals)
        
        # Schedule daily tasks
        schedule.every().day.at(self.config.MARKET_OPEN_TIME).do(self._on_market_open)
        schedule.every().day.at(self.config.MARKET_CLOSE_TIME).do(self._on_market_close)
        
        # Run initial update
        self._update_data()
        
        # Main loop
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)
        
        self.logger.info("Trading loop stopped")
    
    def _update_data(self):
        """
        Update market data.
        """
        self.logger.info("Updating market data")
        
        for symbol in self.symbols:
            try:
                # Get market data
                df = self.get_market_data(symbol)
                
                # Store data
                self.data[symbol] = df
                
                self.logger.info(f"Updated market data for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to update market data for {symbol}: {e}")
        
        # Update account information
        try:
            account_info = self.get_account_info()
            self.portfolio_value = account_info.get('portfolio_value', self.portfolio_value)
            self.cash = account_info.get('cash', self.cash)
            
            self.logger.info(f"Updated account information: Portfolio value: {self.portfolio_value}, Cash: {self.cash}")
        except Exception as e:
            self.logger.error(f"Failed to update account information: {e}")
        
        # Update positions
        try:
            positions = self.get_positions()
            self.positions = positions
            
            self.logger.info(f"Updated positions: {self.positions}")
        except Exception as e:
            self.logger.error(f"Failed to update positions: {e}")
        
        # Update last update time
        self.last_update_time = datetime.now()
    
    def _check_signals(self):
        """
        Check for trading signals and execute trades.
        """
        if not self.data:
            self.logger.warning("No market data available")
            return
        
        # Update strategy with latest data
        self.strategy.set_data(self.data)
        
        # Generate signals
        signals = self.strategy.generate_signals()
        
        # Execute trades based on signals
        for symbol, signal_df in signals.items():
            if symbol not in self.symbols:
                continue
            
            # Get latest signal
            latest_signal = signal_df.iloc[-1]
            
            # Check for position change
            if 'Position' in latest_signal and latest_signal['Position'] != 0:
                # Calculate target position
                target_position = self.strategy.calculate_target_position(
                    symbol, latest_signal, self.portfolio_value
                )
                
                # Get current position
                current_position = self.positions.get(symbol, 0)
                
                # Calculate position change
                position_change = target_position - current_position
                
                if position_change != 0:
                    # Execute trade
                    self._execute_trade(symbol, position_change, latest_signal)
    
    def _execute_trade(self, symbol, quantity, signal_data):
        """
        Execute a trade.
        
        Args:
            symbol (str): Symbol
            quantity (int): Quantity (positive for buy, negative for sell)
            signal_data (pd.Series): Signal data
        """
        if quantity == 0:
            return
        
        try:
            # Determine order type
            order_type = 'market'
            price = None
            
            # Get current price
            current_price = signal_data.get('Close', 0)
            
            # Apply risk management
            if quantity > 0:  # Buy
                # Check if we have enough cash
                required_cash = current_price * quantity * (1 + self.config.COMMISSION_RATE)
                if required_cash > self.cash:
                    adjusted_quantity = int(self.cash / (current_price * (1 + self.config.COMMISSION_RATE)))
                    self.logger.warning(f"Adjusted buy quantity from {quantity} to {adjusted_quantity} due to insufficient cash")
                    quantity = adjusted_quantity
                
                # Check position size limit
                max_position_value = self.portfolio_value * self.config.MAX_POSITION_SIZE
                position_value = current_price * quantity
                if position_value > max_position_value:
                    adjusted_quantity = int(max_position_value / current_price)
                    self.logger.warning(f"Adjusted buy quantity from {quantity} to {adjusted_quantity} due to position size limit")
                    quantity = adjusted_quantity
            
            # Skip if quantity is zero after adjustments
            if quantity == 0:
                self.logger.warning(f"Skipping trade for {symbol} due to zero quantity after adjustments")
                return
            
            # Log trade intent
            action = "Buy" if quantity > 0 else "Sell"
            self.logger.info(f"{action} {abs(quantity)} shares of {symbol} at {current_price}")
            
            # Place order
            order_id = self.place_order(
                symbol=symbol,
                order_type=order_type,
                quantity=abs(quantity),
                price=price,
                time_in_force='day'
            )
            
            # Log order placement
            self.logger.info(f"Placed {order_type} order {order_id} to {action.lower()} {abs(quantity)} shares of {symbol}")
            
            # Store order
            order = {
                'order_id': order_id,
                'symbol': symbol,
                'action': action.lower(),
                'quantity': abs(quantity),
                'price': current_price,
                'order_type': order_type,
                'time': datetime.now(),
                'status': 'placed'
            }
            self.orders.append(order)
            
            # Send notification
            self._send_notification(f"{action} {abs(quantity)} shares of {symbol} at {current_price}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute trade for {symbol}: {e}")
    
    def _on_market_open(self):
        """
        Actions to perform at market open.
        """
        self.logger.info("Market open")
        
        # Update data
        self._update_data()
        
        # Check signals
        self._check_signals()
    
    def _on_market_close(self):
        """
        Actions to perform at market close.
        """
        self.logger.info("Market close")
        
        # Update data
        self._update_data()
        
        # Generate end-of-day report
        self._generate_report()
    
    def _generate_report(self):
        """
        Generate trading report.
        """
        self.logger.info("Generating trading report")
        
        # Create report directory if it doesn't exist
        report_dir = os.path.join(LOGS_DIR, "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate report filename
        report_date = datetime.now().strftime("%Y%m%d")
        report_file = os.path.join(report_dir, f"trading_report_{report_date}.json")
        
        # Prepare report data
        report = {
            'date': report_date,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions': self.positions,
            'orders': self.orders,
            'performance': {
                'daily_return': 0,  # Calculate daily return
                'total_return': (self.portfolio_value / self.initial_capital) - 1
            }
        }
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        self.logger.info(f"Trading report saved to {report_file}")
        
        # Send report notification
        self._send_notification(f"Trading Report {report_date}: Portfolio Value: {self.portfolio_value:.2f}, Daily Return: {report['performance']['daily_return']:.2%}")
    
    def _send_notification(self, message):
        """
        Send notification.
        
        Args:
            message (str): Notification message
        """
        if not self.config.ENABLE_EMAIL_NOTIFICATIONS:
            return
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.SMTP_USER
            msg['To'] = self.config.NOTIFICATION_EMAIL
            msg['Subject'] = f"HKStock-Quant Trading Notification"
            
            # Add message body
            body = f"""
            <html>
            <body>
                <h2>HKStock-Quant Trading Notification</h2>
                <p>{message}</p>
                <p>Time: {datetime.now()}</p>
            </body>
            </html>
            """
            msg.attach(MIMEText(body, 'html'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT)
            server.starttls()
            server.login(self.config.SMTP_USER, self.config.SMTP_PASSWORD)
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Notification sent: {message}")
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
    
    def get_performance(self):
        """
        Get trading performance.
        
        Returns:
            dict: Performance metrics
        """
        # Calculate performance metrics
        performance = {
            'initial_capital': self.initial_capital,
            'current_value': self.portfolio_value,
            'total_return': (self.portfolio_value / self.initial_capital) - 1,
            'cash': self.cash,
            'positions': self.positions,
            'last_update': self.last_update_time
        }
        
        return performance
