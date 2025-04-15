import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import json

from src.trading.base_trader import BaseTrader
from src.utils.logger import get_logger
from src.config.config import trading_config

# Import Futu API
try:
    from futu import *
except ImportError:
    pass  # Will be handled in the connect method

class FutuTrader(BaseTrader):
    """
    Trader implementation for Futu API.
    
    This class implements the BaseTrader interface for the Futu trading platform,
    which provides access to Hong Kong, US, and China A-share markets.
    """
    
    def __init__(self, strategy, symbols, initial_capital=None, config_path=None):
        """
        Initialize the Futu trader.
        
        Args:
            strategy: Strategy instance or class
            symbols (list): List of symbols to trade
            initial_capital (float): Initial capital
            config_path (str): Path to configuration file
        """
        super().__init__(strategy, symbols, initial_capital, config_path)
        
        # Futu API specific attributes
        self.quote_ctx = None
        self.trade_ctx = None
        self.market_map = {
            'HK': TrdMarket.HK,  # Hong Kong market
            'US': TrdMarket.US,  # US market
            'CN': TrdMarket.SHANGHAI,  # China A-share market (Shanghai)
            'SZ': TrdMarket.SHENZHEN  # China A-share market (Shenzhen)
        }
        
        # Default to HK market
        self.market = TrdMarket.HK
        
        # Map symbol to market
        self.symbol_market_map = {}
        for symbol in self.symbols:
            if symbol.endswith('.HK'):
                self.symbol_market_map[symbol] = TrdMarket.HK
            elif symbol.endswith('.US'):
                self.symbol_market_map[symbol] = TrdMarket.US
            elif symbol.endswith('.SH'):
                self.symbol_market_map[symbol] = TrdMarket.SHANGHAI
            elif symbol.endswith('.SZ'):
                self.symbol_market_map[symbol] = TrdMarket.SHENZHEN
            else:
                # Default to HK market
                self.symbol_market_map[symbol] = TrdMarket.HK
        
        self.logger = get_logger("FutuTrader")
    
    def connect(self):
        """
        Connect to the Futu OpenD API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Check if Futu API is installed
            try:
                from futu import OpenQuoteContext, OpenHKTradeContext, TrdEnv
            except ImportError:
                self.logger.error("Futu API not installed. Please install it with 'pip install futu-api'")
                return False
            
            # Connect to Futu OpenD
            self.logger.info(f"Connecting to Futu OpenD at {self.config.FUTU_API_HOST}:{self.config.FUTU_API_PORT}")
            
            # Initialize quote context
            self.quote_ctx = OpenQuoteContext(host=self.config.FUTU_API_HOST, port=self.config.FUTU_API_PORT)
            
            # Initialize trade context
            self.trade_ctx = OpenHKTradeContext(host=self.config.FUTU_API_HOST, port=self.config.FUTU_API_PORT)
            
            # Unlock trade
            if self.config.FUTU_API_UNLOCK_PASSWORD:
                ret, data = self.trade_ctx.unlock_trade(password=self.config.FUTU_API_UNLOCK_PASSWORD)
                if ret != RET_OK:
                    self.logger.error(f"Failed to unlock trade: {data}")
                    return False
                self.logger.info("Trade unlocked")
            
            # Subscribe to market data for all symbols
            for symbol in self.symbols:
                ret, data = self.quote_ctx.subscribe(symbol, [SubType.QUOTE, SubType.ORDER_BOOK, SubType.K_DAY, SubType.K_1M])
                if ret != RET_OK:
                    self.logger.error(f"Failed to subscribe to {symbol}: {data}")
                else:
                    self.logger.info(f"Subscribed to {symbol}")
            
            self.logger.info("Connected to Futu OpenD")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to connect to Futu OpenD: {e}")
            return False
    
    def disconnect(self):
        """
        Disconnect from the Futu OpenD API.
        
        Returns:
            bool: True if disconnection is successful, False otherwise
        """
        try:
            # Close quote context
            if self.quote_ctx:
                self.quote_ctx.close()
                self.quote_ctx = None
            
            # Close trade context
            if self.trade_ctx:
                self.trade_ctx.close()
                self.trade_ctx = None
            
            self.logger.info("Disconnected from Futu OpenD")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to disconnect from Futu OpenD: {e}")
            return False
    
    def get_market_data(self, symbol, timeframe='1d', bars=100):
        """
        Get market data for a symbol.
        
        Args:
            symbol (str): Symbol
            timeframe (str): Timeframe ('1d', '1m', etc.)
            bars (int): Number of bars
            
        Returns:
            pd.DataFrame: DataFrame with market data
        """
        if not self.quote_ctx:
            self.logger.error("Not connected to Futu OpenD")
            return pd.DataFrame()
        
        try:
            # Map timeframe to Futu API timeframe
            ktype_map = {
                '1m': KLType.K_1M,
                '5m': KLType.K_5M,
                '15m': KLType.K_15M,
                '30m': KLType.K_30M,
                '60m': KLType.K_60M,
                '1d': KLType.K_DAY,
                '1w': KLType.K_WEEK,
                '1M': KLType.K_MON
            }
            
            ktype = ktype_map.get(timeframe, KLType.K_DAY)
            
            # Get K-line data
            ret, data = self.quote_ctx.get_history_kline(
                symbol, start=None, end=None, ktype=ktype, autype=AuType.QFQ, max_count=bars
            )
            
            if ret != RET_OK:
                self.logger.error(f"Failed to get market data for {symbol}: {data}")
                return pd.DataFrame()
            
            # Rename columns to standard format
            data = data.rename(columns={
                'time_key': 'timestamp',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'turnover': 'Turnover',
                'pe_ratio': 'PE',
                'turnover_rate': 'TurnoverRate',
                'last_close': 'PrevClose'
            })
            
            # Convert timestamp to datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Set index
            data = data.set_index('timestamp')
            
            return data
        
        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_account_info(self):
        """
        Get account information.
        
        Returns:
            dict: Account information
        """
        if not self.trade_ctx:
            self.logger.error("Not connected to Futu OpenD")
            return {}
        
        try:
            # Get account balance
            ret, data = self.trade_ctx.accinfo_query(trd_env=TrdEnv.REAL)
            
            if ret != RET_OK:
                self.logger.error(f"Failed to get account information: {data}")
                return {}
            
            # Extract account information
            account_info = {
                'power': data['power'][0],  # Buying power
                'total_assets': data['total_assets'][0],  # Total assets
                'cash': data['cash'][0],  # Cash
                'market_value': data['market_val'][0],  # Market value of positions
                'frozen_cash': data['frozen_cash'][0],  # Frozen cash
                'net_assets': data['net_assets'][0],  # Net assets
                'currency': data['currency'][0]  # Currency
            }
            
            # Set portfolio value
            account_info['portfolio_value'] = account_info['total_assets']
            
            return account_info
        
        except Exception as e:
            self.logger.error(f"Failed to get account information: {e}")
            return {}
    
    def get_positions(self):
        """
        Get current positions.
        
        Returns:
            dict: Dictionary of positions
        """
        if not self.trade_ctx:
            self.logger.error("Not connected to Futu OpenD")
            return {}
        
        try:
            # Get positions
            ret, data = self.trade_ctx.position_list_query(trd_env=TrdEnv.REAL)
            
            if ret != RET_OK:
                self.logger.error(f"Failed to get positions: {data}")
                return {}
            
            # Extract positions
            positions = {}
            for _, row in data.iterrows():
                symbol = row['code']
                quantity = row['qty']
                positions[symbol] = quantity
            
            # Add zeros for symbols without positions
            for symbol in self.symbols:
                if symbol not in positions:
                    positions[symbol] = 0
            
            return positions
        
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return {}
    
    def place_order(self, symbol, order_type, quantity, price=None, stop_price=None, time_in_force='day'):
        """
        Place an order.
        
        Args:
            symbol (str): Symbol
            order_type (str): Order type ('market', 'limit', 'stop', 'stop_limit')
            quantity (int): Quantity
            price (float): Price for limit orders
            stop_price (float): Stop price for stop orders
            time_in_force (str): Time in force
            
        Returns:
            str: Order ID
        """
        if not self.trade_ctx:
            self.logger.error("Not connected to Futu OpenD")
            return None
        
        try:
            # Map order type to Futu API order type
            order_type_map = {
                'market': OrderType.MARKET,
                'limit': OrderType.NORMAL,
                'stop': OrderType.STOP,
                'stop_limit': OrderType.STOP_LIMIT
            }
            
            futu_order_type = order_type_map.get(order_type.lower(), OrderType.NORMAL)
            
            # Determine order direction
            current_position = self.positions.get(symbol, 0)
            
            if quantity > current_position:
                # Buy
                trd_side = TrdSide.BUY
                qty = quantity - current_position
            else:
                # Sell
                trd_side = TrdSide.SELL
                qty = current_position - quantity
            
            # Skip if quantity is zero
            if qty == 0:
                self.logger.warning(f"Skipping order for {symbol} due to zero quantity")
                return None
            
            # Get market for symbol
            market = self.symbol_market_map.get(symbol, TrdMarket.HK)
            
            # Place order
            ret, data = self.trade_ctx.place_order(
                price=price if price else 0,
                qty=qty,
                code=symbol,
                trd_side=trd_side,
                order_type=futu_order_type,
                trd_env=TrdEnv.REAL,
                trd_mkt=market
            )
            
            if ret != RET_OK:
                self.logger.error(f"Failed to place order for {symbol}: {data}")
                return None
            
            # Extract order ID
            order_id = data['order_id'][0]
            
            self.logger.info(f"Placed order {order_id} for {symbol}: {trd_side} {qty} shares at {price}")
            
            return order_id
        
        except Exception as e:
            self.logger.error(f"Failed to place order for {symbol}: {e}")
            return None
    
    def cancel_order(self, order_id):
        """
        Cancel an order.
        
        Args:
            order_id (str): Order ID
            
        Returns:
            bool: True if cancellation is successful, False otherwise
        """
        if not self.trade_ctx:
            self.logger.error("Not connected to Futu OpenD")
            return False
        
        try:
            # Cancel order
            ret, data = self.trade_ctx.modify_order(
                modify_order_op=ModifyOrderOp.CANCEL,
                order_id=order_id,
                qty=0,
                price=0,
                trd_env=TrdEnv.REAL
            )
            
            if ret != RET_OK:
                self.logger.error(f"Failed to cancel order {order_id}: {data}")
                return False
            
            self.logger.info(f"Cancelled order {order_id}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id):
        """
        Get order status.
        
        Args:
            order_id (str): Order ID
            
        Returns:
            dict: Order status
        """
        if not self.trade_ctx:
            self.logger.error("Not connected to Futu OpenD")
            return {}
        
        try:
            # Get order status
            ret, data = self.trade_ctx.order_list_query(
                order_id=order_id,
                trd_env=TrdEnv.REAL
            )
            
            if ret != RET_OK:
                self.logger.error(f"Failed to get order status for {order_id}: {data}")
                return {}
            
            if len(data) == 0:
                self.logger.warning(f"No order found with ID {order_id}")
                return {}
            
            # Extract order status
            order_status = {
                'order_id': data['order_id'][0],
                'status': data['order_status'][0],
                'status_name': data['status_name'][0],
                'symbol': data['code'][0],
                'side': data['trd_side'][0],
                'quantity': data['qty'][0],
                'price': data['price'][0],
                'create_time': data['create_time'][0],
                'updated_time': data['updated_time'][0]
            }
            
            return order_status
        
        except Exception as e:
            self.logger.error(f"Failed to get order status for {order_id}: {e}")
            return {}
    
    def get_market_state(self, symbol):
        """
        Get market state for a symbol.
        
        Args:
            symbol (str): Symbol
            
        Returns:
            str: Market state
        """
        if not self.quote_ctx:
            self.logger.error("Not connected to Futu OpenD")
            return "Unknown"
        
        try:
            # Get market state
            ret, data = self.quote_ctx.get_market_state([symbol])
            
            if ret != RET_OK:
                self.logger.error(f"Failed to get market state for {symbol}: {data}")
                return "Unknown"
            
            # Extract market state
            market_state = data['market_state'][0]
            
            return market_state
        
        except Exception as e:
            self.logger.error(f"Failed to get market state for {symbol}: {e}")
            return "Unknown"
    
    def get_quote(self, symbol):
        """
        Get real-time quote for a symbol.
        
        Args:
            symbol (str): Symbol
            
        Returns:
            dict: Quote data
        """
        if not self.quote_ctx:
            self.logger.error("Not connected to Futu OpenD")
            return {}
        
        try:
            # Get real-time quote
            ret, data = self.quote_ctx.get_stock_quote([symbol])
            
            if ret != RET_OK:
                self.logger.error(f"Failed to get quote for {symbol}: {data}")
                return {}
            
            # Extract quote data
            quote = {
                'symbol': data['code'][0],
                'name': data['name'][0],
                'last_price': data['last_price'][0],
                'open_price': data['open_price'][0],
                'high_price': data['high_price'][0],
                'low_price': data['low_price'][0],
                'prev_close': data['prev_close_price'][0],
                'volume': data['volume'][0],
                'turnover': data['turnover'][0],
                'timestamp': data['update_time'][0]
            }
            
            return quote
        
        except Exception as e:
            self.logger.error(f"Failed to get quote for {symbol}: {e}")
            return {}
    
    def get_order_book(self, symbol):
        """
        Get order book for a symbol.
        
        Args:
            symbol (str): Symbol
            
        Returns:
            dict: Order book data
        """
        if not self.quote_ctx:
            self.logger.error("Not connected to Futu OpenD")
            return {}
        
        try:
            # Get order book
            ret, data = self.quote_ctx.get_order_book(symbol, num=10)
            
            if ret != RET_OK:
                self.logger.error(f"Failed to get order book for {symbol}: {data}")
                return {}
            
            # Extract order book data
            order_book = {
                'symbol': symbol,
                'bid': [],
                'ask': [],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Extract bid data
            for i in range(len(data['Bid'])):
                order_book['bid'].append({
                    'price': data['Bid'][i][0],
                    'volume': data['Bid'][i][1]
                })
            
            # Extract ask data
            for i in range(len(data['Ask'])):
                order_book['ask'].append({
                    'price': data['Ask'][i][0],
                    'volume': data['Ask'][i][1]
                })
            
            return order_book
        
        except Exception as e:
            self.logger.error(f"Failed to get order book for {symbol}: {e}")
            return {}
    
    def get_trading_days(self, market, start_date=None, end_date=None):
        """
        Get trading days for a market.
        
        Args:
            market (str): Market ('HK', 'US', 'CN')
            start_date (str): Start date in format YYYY-MM-DD
            end_date (str): End date in format YYYY-MM-DD
            
        Returns:
            list: List of trading days
        """
        if not self.quote_ctx:
            self.logger.error("Not connected to Futu OpenD")
            return []
        
        try:
            # Set default dates if not provided
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Get market from string
            market_code = self.market_map.get(market.upper(), TrdMarket.HK)
            
            # Get trading days
            ret, data = self.quote_ctx.get_trading_days(market_code, start=start_date, end=end_date)
            
            if ret != RET_OK:
                self.logger.error(f"Failed to get trading days for {market}: {data}")
                return []
            
            # Extract trading days
            trading_days = data['trading_day'].tolist()
            
            return trading_days
        
        except Exception as e:
            self.logger.error(f"Failed to get trading days for {market}: {e}")
            return []
    
    def is_trading_day(self, date=None, market='HK'):
        """
        Check if a date is a trading day.
        
        Args:
            date (str): Date in format YYYY-MM-DD
            market (str): Market ('HK', 'US', 'CN')
            
        Returns:
            bool: True if the date is a trading day, False otherwise
        """
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        trading_days = self.get_trading_days(market, start_date=date, end_date=date)
        
        return len(trading_days) > 0
