import os
import pandas as pd
import numpy as np
import akshare as ak
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

from src.utils.logger import get_logger
from src.config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = get_logger("DataFetcher")

class DataFetcher:
    """
    Base class for fetching market data
    """
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or RAW_DATA_DIR
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def get_data(self, symbol, start_date, end_date, timeframe='1d', source='auto'):
        """
        Get historical price data for a symbol
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in format YYYY-MM-DD
            end_date (str): End date in format YYYY-MM-DD
            timeframe (str): Data frequency ('1d', '1h', etc.)
            source (str): Data source ('yfinance', 'akshare', 'auto')
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        raise NotImplementedError("Subclasses must implement get_data method")
    
    def _cache_file_path(self, symbol, timeframe, start_date, end_date):
        """Generate a cache file path for the data"""
        cache_filename = f"{symbol}_{timeframe}_{start_date}_{end_date}.csv"
        return os.path.join(self.cache_dir, cache_filename)
    
    def _save_to_cache(self, df, cache_path):
        """Save data to cache"""
        df.to_csv(cache_path)
        logger.info(f"Data saved to cache: {cache_path}")
    
    def _load_from_cache(self, cache_path):
        """Load data from cache if available"""
        if os.path.exists(cache_path):
            logger.info(f"Loading data from cache: {cache_path}")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df
        return None


class HKStockDataFetcher(DataFetcher):
    """
    Data fetcher for Hong Kong stocks
    """
    def __init__(self, cache_dir=None):
        super().__init__(cache_dir)
    
    def get_data(self, symbol, start_date, end_date, timeframe='1d', source='auto'):
        """
        Get historical price data for a Hong Kong stock
        
        Args:
            symbol (str): Stock symbol (e.g., '0700.HK')
            start_date (str): Start date in format YYYY-MM-DD
            end_date (str): End date in format YYYY-MM-DD
            timeframe (str): Data frequency ('1d', '1h', etc.)
            source (str): Data source ('yfinance', 'akshare', 'auto')
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        # Check cache first
        cache_path = self._cache_file_path(symbol, timeframe, start_date, end_date)
        cached_data = self._load_from_cache(cache_path)
        if cached_data is not None:
            return cached_data
        
        # Determine data source
        if source == 'auto':
            # Try akshare first for HK stocks, fallback to yfinance
            try:
                df = self._fetch_from_akshare(symbol, start_date, end_date)
            except Exception as e:
                logger.warning(f"Failed to fetch data from akshare: {e}")
                df = self._fetch_from_yfinance(symbol, start_date, end_date, timeframe)
        elif source == 'akshare':
            df = self._fetch_from_akshare(symbol, start_date, end_date)
        elif source == 'yfinance':
            df = self._fetch_from_yfinance(symbol, start_date, end_date, timeframe)
        else:
            raise ValueError(f"Unsupported data source: {source}")
        
        # Save to cache
        if df is not None and not df.empty:
            self._save_to_cache(df, cache_path)
        
        return df
    
    def _fetch_from_akshare(self, symbol, start_date, end_date):
        """Fetch data from akshare"""
        logger.info(f"Fetching data from akshare for {symbol}")
        
        # Convert symbol format if needed (0700.HK -> 00700)
        if '.HK' in symbol:
            symbol_code = symbol.split('.')[0].zfill(5)
        else:
            symbol_code = symbol.zfill(5)
        
        # Fetch data
        try:
            df = ak.stock_hk_daily(symbol=symbol_code, adjust="qfq")
            
            # Filter by date
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            # Rename columns to standard format
            df = df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Set index
            df = df.set_index('Date')
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data from akshare: {e}")
            raise
    
    def _fetch_from_yfinance(self, symbol, start_date, end_date, timeframe):
        """Fetch data from yfinance"""
        logger.info(f"Fetching data from yfinance for {symbol}")
        
        # Ensure symbol has .HK suffix
        if not symbol.endswith('.HK'):
            symbol = f"{symbol}.HK"
        
        # Fetch data
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=timeframe)
            
            # Rename columns to standard format
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data from yfinance: {e}")
            raise
    
    def get_hk_stock_list(self):
        """Get list of all Hong Kong stocks"""
        try:
            logger.info("Fetching Hong Kong stock list")
            stock_list = ak.stock_hk_spot_em()
            return stock_list
        except Exception as e:
            logger.error(f"Error fetching Hong Kong stock list: {e}")
            return pd.DataFrame()
    
    def get_index_data(self, index_code='HSI', start_date=None, end_date=None):
        """
        Get Hong Kong index data
        
        Args:
            index_code (str): Index code, e.g., 'HSI' for Hang Seng Index
            start_date (str): Start date in format YYYY-MM-DD
            end_date (str): End date in format YYYY-MM-DD
            
        Returns:
            pd.DataFrame: DataFrame with index data
        """
        logger.info(f"Fetching index data for {index_code}")
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Use yfinance for index data
            index_symbol = f"^{index_code}"
            ticker = yf.Ticker(index_symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching index data: {e}")
            return pd.DataFrame()


class DataProcessor:
    """
    Process and transform raw market data
    """
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or PROCESSED_DATA_DIR
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("DataProcessor")
    
    def add_technical_indicators(self, df, indicators=None):
        """
        Add technical indicators to the dataframe
        
        Args:
            df (pd.DataFrame): OHLCV dataframe
            indicators (list): List of indicators to add
            
        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        if indicators is None:
            indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger']
        
        result_df = df.copy()
        
        for indicator in indicators:
            if indicator == 'sma':
                # Simple Moving Averages
                result_df['SMA_5'] = result_df['Close'].rolling(window=5).mean()
                result_df['SMA_10'] = result_df['Close'].rolling(window=10).mean()
                result_df['SMA_20'] = result_df['Close'].rolling(window=20).mean()
                result_df['SMA_50'] = result_df['Close'].rolling(window=50).mean()
                result_df['SMA_200'] = result_df['Close'].rolling(window=200).mean()
            
            elif indicator == 'ema':
                # Exponential Moving Averages
                result_df['EMA_5'] = result_df['Close'].ewm(span=5, adjust=False).mean()
                result_df['EMA_10'] = result_df['Close'].ewm(span=10, adjust=False).mean()
                result_df['EMA_20'] = result_df['Close'].ewm(span=20, adjust=False).mean()
                result_df['EMA_50'] = result_df['Close'].ewm(span=50, adjust=False).mean()
                result_df['EMA_200'] = result_df['Close'].ewm(span=200, adjust=False).mean()
            
            elif indicator == 'rsi':
                # Relative Strength Index
                delta = result_df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                result_df['RSI_14'] = 100 - (100 / (1 + rs))
            
            elif indicator == 'macd':
                # MACD
                ema_12 = result_df['Close'].ewm(span=12, adjust=False).mean()
                ema_26 = result_df['Close'].ewm(span=26, adjust=False).mean()
                result_df['MACD'] = ema_12 - ema_26
                result_df['MACD_Signal'] = result_df['MACD'].ewm(span=9, adjust=False).mean()
                result_df['MACD_Hist'] = result_df['MACD'] - result_df['MACD_Signal']
            
            elif indicator == 'bollinger':
                # Bollinger Bands
                result_df['BB_Middle'] = result_df['Close'].rolling(window=20).mean()
                result_df['BB_Std'] = result_df['Close'].rolling(window=20).std()
                result_df['BB_Upper'] = result_df['BB_Middle'] + (result_df['BB_Std'] * 2)
                result_df['BB_Lower'] = result_df['BB_Middle'] - (result_df['BB_Std'] * 2)
            
            elif indicator == 'atr':
                # Average True Range
                high_low = result_df['High'] - result_df['Low']
                high_close = np.abs(result_df['High'] - result_df['Close'].shift())
                low_close = np.abs(result_df['Low'] - result_df['Close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = np.max(ranges, axis=1)
                result_df['ATR_14'] = true_range.rolling(14).mean()
        
        return result_df
    
    def save_processed_data(self, df, symbol, suffix='processed'):
        """Save processed data to file"""
        filename = f"{symbol}_{suffix}.csv"
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath)
        self.logger.info(f"Processed data saved to {filepath}")
        return filepath
