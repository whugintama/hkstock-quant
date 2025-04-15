import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Logs directory
LOGS_DIR = ROOT_DIR / "logs"

class Config:
    """Base configuration class"""
    # Trading parameters
    INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", 100000))
    COMMISSION_RATE = float(os.getenv("COMMISSION_RATE", 0.0003))
    SLIPPAGE = float(os.getenv("SLIPPAGE", 0.001))
    RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", 0.03))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", str(LOGS_DIR / "app.log"))
    
    # Database
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", 5432))
    DB_NAME = os.getenv("DB_NAME", "hkstock_quant")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    
    @classmethod
    def load_yaml_config(cls, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            yaml_config = yaml.safe_load(file)
        return yaml_config

class BacktestConfig(Config):
    """Configuration for backtesting"""
    DEFAULT_TIMEFRAME = "1d"  # Default timeframe for backtesting
    PERFORMANCE_METRICS = [
        "total_return", "annual_return", "sharpe_ratio", "max_drawdown",
        "win_rate", "profit_factor", "sortino_ratio", "calmar_ratio"
    ]

class TradingConfig(Config):
    """Configuration for live trading"""
    # Futu API settings
    FUTU_API_HOST = os.getenv("FUTU_API_HOST", "127.0.0.1")
    FUTU_API_PORT = int(os.getenv("FUTU_API_PORT", 11111))
    FUTU_API_UNLOCK_PASSWORD = os.getenv("FUTU_API_UNLOCK_PASSWORD", "")
    
    # Trading session
    MARKET_OPEN_TIME = "09:30:00"  # HK market open time
    MARKET_CLOSE_TIME = "16:00:00"  # HK market close time
    
    # Risk management
    MAX_POSITION_SIZE = 0.2  # Maximum position size as a fraction of portfolio
    STOP_LOSS_PCT = 0.05  # Default stop loss percentage
    
    # Notification settings
    ENABLE_EMAIL_NOTIFICATIONS = os.getenv("ENABLE_EMAIL_NOTIFICATIONS", "False").lower() == "true"
    SMTP_SERVER = os.getenv("SMTP_SERVER", "")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    SMTP_USER = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    NOTIFICATION_EMAIL = os.getenv("NOTIFICATION_EMAIL", "")

# Create default config instance
config = Config()
backtest_config = BacktestConfig()
trading_config = TradingConfig()
