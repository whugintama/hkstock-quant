import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from src.config.config import config

class Logger:
    """
    Custom logger for the application
    """
    def __init__(self, name, log_file=None, log_level=None):
        self.name = name
        self.log_level = log_level or config.LOG_LEVEL
        self.log_file = log_file or config.LOG_FILE_PATH
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self._get_log_level())
        self.logger.propagate = False
        
        # Clear existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Add handlers
        self._add_console_handler()
        self._add_file_handler()
    
    def _get_log_level(self):
        """Convert string log level to logging level"""
        levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        return levels.get(self.log_level.upper(), logging.INFO)
    
    def _add_console_handler(self):
        """Add handler for console output"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self._get_log_level())
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self):
        """Add handler for file output"""
        # Create log directory if it doesn't exist
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(self._get_log_level())
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def get_logger(self):
        """Return the configured logger"""
        return self.logger

# Create default logger
def get_logger(name):
    """Get a logger instance with the given name"""
    return Logger(name).get_logger()
