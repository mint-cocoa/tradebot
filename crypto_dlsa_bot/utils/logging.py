"""
Logging utilities for Crypto-DLSA Bot
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_directory: str = "./logs",
    log_filename: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_directory: Directory to store log files
        log_filename: Custom log filename (optional)
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    Path(log_directory).mkdir(parents=True, exist_ok=True)
    
    # Generate log filename if not provided
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"crypto_dlsa_{timestamp}.log"
    
    log_filepath = os.path.join(log_directory, log_filename)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('crypto_dlsa_bot')
    logger.info(f"Logging initialized. Log file: {log_filepath}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger instance for specific module"""
    return logging.getLogger(f'crypto_dlsa_bot.{name}')