#!/usr/bin/env python3
"""
Demo script to test Binance Data Collector functionality
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from crypto_dlsa_bot.services.binance_data_collector import BinanceDataCollector
from crypto_dlsa_bot.utils.logging import get_logger

def main():
    """Main demo function"""
    logger = get_logger(__name__)
    
    # Initialize collector
    collector = BinanceDataCollector(
        data_dir="demo_data",
        max_retries=2,
        rate_limit_delay=0.5,
        max_workers=2
    )
    
    logger.info("=== Binance Data Collector Demo ===")
    
    # Test 1: Get available symbols
    logger.info("1. Testing get_available_symbols...")
    try:
        symbols = collector.get_available_symbols('spot')
        logger.info(f"Found {len(symbols)} available symbols")
        logger.info(f"First 10 symbols: {symbols[:10]}")
    except Exception as e:
        logger.error(f"Failed to get symbols: {e}")
        symbols = collector.MAJOR_SYMBOLS[:5]  # Use fallback
        logger.info(f"Using fallback symbols: {symbols}")
    
    # Test 2: Collect small amount of recent data (API method)
    logger.info("2. Testing API data collection...")
    try:
        # Use recent dates to force API collection
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(hours=6)  # 6 hours of data
        
        test_symbols = symbols[:2]  # Test with first 2 symbols
        
        logger.info(f"Collecting data for {test_symbols} from {start_date} to {end_date}")
        
        data = collector.collect_ohlcv(
            symbols=test_symbols,
            timeframe='1h',
            start_date=start_date,
            end_date=end_date,
            market_type='spot'
        )
        
        if not data.empty:
            logger.info(f"Successfully collected {len(data)} records")
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Symbols in data: {data['symbol'].unique().tolist()}")
            logger.info(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
            
            # Test data validation
            is_valid = collector.validate_data(data)
            logger.info(f"Data validation result: {is_valid}")
            
            # Test data quality report
            quality_report = collector.get_data_quality_report(data)
            logger.info(f"Data quality score: {quality_report['data_quality']['overall_score']}")
            
            # Test data cleaning
            cleaned_data = collector.clean_data(data)
            logger.info(f"Data after cleaning: {len(cleaned_data)} records")
            
            # Test saving data
            if len(cleaned_data) > 0:
                file_path = collector.save_data(cleaned_data, 'demo_test_data.csv')
                logger.info(f"Saved data to: {file_path}")
                
                # Test loading data
                loaded_data = collector.load_data(file_path)
                logger.info(f"Loaded data: {len(loaded_data)} records")
                
                # Clean up
                file_path.unlink()
                logger.info("Cleaned up test file")
        else:
            logger.warning("No data collected")
            
    except Exception as e:
        logger.error(f"API data collection failed: {e}")
    
    # Test 3: Test public data method (if available)
    logger.info("3. Testing public data availability...")
    if collector.use_public_data and collector.public_data_repo.exists():
        logger.info("Binance public data repository found")
        logger.info(f"Repository path: {collector.public_data_repo}")
        
        # Test with historical dates (should trigger public data method)
        historical_end = datetime.now() - timedelta(days=30)
        historical_start = historical_end - timedelta(days=1)
        
        should_use_public = collector._should_use_public_data(historical_start, historical_end)
        logger.info(f"Should use public data for historical dates: {should_use_public}")
    else:
        logger.info("Binance public data repository not available")
    
    logger.info("=== Demo completed ===")

if __name__ == "__main__":
    main()