#!/usr/bin/env python3
"""
Cryptocurrency Data Download and Preprocessing Script

This script demonstrates the complete data pipeline:
1. Download historical OHLCV data using Binance Data Collector
2. Preprocess and validate the data
3. Store in efficient Parquet format
4. Generate quality reports

Usage:
    python scripts/download_and_preprocess_data.py --symbols BTCUSDT,ETHUSDT --timeframe 1h --days 30
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crypto_dlsa_bot.services.binance_data_collector import BinanceDataCollector
from crypto_dlsa_bot.services.data_preprocessor import DataPreprocessor
from crypto_dlsa_bot.utils.logging import get_logger


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Download and preprocess cryptocurrency data"
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT,ETHUSDT,BNBUSDT",
        help="Comma-separated list of trading symbols (default: BTCUSDT,ETHUSDT,BNBUSDT)"
    )
    
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        choices=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"],
        help="Time interval for data collection (default: 1h)"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of historical data to download (default: 30)"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date in YYYY-MM-DD format (overrides --days)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date in YYYY-MM-DD format (default: today)"
    )
    
    parser.add_argument(
        "--market-type",
        type=str,
        default="spot",
        choices=["spot", "futures"],
        help="Market type (default: spot)"
    )
    
    parser.add_argument(
        "--missing-data-method",
        type=str,
        default="interpolate",
        choices=["interpolate", "forward_fill", "drop"],
        help="Method for handling missing data (default: interpolate)"
    )
    
    parser.add_argument(
        "--partition-by",
        type=str,
        default="date",
        choices=["date", "symbol", "year_month"],
        help="Partitioning strategy for storage (default: date)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base directory for data storage (default: data)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save processed data to storage (for testing)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    import logging
    
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('data_pipeline.log')
        ]
    )


def main():
    """Main execution function"""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    logger = get_logger(__name__)
    logger.info("Starting cryptocurrency data download and preprocessing pipeline")
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    logger.info(f"Target symbols: {symbols}")
    
    # Parse dates
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_date = datetime.now() - timedelta(days=args.days)
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.now()
    
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Market type: {args.market_type}")
    
    try:
        # Initialize data collector
        logger.info("Initializing Binance Data Collector...")
        collector = BinanceDataCollector(
            data_dir=os.path.join(args.data_dir, "raw", "binance"),
            use_public_data=True,
            max_workers=4
        )
        
        # Initialize data preprocessor
        logger.info("Initializing Data Preprocessor...")
        preprocessor = DataPreprocessor(
            storage_dir=os.path.join(args.data_dir, "processed"),
            raw_data_dir=os.path.join(args.data_dir, "raw"),
            metadata_dir=os.path.join(args.data_dir, "metadata")
        )
        
        # Collect raw data
        logger.info("Starting data collection...")
        raw_data = collector.collect_ohlcv(
            symbols=symbols,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date,
            market_type=args.market_type
        )
        
        if raw_data.empty:
            logger.error("No data collected. Exiting.")
            return 1
        
        logger.info(f"Collected {len(raw_data)} raw records")
        
        # Display data summary
        print("\n" + "="*60)
        print("RAW DATA SUMMARY")
        print("="*60)
        print(f"Total records: {len(raw_data):,}")
        print(f"Symbols: {sorted(raw_data['symbol'].unique())}")
        print(f"Date range: {raw_data['timestamp'].min()} to {raw_data['timestamp'].max()}")
        print(f"Timeframe: {args.timeframe}")
        
        # Show sample data
        print("\nSample data (first 5 records):")
        print(raw_data.head().to_string())
        
        # Process data through pipeline
        logger.info("Starting data preprocessing pipeline...")
        processed_data, processing_report = preprocessor.process_pipeline(
            raw_data=raw_data,
            timeframe=args.timeframe,
            missing_data_method=args.missing_data_method,
            save_to_storage=not args.no_save,
            partition_by=args.partition_by
        )
        
        # Display processing results
        print("\n" + "="*60)
        print("PROCESSING RESULTS")
        print("="*60)
        print(f"Processing success: {processing_report.get('success', False)}")
        print(f"Steps completed: {', '.join(processing_report.get('steps_completed', []))}")
        print(f"Final record count: {processing_report.get('final_record_count', 0):,}")
        
        if processing_report.get('errors'):
            print(f"Errors: {processing_report['errors']}")
        
        if processing_report.get('warnings'):
            print(f"Warnings: {processing_report['warnings']}")
        
        # Display quality report
        if 'quality_report' in processing_report:
            quality_report = processing_report['quality_report']
            print("\n" + "="*60)
            print("DATA QUALITY REPORT")
            print("="*60)
            print(f"Total records: {quality_report.get('total_records', 0):,}")
            print(f"Unique symbols: {quality_report.get('symbols', 0)}")
            print(f"Average quality score: {quality_report.get('average_quality_score', 0):.3f}")
            
            if 'records_by_quality' in quality_report:
                quality_breakdown = quality_report['records_by_quality']
                print(f"High quality records (≥0.9): {quality_breakdown.get('high_quality', 0):,}")
                print(f"Medium quality records (0.7-0.9): {quality_breakdown.get('medium_quality', 0):,}")
                print(f"Low quality records (<0.7): {quality_breakdown.get('low_quality', 0):,}")
            
            if quality_report.get('quality_issues'):
                print(f"Quality issues found: {len(quality_report['quality_issues'])}")
                for issue in quality_report['quality_issues'][:5]:  # Show first 5 issues
                    print(f"  - {issue}")
                if len(quality_report['quality_issues']) > 5:
                    print(f"  ... and {len(quality_report['quality_issues']) - 5} more issues")
        
        # Display storage information
        if not args.no_save and processing_report.get('storage_path'):
            print("\n" + "="*60)
            print("STORAGE INFORMATION")
            print("="*60)
            print(f"Data saved to: {processing_report['storage_path']}")
            
            # Get storage summary
            storage_summary = preprocessor.get_storage_summary()
            print(f"Total files in storage: {storage_summary.get('total_files', 0)}")
            print(f"Total storage size: {storage_summary.get('total_size_mb', 0):.2f} MB")
            print(f"Available timeframes: {list(storage_summary.get('timeframes', {}).keys())}")
            print(f"Available symbols: {len(storage_summary.get('symbols', []))}")
        
        # Show processed data sample
        if not processed_data.empty:
            print("\n" + "="*60)
            print("PROCESSED DATA SAMPLE")
            print("="*60)
            print("First 5 records:")
            print(processed_data.head().to_string())
            
            print("\nLast 5 records:")
            print(processed_data.tail().to_string())
            
            # Show basic statistics
            print("\nBasic statistics:")
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in numeric_columns if col in processed_data.columns]
            if available_columns:
                print(processed_data[available_columns].describe())
        
        logger.info("Data pipeline completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)