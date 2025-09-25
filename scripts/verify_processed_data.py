#!/usr/bin/env python3
"""
Verify Processed Data Script

This script loads and verifies the processed cryptocurrency data
stored in Parquet format.
"""

import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crypto_dlsa_bot.services.data_preprocessor import DataPreprocessor
from crypto_dlsa_bot.utils.logging import get_logger


def main():
    """Main verification function"""
    logger = get_logger(__name__)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        storage_dir="data/processed",
        metadata_dir="data/metadata"
    )
    
    # Get storage summary
    print("="*60)
    print("STORAGE SUMMARY")
    print("="*60)
    
    summary = preprocessor.get_storage_summary()
    print(f"Total files: {summary['total_files']}")
    print(f"Total size: {summary['total_size_mb']:.2f} MB")
    print(f"Available timeframes: {list(summary['timeframes'].keys())}")
    print(f"Available symbols: {len(summary['symbols'])}")
    print(f"Symbols: {summary['symbols']}")
    
    if summary['last_updated']:
        print(f"Last updated: {summary['last_updated']}")
    
    # Show details for each timeframe
    for timeframe, tf_data in summary['timeframes'].items():
        print(f"\nTimeframe {timeframe}:")
        print(f"  Files: {tf_data['files']}")
        print(f"  Size: {tf_data['size_mb']:.2f} MB")
        print(f"  Symbols: {tf_data['symbols']}")
    
    # Load and display sample data from each timeframe
    for timeframe in summary['timeframes'].keys():
        print(f"\n{'='*60}")
        print(f"SAMPLE DATA - {timeframe.upper()} TIMEFRAME")
        print(f"{'='*60}")
        
        try:
            # Find parquet files for this timeframe
            timeframe_dir = Path("data/processed/ohlcv") / timeframe / "multi_symbol"
            parquet_files = list(timeframe_dir.glob("*.parquet"))
            
            if parquet_files:
                # Load the most recent file
                latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
                print(f"Loading from: {latest_file}")
                
                df = pd.read_parquet(latest_file)
                
                print(f"\nDataset info:")
                print(f"  Records: {len(df):,}")
                print(f"  Symbols: {sorted(df['symbol'].unique())}")
                print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"  Columns: {list(df.columns)}")
                
                # Show basic statistics
                print(f"\nBasic statistics:")
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                available_cols = [col for col in numeric_cols if col in df.columns]
                if available_cols:
                    stats = df[available_cols].describe()
                    print(stats)
                
                # Show sample records
                print(f"\nFirst 3 records:")
                print(df.head(3).to_string())
                
                print(f"\nLast 3 records:")
                print(df.tail(3).to_string())
                
                # Data quality check
                if 'quality_score' in df.columns:
                    avg_quality = df['quality_score'].mean()
                    print(f"\nData quality:")
                    print(f"  Average quality score: {avg_quality:.3f}")
                    print(f"  High quality records (≥0.9): {(df['quality_score'] >= 0.9).sum():,}")
                    print(f"  Medium quality records (0.7-0.9): {((df['quality_score'] >= 0.7) & (df['quality_score'] < 0.9)).sum():,}")
                    print(f"  Low quality records (<0.7): {(df['quality_score'] < 0.7).sum():,}")
                
            else:
                print(f"No parquet files found for timeframe {timeframe}")
                
        except Exception as e:
            print(f"Error loading data for timeframe {timeframe}: {e}")
    
    print(f"\n{'='*60}")
    print("VERIFICATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()