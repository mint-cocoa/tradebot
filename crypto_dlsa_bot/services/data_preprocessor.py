"""
Data Preprocessing and Storage System

This module implements comprehensive data preprocessing and storage functionality:
1. Binance Kline data format conversion to standard OHLCV format
2. Multi-source data integration and deduplication
3. Missing data handling and quality validation
4. Efficient Parquet-based storage system
5. Data validation pipeline and metadata management
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import json
import hashlib

from crypto_dlsa_bot.models.data_models import OHLCVData
from crypto_dlsa_bot.utils.logging import get_logger
from crypto_dlsa_bot.utils.validation import validate_symbol, validate_timeframe


class DataPreprocessor:
    """
    Comprehensive data preprocessing and storage system for cryptocurrency data
    """
    
    def __init__(
        self,
        storage_dir: str = "data/processed",
        raw_data_dir: str = "data/raw",
        metadata_dir: str = "data/metadata",
        compression: str = "snappy",
        max_workers: int = 4
    ):
        """Initialize Data Preprocessor"""
        self.logger = get_logger(__name__)
        
        # Setup directories
        self.storage_dir = Path(storage_dir)
        self.raw_data_dir = Path(raw_data_dir)
        self.metadata_dir = Path(metadata_dir)
        
        for directory in [self.storage_dir, self.raw_data_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.compression = compression
        self.max_workers = max_workers
        
        self.logger.info(f"Initialized DataPreprocessor with storage: {self.storage_dir}")
    
    def convert_binance_kline_format(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Convert Binance Kline data format to standard OHLCV format"""
        if raw_data.empty:
            return pd.DataFrame()
        
        self.logger.info(f"Converting {len(raw_data)} Binance Kline records to standard format")
        
        # Handle different input formats
        if isinstance(raw_data, list):
            columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count',
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
            ]
            raw_data = pd.DataFrame(raw_data, columns=columns)
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in raw_data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create standardized DataFrame
        standardized_data = pd.DataFrame()
        
        # Convert timestamp
        if raw_data['timestamp'].dtype == 'object':
            standardized_data['timestamp'] = pd.to_datetime(raw_data['timestamp'], unit='ms', errors='coerce')
        else:
            timestamps = pd.to_numeric(raw_data['timestamp'], errors='coerce')
            if timestamps.max() < 1e12:
                standardized_data['timestamp'] = pd.to_datetime(timestamps, unit='s')
            else:
                standardized_data['timestamp'] = pd.to_datetime(timestamps, unit='ms')
        
        # Copy symbol if exists
        if 'symbol' in raw_data.columns:
            standardized_data['symbol'] = raw_data['symbol']
        else:
            standardized_data['symbol'] = 'UNKNOWN'
        
        # Convert price and volume data
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            standardized_data[col] = pd.to_numeric(raw_data[col], errors='coerce')
        
        standardized_data['volume'] = pd.to_numeric(raw_data['volume'], errors='coerce')
        
        # Add optional columns
        if 'quote_volume' in raw_data.columns:
            standardized_data['quote_volume'] = pd.to_numeric(raw_data['quote_volume'], errors='coerce')
        else:
            standardized_data['quote_volume'] = standardized_data['volume'] * standardized_data['close']
        
        # Add metadata
        standardized_data['data_source'] = 'binance'
        standardized_data['quality_score'] = 1.0
        
        # Remove rows with invalid timestamps
        standardized_data = standardized_data.dropna(subset=['timestamp'])
        
        self.logger.info(f"Converted to {len(standardized_data)} standardized records")
        return standardized_data
    
    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Comprehensive data quality validation and scoring"""
        if data.empty:
            return data, {"status": "empty", "total_records": 0}

        self.logger.info(f"Validating data quality for {len(data)} records")
        
        validated_data = data.copy()
        quality_issues = []
        
        # Initialize quality scores
        validated_data['quality_score'] = 1.0
        
        # Check for required columns
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in validated_data.columns]
        
        if missing_columns:
            quality_issues.append(f"Missing required columns: {missing_columns}")
            return pd.DataFrame(), {"status": "invalid", "issues": quality_issues}
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (validated_data['high'] < validated_data[['open', 'close']].max(axis=1)) |
            (validated_data['low'] > validated_data[['open', 'close']].min(axis=1))
        )
        
        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            quality_issues.append(f"Invalid OHLC relationships: {invalid_count} records")
            validated_data.loc[invalid_ohlc, 'quality_score'] *= 0.5
        
        # Check for negative or zero prices
        price_columns = ['open', 'high', 'low', 'close']
        negative_prices = (validated_data[price_columns] <= 0).any(axis=1)
        
        if negative_prices.any():
            negative_count = negative_prices.sum()
            quality_issues.append(f"Negative or zero prices: {negative_count} records")
            validated_data.loc[negative_prices, 'quality_score'] *= 0.3
        
        # Check for negative volumes
        negative_volume = validated_data['volume'] < 0
        
        if negative_volume.any():
            negative_vol_count = negative_volume.sum()
            quality_issues.append(f"Negative volumes: {negative_vol_count} records")
            validated_data.loc[negative_volume, 'quality_score'] *= 0.7
        
        # Generate quality report
        quality_report = {
            "status": "validated",
            "total_records": len(validated_data),
            "symbols": validated_data['symbol'].nunique(),
            "quality_issues": quality_issues,
            "average_quality_score": validated_data['quality_score'].mean(),
            "records_by_quality": {
                "high_quality": (validated_data['quality_score'] >= 0.9).sum(),
                "medium_quality": ((validated_data['quality_score'] >= 0.7) & 
                                 (validated_data['quality_score'] < 0.9)).sum(),
                "low_quality": (validated_data['quality_score'] < 0.7).sum()
            },
            "date_range": {
                "start": validated_data['timestamp'].min().isoformat() if not validated_data.empty else None,
                "end": validated_data['timestamp'].max().isoformat() if not validated_data.empty else None
            }
        }
        
        self.logger.info(f"Data quality validation completed. Average quality score: {quality_report['average_quality_score']:.3f}")

        return validated_data, quality_report

    def validate_ohlcv_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Lightweight wrapper that normalises OHLCV inputs and validates quality."""

        if data.empty:
            return data.copy()

        df = data.copy()
        if 'timestamp' not in df.columns:
            if 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            else:
                raise ValueError("OHLCV 데이터에 date 또는 timestamp 컬럼이 필요합니다")
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        if 'close' not in df.columns:
            raise ValueError("OHLCV 데이터에 close 컬럼이 필요합니다")
        for col in ['open', 'high', 'low']:
            if col not in df.columns:
                df[col] = df['close']
        if 'volume' not in df.columns:
            df['volume'] = 0.0

        validated_df, _ = self.validate_data_quality(df)
        if validated_df.empty:
            return validated_df

        if 'date' not in validated_df.columns:
            validated_df['date'] = validated_df['timestamp']
        validated_df['date'] = pd.to_datetime(validated_df['date'])
        validated_df = validated_df.sort_values(['symbol', 'date']).reset_index(drop=True)
        validated_df = validated_df.drop(columns=['timestamp'], errors='ignore')
        return validated_df
    
    def save_to_parquet(
        self,
        data: pd.DataFrame,
        symbol: str = None,
        timeframe: str = "1h",
        partition_by: str = "date"
    ) -> str:
        """Save data to Parquet format with efficient partitioning"""
        if data.empty:
            self.logger.warning("Cannot save empty DataFrame")
            return ""
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Add partitioning columns
        data = data.copy()
        data['date'] = data['timestamp'].dt.date
        
        # Determine output path structure
        if symbol:
            base_path = self.storage_dir / "ohlcv" / timeframe / symbol
        else:
            base_path = self.storage_dir / "ohlcv" / timeframe / "multi_symbol"
        
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Save as single file for simplicity
        output_path = base_path / f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        
        try:
            # Convert to PyArrow table and save
            table = pa.Table.from_pandas(data, preserve_index=False)
            pq.write_table(table, str(output_path), compression=self.compression)
            
            self.logger.info(f"Saved {len(data)} records to {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
            return ""
    
    def process_pipeline(
        self,
        raw_data: pd.DataFrame,
        symbol: str = None,
        timeframe: str = "1h",
        missing_data_method: str = "interpolate",
        save_to_storage: bool = True,
        partition_by: str = "date"
    ) -> Tuple[pd.DataFrame, Dict]:
        """Complete data processing pipeline"""
        self.logger.info(f"Starting data processing pipeline for {len(raw_data) if not raw_data.empty else 0} records")
        
        processing_report = {
            "start_time": datetime.now().isoformat(),
            "steps_completed": [],
            "errors": [],
            "warnings": []
        }
        
        try:
            # Step 1: Convert format if needed
            if not raw_data.empty:
                if 'close_time' in raw_data.columns or len(raw_data.columns) >= 12:
                    processed_data = self.convert_binance_kline_format(raw_data)
                    processing_report["steps_completed"].append("format_conversion")
                else:
                    processed_data = raw_data.copy()
                    processing_report["steps_completed"].append("format_validation")
            else:
                processed_data = raw_data
            
            # Step 2: Validate data quality
            if not processed_data.empty:
                processed_data, quality_report = self.validate_data_quality(processed_data)
                processing_report["quality_report"] = quality_report
                processing_report["steps_completed"].append("quality_validation")
            
            # Step 3: Save to storage if requested
            if save_to_storage and not processed_data.empty:
                storage_path = self.save_to_parquet(
                    processed_data,
                    symbol=symbol,
                    timeframe=timeframe,
                    partition_by=partition_by
                )
                processing_report["storage_path"] = storage_path
                processing_report["steps_completed"].append("storage")
            
            # Final report
            processing_report["end_time"] = datetime.now().isoformat()
            processing_report["success"] = True
            processing_report["final_record_count"] = len(processed_data)
            
            self.logger.info(f"Data processing pipeline completed successfully. Final records: {len(processed_data)}")
            
        except Exception as e:
            processing_report["errors"].append(str(e))
            processing_report["success"] = False
            processing_report["end_time"] = datetime.now().isoformat()
            self.logger.error(f"Data processing pipeline failed: {e}")
            processed_data = pd.DataFrame()
        
        return processed_data, processing_report
    
    def get_storage_summary(self) -> Dict:
        """Get summary of stored data"""
        summary = {
            "total_files": 0,
            "total_size_mb": 0,
            "timeframes": {},
            "symbols": set(),
            "last_updated": None
        }
        
        try:
            ohlcv_dir = self.storage_dir / "ohlcv"
            if not ohlcv_dir.exists():
                return summary
            
            for timeframe_dir in ohlcv_dir.iterdir():
                if timeframe_dir.is_dir():
                    timeframe = timeframe_dir.name
                    summary["timeframes"][timeframe] = {
                        "symbols": [],
                        "files": 0,
                        "size_mb": 0
                    }
                    
                    for symbol_dir in timeframe_dir.iterdir():
                        if symbol_dir.is_dir():
                            symbol = symbol_dir.name
                            summary["symbols"].add(symbol)
                            summary["timeframes"][timeframe]["symbols"].append(symbol)
                            
                            for file_path in symbol_dir.rglob("*.parquet"):
                                summary["total_files"] += 1
                                summary["timeframes"][timeframe]["files"] += 1
                                
                                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                                summary["total_size_mb"] += file_size_mb
                                summary["timeframes"][timeframe]["size_mb"] += file_size_mb
                                
                                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                                if summary["last_updated"] is None or file_mtime > summary["last_updated"]:
                                    summary["last_updated"] = file_mtime
            
            summary["symbols"] = sorted(list(summary["symbols"]))
            
        except Exception as e:
            self.logger.error(f"Failed to generate storage summary: {e}")
        
        return summary
