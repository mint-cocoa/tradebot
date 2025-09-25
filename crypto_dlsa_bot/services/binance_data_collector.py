"""
Binance Public Data Collector Implementation

This module implements data collection from Binance using:
1. Binance Public Data repository scripts for bulk historical data downloads
2. Binance API for real-time and recent data collection
"""

import os
import logging
import subprocess
import zipfile
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from binance.client import Client
    from binance import BinanceAPIException, BinanceRequestException
except ImportError as e:
    logging.error(f"Required Binance libraries not installed: {e}")
    Client = None

from crypto_dlsa_bot.services.interfaces import DataCollectorInterface
from crypto_dlsa_bot.models.data_models import OHLCVData
from crypto_dlsa_bot.utils.logging import get_logger
from crypto_dlsa_bot.utils.validation import validate_symbol, validate_timeframe


class BinanceDataCollector(DataCollectorInterface):
    """
    Binance Public Data Collector for historical OHLCV data
    
    Uses Binance Public Data repository for bulk historical downloads
    and Binance API for recent/real-time data collection.
    """
    
    # Supported timeframes mapping (Binance Public Data format)
    TIMEFRAME_MAPPING = {
        '1m': '1m',
        '3m': '3m',
        '5m': '5m', 
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
        '6h': '6h',
        '8h': '8h',
        '12h': '12h',
        '1d': '1d',
        '3d': '3d',
        '1w': '1w'
    }
    
    # Major cryptocurrency pairs
    MAJOR_SYMBOLS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
        'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT',
        'LTCUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'FILUSDT'
    ]
    
    def __init__(
        self,
        data_dir: str = "data/binance",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit_delay: float = 0.1,
        max_workers: int = 4,
        use_public_data: bool = True
    ):
        """
        Initialize Binance Data Collector
        
        Args:
            data_dir: Directory to store downloaded data
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            rate_limit_delay: Delay between API calls to avoid rate limits
            max_workers: Maximum number of concurrent workers
            use_public_data: Whether to use Binance Public Data for bulk downloads
        """
        self.logger = get_logger(__name__)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        self.max_workers = max_workers
        self.use_public_data = use_public_data
        
        # Initialize Binance client (optional for API access)
        self.client = Client() if Client else None
        
        # Setup binance-public-data repository path
        self.public_data_repo = Path(__file__).parent.parent.parent / "binance-public-data"
        self.public_data_python = self.public_data_repo / "python"
        
        # Validate binance-public-data repository exists
        if not self.public_data_repo.exists():
            self.logger.warning(f"Binance public data repository not found at {self.public_data_repo}")
            self.use_public_data = False
        
        self.logger.info(f"Initialized BinanceDataCollector with data_dir: {self.data_dir}")
        self.logger.info(f"Using public data scripts: {self.use_public_data}")
    
    def collect_ohlcv(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        market_type: str = "spot"
    ) -> pd.DataFrame:
        """
        Collect OHLCV data for specified symbols and time range
        
        Args:
            symbols: List of trading symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
            timeframe: Time interval (e.g., '1h', '1d')
            start_date: Start date for data collection
            end_date: End date for data collection
            market_type: Market type ('spot' or 'futures')
            
        Returns:
            DataFrame with OHLCV data
        """
        if timeframe not in self.TIMEFRAME_MAPPING:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        if market_type not in ['spot', 'futures']:
            raise ValueError(f"Unsupported market type: {market_type}")
        
        self.logger.info(
            f"Starting OHLCV data collection for {len(symbols)} symbols "
            f"from {start_date} to {end_date} with {timeframe} timeframe"
        )
        
        # Determine collection method based on date range and availability
        if self.use_public_data and self._should_use_public_data(start_date, end_date):
            return self._collect_using_public_data(symbols, timeframe, start_date, end_date, market_type)
        else:
            return self._collect_using_api(symbols, timeframe, start_date, end_date, market_type)
    
    def _should_use_public_data(self, start_date: datetime, end_date: datetime) -> bool:
        """
        Determine if public data should be used based on date range
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            True if public data should be used, False otherwise
        """
        # Use public data for historical data (older than 7 days)
        cutoff_date = datetime.now() - timedelta(days=7)
        return end_date < cutoff_date
    
    def _collect_using_public_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        market_type: str
    ) -> pd.DataFrame:
        """
        Collect data using binance-public-data scripts
        
        Args:
            symbols: List of trading symbols
            timeframe: Time interval
            start_date: Start date
            end_date: End date
            market_type: Market type
            
        Returns:
            DataFrame with OHLCV data
        """
        self.logger.info("Using binance-public-data scripts for bulk download")
        
        # Create temporary download directory
        download_dir = self.data_dir / "temp_download"
        download_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download data using public data scripts
            self._download_public_data(symbols, timeframe, start_date, end_date, market_type, download_dir)
            
            # Process downloaded files
            all_data = []
            for symbol in symbols:
                symbol_data = self._process_downloaded_files(symbol, timeframe, download_dir, market_type)
                if not symbol_data.empty:
                    all_data.append(symbol_data)
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                combined_data = combined_data.sort_values(['symbol', 'timestamp'])
                return combined_data
            else:
                return pd.DataFrame()
                
        finally:
            # Clean up temporary directory
            import shutil
            if download_dir.exists():
                shutil.rmtree(download_dir)
    
    def _download_public_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        market_type: str,
        download_dir: Path
    ):
        """
        Download data using binance-public-data scripts
        
        Args:
            symbols: List of trading symbols
            timeframe: Time interval
            start_date: Start date
            end_date: End date
            market_type: Market type
            download_dir: Directory to store downloaded data
        """
        # Prepare command arguments
        script_path = self.public_data_python / "download-kline.py"
        
        # Map market type to script argument
        market_type_arg = "spot" if market_type == "spot" else "um"
        
        # Build command
        cmd = [
            "python3", str(script_path),
            "-t", market_type_arg,
            "-s"] + symbols + [
            "-i", timeframe,
            "-startDate", start_date.strftime("%Y-%m-%d"),
            "-endDate", end_date.strftime("%Y-%m-%d"),
            "-folder", str(download_dir),
            "-skip-monthly", "0",
            "-skip-daily", "0"
        ]
        
        self.logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Set environment variable for the script
        env = os.environ.copy()
        env["STORE_DIRECTORY"] = str(download_dir)
        
        try:
            # Execute the download script
            result = subprocess.run(
                cmd,
                cwd=self.public_data_python,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                self.logger.error(f"Download script failed: {result.stderr}")
                raise RuntimeError(f"Download failed: {result.stderr}")
            
            self.logger.info("Download completed successfully")
            
        except subprocess.TimeoutExpired:
            self.logger.error("Download script timed out")
            raise RuntimeError("Download timed out")
        except Exception as e:
            self.logger.error(f"Failed to execute download script: {e}")
            raise
    
    def _process_downloaded_files(
        self,
        symbol: str,
        timeframe: str,
        download_dir: Path,
        market_type: str
    ) -> pd.DataFrame:
        """
        Process downloaded zip files and extract OHLCV data
        
        Args:
            symbol: Trading symbol
            timeframe: Time interval
            download_dir: Directory containing downloaded files
            market_type: Market type
            
        Returns:
            DataFrame with processed OHLCV data
        """
        # Determine data path structure based on market type
        if market_type == "spot":
            data_path = download_dir / "data" / "spot" / "monthly" / "klines" / symbol / timeframe
            daily_path = download_dir / "data" / "spot" / "daily" / "klines" / symbol / timeframe
        else:
            data_path = download_dir / "data" / "futures" / "um" / "monthly" / "klines" / symbol / timeframe
            daily_path = download_dir / "data" / "futures" / "um" / "daily" / "klines" / symbol / timeframe
        
        all_data = []
        
        # Process monthly files
        if data_path.exists():
            all_data.extend(self._process_zip_files(data_path, symbol))
        
        # Process daily files
        if daily_path.exists():
            all_data.extend(self._process_zip_files(daily_path, symbol))
        
        if all_data:
            df = pd.DataFrame(all_data)
            df = df.drop_duplicates(subset=['timestamp', 'symbol'])
            df = df.sort_values('timestamp')
            return df
        else:
            return pd.DataFrame()
    
    def _process_zip_files(self, data_path: Path, symbol: str) -> List[Dict]:
        """
        Process zip files in a directory and extract OHLCV data
        
        Args:
            data_path: Path to directory containing zip files
            symbol: Trading symbol
            
        Returns:
            List of OHLCV data dictionaries
        """
        all_data = []
        
        for zip_file in data_path.glob("*.zip"):
            try:
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    # Get the CSV file name (should be same as zip but with .csv extension)
                    csv_filename = zip_file.stem + ".csv"
                    
                    if csv_filename in zf.namelist():
                        with zf.open(csv_filename) as csv_file:
                            # Read CSV data
                            csv_reader = csv.reader(csv_file.read().decode('utf-8').splitlines())
                            
                            for row in csv_reader:
                                if len(row) >= 6:  # Ensure we have enough columns
                                    try:
                                        ohlcv_data = {
                                            'timestamp': datetime.fromtimestamp(int(row[0]) / 1000),
                                            'symbol': symbol,
                                            'open': float(row[1]),
                                            'high': float(row[2]),
                                            'low': float(row[3]),
                                            'close': float(row[4]),
                                            'volume': float(row[5])
                                        }
                                        all_data.append(ohlcv_data)
                                    except (ValueError, IndexError) as e:
                                        self.logger.warning(f"Invalid row in {zip_file}: {row}, error: {e}")
                                        continue
                    
            except Exception as e:
                self.logger.error(f"Failed to process {zip_file}: {e}")
                continue
        
        return all_data
    
    def _collect_using_api(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        market_type: str
    ) -> pd.DataFrame:
        """
        Collect data using Binance API (fallback method)
        
        Args:
            symbols: List of trading symbols
            timeframe: Time interval
            start_date: Start date
            end_date: End date
            market_type: Market type
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.client:
            self.logger.error("Binance client not available for API collection")
            return pd.DataFrame()
        
        self.logger.info("Using Binance API for data collection")
        
        all_data = []
        
        # Use ThreadPoolExecutor for concurrent data collection
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for each symbol
            future_to_symbol = {
                executor.submit(
                    self._collect_symbol_data_api,
                    symbol, timeframe, start_date, end_date, market_type
                ): symbol for symbol in symbols
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(symbols), desc="Collecting data via API") as pbar:
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        symbol_data = future.result()
                        if not symbol_data.empty:
                            all_data.append(symbol_data)
                        pbar.set_postfix(symbol=symbol)
                    except Exception as e:
                        self.logger.error(f"Failed to collect data for {symbol}: {e}")
                    finally:
                        pbar.update(1)
        
        if not all_data:
            self.logger.warning("No data collected for any symbols")
            return pd.DataFrame()
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values(['symbol', 'timestamp'])
        
        self.logger.info(f"Successfully collected {len(combined_data)} records via API")
        return combined_data
    
    def _collect_symbol_data_api(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        market_type: str
    ) -> pd.DataFrame:
        """
        Collect data for a single symbol with retry logic
        
        Args:
            symbol: Trading symbol
            timeframe: Time interval
            start_date: Start date
            end_date: End date
            market_type: Market type ('spot' or 'futures')
            
        Returns:
            DataFrame with OHLCV data for the symbol
        """
        for attempt in range(self.max_retries):
            try:
                # Add delay to avoid rate limits
                time.sleep(self.rate_limit_delay)
                
                # Convert dates to milliseconds
                start_ms = int(start_date.timestamp() * 1000)
                end_ms = int(end_date.timestamp() * 1000)
                
                # Collect data in chunks to handle large date ranges
                data_chunks = []
                current_start = start_ms
                chunk_size_days = 30  # 30 days per chunk
                chunk_size_ms = chunk_size_days * 24 * 60 * 60 * 1000
                
                while current_start < end_ms:
                    current_end = min(current_start + chunk_size_ms, end_ms)
                    
                    # Get klines data
                    if market_type == 'spot':
                        klines = self.client.get_klines(
                            symbol=symbol,
                            interval=self.TIMEFRAME_MAPPING[timeframe],
                            startTime=current_start,
                            endTime=current_end,
                            limit=1000
                        )
                    else:
                        klines = self.client.futures_klines(
                            symbol=symbol,
                            interval=self.TIMEFRAME_MAPPING[timeframe],
                            startTime=current_start,
                            endTime=current_end,
                            limit=1000
                        )
                    
                    if klines:
                        chunk_df = self._process_klines_data(klines, symbol)
                        data_chunks.append(chunk_df)
                    
                    current_start = current_end + 1
                    
                    # Small delay between chunks
                    time.sleep(0.1)
                
                if data_chunks:
                    symbol_data = pd.concat(data_chunks, ignore_index=True)
                    symbol_data = symbol_data.drop_duplicates(subset=['timestamp', 'symbol'])
                    return symbol_data
                else:
                    self.logger.warning(f"No data found for {symbol}")
                    return pd.DataFrame()
                    
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1} failed for {symbol}: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    self.logger.error(f"All attempts failed for {symbol}")
                    return pd.DataFrame()
        
        return pd.DataFrame()
    
    def _process_klines_data(self, klines: List, symbol: str) -> pd.DataFrame:
        """
        Process raw klines data into standardized format
        
        Args:
            klines: Raw klines data from Binance API
            symbol: Trading symbol
            
        Returns:
            Processed DataFrame
        """
        processed_data = []
        
        for kline in klines:
            try:
                ohlcv_data = OHLCVData(
                    timestamp=datetime.fromtimestamp(kline[0] / 1000),
                    symbol=symbol,
                    open=float(kline[1]),
                    high=float(kline[2]),
                    low=float(kline[3]),
                    close=float(kline[4]),
                    volume=float(kline[5])
                )
                processed_data.append({
                    'timestamp': ohlcv_data.timestamp,
                    'symbol': ohlcv_data.symbol,
                    'open': ohlcv_data.open,
                    'high': ohlcv_data.high,
                    'low': ohlcv_data.low,
                    'close': ohlcv_data.close,
                    'volume': ohlcv_data.volume
                })
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Invalid kline data for {symbol}: {e}")
                continue
        
        return pd.DataFrame(processed_data)
    
    def get_available_symbols(self, market_type: str = "spot") -> List[str]:
        """
        Get list of available symbols from Binance
        
        Args:
            market_type: Market type ('spot' or 'futures')
            
        Returns:
            List of available symbols
        """
        try:
            if market_type == 'spot':
                exchange_info = self.client.get_exchange_info()
            else:
                exchange_info = self.client.futures_exchange_info()
            
            symbols = [
                s['symbol'] for s in exchange_info['symbols'] 
                if s['status'] == 'TRADING' and s['symbol'].endswith('USDT')
            ]
            
            return sorted(symbols)
            
        except Exception as e:
            self.logger.error(f"Failed to get available symbols: {e}")
            return self.MAJOR_SYMBOLS
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate collected OHLCV data quality
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            True if data is valid, False otherwise
        """
        if data.empty:
            self.logger.warning("Data validation failed: empty DataFrame")
            return False
        
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in data.columns]
            self.logger.error(f"Missing required columns in data: {missing_cols}")
            return False
        
        # Check for null values
        null_counts = data[required_columns].isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Found null values in data: {null_counts.to_dict()}")
            return False
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (data['high'] < data[['open', 'close']].max(axis=1)) |
            (data['low'] > data[['open', 'close']].min(axis=1))
        )
        
        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            self.logger.warning(f"Found {invalid_count} invalid OHLC relationships")
            
            # Log some examples of invalid data
            invalid_samples = data[invalid_ohlc].head(3)
            for _, row in invalid_samples.iterrows():
                self.logger.warning(
                    f"Invalid OHLC for {row['symbol']} at {row['timestamp']}: "
                    f"O={row['open']}, H={row['high']}, L={row['low']}, C={row['close']}"
                )
            return False
        
        # Check for negative or zero prices
        price_columns = ['open', 'high', 'low', 'close']
        negative_prices = (data[price_columns] <= 0).any(axis=1)
        if negative_prices.any():
            self.logger.warning(f"Found {negative_prices.sum()} rows with negative or zero prices")
            return False
        
        # Check for negative volumes
        if (data['volume'] < 0).any():
            negative_volume_count = (data['volume'] < 0).sum()
            self.logger.warning(f"Found {negative_volume_count} negative volume values")
            return False
        
        # Check for duplicate timestamps per symbol
        duplicates = data.duplicated(subset=['symbol', 'timestamp'])
        if duplicates.any():
            duplicate_count = duplicates.sum()
            self.logger.warning(f"Found {duplicate_count} duplicate timestamp entries")
            return False
        
        # Check for reasonable price ranges (basic sanity check)
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol]
            price_range = symbol_data[price_columns].max().max() / symbol_data[price_columns].min().min()
            
            if price_range > 1000:  # Price changed by more than 1000x
                self.logger.warning(f"Suspicious price range for {symbol}: {price_range:.2f}x")
        
        # Check timestamp ordering within each symbol
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].sort_index()
            if not symbol_data['timestamp'].is_monotonic_increasing:
                self.logger.warning(f"Timestamps not in chronological order for {symbol}")
        
        self.logger.info(f"Data validation passed for {len(data)} records across {data['symbol'].nunique()} symbols")
        return True
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and fix common data quality issues
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            Cleaned DataFrame
        """
        if data.empty:
            return data
        
        self.logger.info(f"Starting data cleaning for {len(data)} records")
        original_count = len(data)
        
        # Remove rows with null values
        data = data.dropna(subset=['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
        
        # Remove rows with negative or zero prices
        price_columns = ['open', 'high', 'low', 'close']
        data = data[(data[price_columns] > 0).all(axis=1)]
        
        # Remove rows with negative volumes
        data = data[data['volume'] >= 0]
        
        # Fix invalid OHLC relationships by adjusting high/low
        invalid_high = data['high'] < data[['open', 'close']].max(axis=1)
        if invalid_high.any():
            self.logger.info(f"Fixing {invalid_high.sum()} invalid high prices")
            data.loc[invalid_high, 'high'] = data.loc[invalid_high, ['open', 'close']].max(axis=1)
        
        invalid_low = data['low'] > data[['open', 'close']].min(axis=1)
        if invalid_low.any():
            self.logger.info(f"Fixing {invalid_low.sum()} invalid low prices")
            data.loc[invalid_low, 'low'] = data.loc[invalid_low, ['open', 'close']].min(axis=1)
        
        # Remove duplicates
        data = data.drop_duplicates(subset=['symbol', 'timestamp'])
        
        # Sort by symbol and timestamp
        data = data.sort_values(['symbol', 'timestamp'])
        
        # Reset index
        data = data.reset_index(drop=True)
        
        cleaned_count = len(data)
        removed_count = original_count - cleaned_count
        
        if removed_count > 0:
            self.logger.info(f"Data cleaning completed: removed {removed_count} invalid records, {cleaned_count} records remaining")
        else:
            self.logger.info("Data cleaning completed: no invalid records found")
        
        return data
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize and standardize OHLCV data
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            Normalized DataFrame
        """
        if data.empty:
            return data
        
        normalized_data = data.copy()
        
        # Sort by symbol and timestamp
        normalized_data = normalized_data.sort_values(['symbol', 'timestamp'])
        
        # Remove duplicates
        normalized_data = normalized_data.drop_duplicates(subset=['symbol', 'timestamp'])
        
        # Reset index
        normalized_data = normalized_data.reset_index(drop=True)
        
        # Ensure proper data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            normalized_data[col] = pd.to_numeric(normalized_data[col], errors='coerce')
        
        # Remove rows with NaN values
        normalized_data = normalized_data.dropna(subset=numeric_columns)
        
        self.logger.info(f"Normalized data: {len(normalized_data)} records")
        return normalized_data
    
    def collect_onchain_metrics(
        self,
        symbols: List[str],
        metrics: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Placeholder for on-chain metrics collection
        (Not implemented in Binance collector)
        """
        self.logger.warning("On-chain metrics collection not supported by Binance collector")
        return pd.DataFrame()
    
    def get_data_quality_report(self, data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data quality report
        
        Args:
            data: OHLCV data to analyze
            
        Returns:
            Dictionary containing quality metrics
        """
        if data.empty:
            return {"status": "empty", "total_records": 0}
        
        report = {
            "status": "analyzed",
            "total_records": len(data),
            "symbols": data['symbol'].nunique(),
            "symbol_list": sorted(data['symbol'].unique().tolist()),
            "date_range": {
                "start": data['timestamp'].min().isoformat(),
                "end": data['timestamp'].max().isoformat(),
                "days": (data['timestamp'].max() - data['timestamp'].min()).days
            },
            "data_quality": {}
        }
        
        # Check for missing data
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_data = data[required_columns].isnull().sum()
        report["data_quality"]["missing_values"] = missing_data.to_dict()
        
        # Check for duplicates
        duplicates = data.duplicated(subset=['symbol', 'timestamp']).sum()
        report["data_quality"]["duplicate_records"] = int(duplicates)
        
        # Check OHLC validity
        invalid_ohlc = (
            (data['high'] < data[['open', 'close']].max(axis=1)) |
            (data['low'] > data[['open', 'close']].min(axis=1))
        ).sum()
        report["data_quality"]["invalid_ohlc"] = int(invalid_ohlc)
        
        # Check for negative values
        negative_prices = (data[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
        negative_volumes = (data['volume'] < 0).sum()
        report["data_quality"]["negative_prices"] = int(negative_prices)
        report["data_quality"]["negative_volumes"] = int(negative_volumes)
        
        # Calculate completeness per symbol
        symbol_completeness = {}
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol]
            expected_records = len(pd.date_range(
                start=symbol_data['timestamp'].min(),
                end=symbol_data['timestamp'].max(),
                freq='H'  # Assuming hourly data for estimation
            ))
            actual_records = len(symbol_data)
            completeness = min(100.0, (actual_records / expected_records) * 100) if expected_records > 0 else 0
            symbol_completeness[symbol] = round(completeness, 2)
        
        report["data_quality"]["completeness_by_symbol"] = symbol_completeness
        
        # Overall quality score
        quality_issues = (
            missing_data.sum() + duplicates + invalid_ohlc + 
            negative_prices + negative_volumes
        )
        quality_score = max(0, 100 - (quality_issues / len(data) * 100))
        report["data_quality"]["overall_score"] = round(quality_score, 2)
        
        return report
    
    def save_data(self, data: pd.DataFrame, filename: str = None) -> Path:
        """
        Save OHLCV data to file with proper formatting
        
        Args:
            data: OHLCV data to save
            filename: Optional filename, auto-generated if not provided
            
        Returns:
            Path to saved file
        """
        if data.empty:
            raise ValueError("Cannot save empty DataFrame")
        
        if filename is None:
            # Generate filename based on data content
            symbols = "_".join(sorted(data['symbol'].unique())[:3])  # First 3 symbols
            if len(data['symbol'].unique()) > 3:
                symbols += f"_and_{len(data['symbol'].unique()) - 3}_more"
            
            start_date = data['timestamp'].min().strftime("%Y%m%d")
            end_date = data['timestamp'].max().strftime("%Y%m%d")
            filename = f"binance_data_{symbols}_{start_date}_to_{end_date}.csv"
        
        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        file_path = self.data_dir / filename
        
        # Sort data before saving
        sorted_data = data.sort_values(['symbol', 'timestamp'])
        
        # Save with proper formatting
        sorted_data.to_csv(file_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
        
        self.logger.info(f"Saved {len(data)} records to {file_path}")
        return file_path
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load OHLCV data from file
        
        Args:
            file_path: Path to data file
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            data = pd.read_csv(file_path, parse_dates=['timestamp'])
            self.logger.info(f"Loaded {len(data)} records from {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {e}")
            raise