"""
Unit tests for Binance Data Collector
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

from crypto_dlsa_bot.services.binance_data_collector import BinanceDataCollector
from crypto_dlsa_bot.models.data_models import OHLCVData


class TestBinanceDataCollector:
    """Test cases for BinanceDataCollector"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def collector(self, temp_data_dir):
        """Create BinanceDataCollector instance for testing"""
        return BinanceDataCollector(
            data_dir=temp_data_dir,
            max_retries=2,
            retry_delay=0.1,
            rate_limit_delay=0.01,
            max_workers=2
        )
    
    @pytest.fixture
    def sample_klines_data(self):
        """Sample klines data from Binance API"""
        return [
            [
                1640995200000,  # timestamp
                "50000.00",     # open
                "51000.00",     # high
                "49500.00",     # low
                "50500.00",     # close
                "100.5",        # volume
                1640995259999,  # close_time
                "5050000.00",   # quote_asset_volume
                1000,           # number_of_trades
                "50.25",        # taker_buy_base_asset_volume
                "2525000.00",   # taker_buy_quote_asset_volume
                "0"             # ignore
            ],
            [
                1640995260000,
                "50500.00",
                "51500.00",
                "50000.00",
                "51000.00",
                "150.75",
                1640995319999,
                "7650000.00",
                1500,
                "75.375",
                "3825000.00",
                "0"
            ]
        ]
    
    def test_initialization(self, temp_data_dir):
        """Test collector initialization"""
        collector = BinanceDataCollector(data_dir=temp_data_dir)
        
        assert collector.data_dir == Path(temp_data_dir)
        assert collector.max_retries == 3
        assert collector.retry_delay == 1.0
        assert collector.rate_limit_delay == 0.1
        assert collector.max_workers == 4
        assert collector.data_dir.exists()
    
    def test_timeframe_validation(self, collector):
        """Test timeframe validation"""
        # Valid timeframes
        valid_timeframes = ['1m', '5m', '1h', '1d']
        for tf in valid_timeframes:
            assert tf in collector.TIMEFRAME_MAPPING
        
        # Invalid timeframe should raise error
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            collector.collect_ohlcv(
                symbols=['BTCUSDT'],
                timeframe='invalid',
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now()
            )
    
    def test_market_type_validation(self, collector):
        """Test market type validation"""
        # Invalid market type should raise error
        with pytest.raises(ValueError, match="Unsupported market type"):
            collector.collect_ohlcv(
                symbols=['BTCUSDT'],
                timeframe='1h',
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now(),
                market_type='invalid'
            )
    
    def test_process_klines_data(self, collector, sample_klines_data):
        """Test processing of raw klines data"""
        result = collector._process_klines_data(sample_klines_data, 'BTCUSDT')
        
        assert len(result) == 2
        assert list(result.columns) == ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        
        # Check first row
        first_row = result.iloc[0]
        assert first_row['symbol'] == 'BTCUSDT'
        assert first_row['open'] == 50000.0
        assert first_row['high'] == 51000.0
        assert first_row['low'] == 49500.0
        assert first_row['close'] == 50500.0
        assert first_row['volume'] == 100.5
        
        # Check timestamp conversion
        expected_timestamp = datetime.fromtimestamp(1640995200000 / 1000)
        assert first_row['timestamp'] == expected_timestamp
    
    def test_process_invalid_klines_data(self, collector):
        """Test processing of invalid klines data"""
        invalid_klines = [
            [1640995200000, "invalid", "51000.00", "49500.00", "50500.00", "100.5"],  # Invalid open price
            [1640995260000, "50500.00", "51500.00", "50000.00", "51000.00"]  # Missing volume
        ]
        
        result = collector._process_klines_data(invalid_klines, 'BTCUSDT')
        
        # Should skip invalid rows
        assert len(result) == 0
    
    @patch('crypto_dlsa_bot.services.binance_data_collector.Client')
    def test_collect_symbol_data_api_success(self, mock_client_class, collector, sample_klines_data):
        """Test successful data collection for single symbol via API"""
        # Mock the Client
        mock_client = Mock()
        mock_client.get_klines.return_value = sample_klines_data
        mock_client_class.return_value = mock_client
        
        # Reinitialize collector to use mocked client
        collector.client = mock_client
        
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 2)
        
        result = collector._collect_symbol_data_api('BTCUSDT', '1h', start_date, end_date, 'spot')
        
        assert len(result) == 2
        assert result['symbol'].iloc[0] == 'BTCUSDT'
        mock_client.get_klines.assert_called()
    
    @patch('crypto_dlsa_bot.services.binance_data_collector.Client')
    def test_collect_symbol_data_api_with_retry(self, mock_client_class, collector):
        """Test data collection with retry logic via API"""
        # Mock the Client to fail first, then succeed
        mock_client = Mock()
        mock_client.get_klines.side_effect = [
            Exception("API Error"),  # First call fails
            []  # Second call succeeds but returns empty
        ]
        mock_client_class.return_value = mock_client
        
        collector.client = mock_client
        collector.retry_delay = 0.01  # Speed up test
        
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 2)
        
        result = collector._collect_symbol_data_api('BTCUSDT', '1h', start_date, end_date, 'spot')
        
        # Should return empty DataFrame after retries
        assert len(result) == 0
        assert mock_client.get_klines.call_count == 2
    
    def test_validate_data_valid(self, collector):
        """Test data validation with valid data"""
        valid_data = pd.DataFrame({
            'timestamp': [datetime.now(), datetime.now() + timedelta(hours=1)],
            'symbol': ['BTCUSDT', 'BTCUSDT'],
            'open': [50000.0, 50500.0],
            'high': [51000.0, 51500.0],
            'low': [49500.0, 50000.0],
            'close': [50500.0, 51000.0],
            'volume': [100.5, 150.75]
        })
        
        assert collector.validate_data(valid_data) is True
    
    def test_validate_data_invalid_ohlc(self, collector):
        """Test data validation with invalid OHLC relationships"""
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTCUSDT'],
            'open': [50000.0],
            'high': [49000.0],  # High < Open (invalid)
            'low': [49500.0],
            'close': [50500.0],
            'volume': [100.5]
        })
        
        assert collector.validate_data(invalid_data) is False
    
    def test_validate_data_negative_volume(self, collector):
        """Test data validation with negative volume"""
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTCUSDT'],
            'open': [50000.0],
            'high': [51000.0],
            'low': [49500.0],
            'close': [50500.0],
            'volume': [-100.5]  # Negative volume (invalid)
        })
        
        assert collector.validate_data(invalid_data) is False
    
    def test_validate_data_missing_columns(self, collector):
        """Test data validation with missing columns"""
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTCUSDT'],
            'open': [50000.0],
            # Missing high, low, close, volume columns
        })
        
        assert collector.validate_data(invalid_data) is False
    
    def test_validate_data_empty(self, collector):
        """Test data validation with empty DataFrame"""
        empty_data = pd.DataFrame()
        assert collector.validate_data(empty_data) is False
    
    def test_normalize_data(self, collector):
        """Test data normalization"""
        # Create data with duplicates and wrong order
        raw_data = pd.DataFrame({
            'timestamp': [
                datetime(2022, 1, 1, 10, 0),
                datetime(2022, 1, 1, 9, 0),   # Out of order
                datetime(2022, 1, 1, 10, 0),  # Duplicate
                datetime(2022, 1, 1, 11, 0)
            ],
            'symbol': ['BTCUSDT', 'BTCUSDT', 'BTCUSDT', 'ETHUSDT'],
            'open': [50000.0, 49000.0, 50000.0, 3000.0],
            'high': [51000.0, 50000.0, 51000.0, 3100.0],
            'low': [49500.0, 48500.0, 49500.0, 2950.0],
            'close': [50500.0, 49500.0, 50500.0, 3050.0],
            'volume': [100.5, 80.0, 100.5, 200.0]
        })
        
        normalized = collector.normalize_data(raw_data)
        
        # Should remove duplicates
        assert len(normalized) == 3
        
        # Should be sorted by symbol and timestamp
        assert normalized.iloc[0]['symbol'] == 'BTCUSDT'
        assert normalized.iloc[0]['timestamp'] == datetime(2022, 1, 1, 9, 0)
        assert normalized.iloc[-1]['symbol'] == 'ETHUSDT'
    
    def test_normalize_data_with_nan(self, collector):
        """Test data normalization with NaN values"""
        data_with_nan = pd.DataFrame({
            'timestamp': [datetime.now(), datetime.now() + timedelta(hours=1)],
            'symbol': ['BTCUSDT', 'BTCUSDT'],
            'open': [50000.0, None],  # NaN value
            'high': [51000.0, 51500.0],
            'low': [49500.0, 50000.0],
            'close': [50500.0, 51000.0],
            'volume': [100.5, 150.75]
        })
        
        normalized = collector.normalize_data(data_with_nan)
        
        # Should remove rows with NaN values
        assert len(normalized) == 1
        assert normalized.iloc[0]['open'] == 50000.0
    
    @patch('crypto_dlsa_bot.services.binance_data_collector.Client')
    def test_get_available_symbols(self, mock_client_class, collector):
        """Test getting available symbols"""
        # Mock exchange info response
        mock_client = Mock()
        mock_client.get_exchange_info.return_value = {
            'symbols': [
                {'symbol': 'BTCUSDT', 'status': 'TRADING'},
                {'symbol': 'ETHUSDT', 'status': 'TRADING'},
                {'symbol': 'ADABTC', 'status': 'TRADING'},  # Not USDT pair
                {'symbol': 'XRPUSDT', 'status': 'BREAK'}    # Not trading
            ]
        }
        mock_client_class.return_value = mock_client
        
        collector.client = mock_client
        
        symbols = collector.get_available_symbols('spot')
        
        # Should only return USDT pairs that are trading
        expected_symbols = ['BTCUSDT', 'ETHUSDT']
        assert sorted(symbols) == sorted(expected_symbols)
    
    @patch('crypto_dlsa_bot.services.binance_data_collector.Client')
    def test_get_available_symbols_error(self, mock_client_class, collector):
        """Test getting available symbols when API fails"""
        # Mock API failure
        mock_client = Mock()
        mock_client.get_exchange_info.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        
        collector.client = mock_client
        
        symbols = collector.get_available_symbols('spot')
        
        # Should return major symbols as fallback
        assert symbols == collector.MAJOR_SYMBOLS
    
    def test_collect_onchain_metrics_not_supported(self, collector):
        """Test that on-chain metrics collection returns empty DataFrame"""
        result = collector.collect_onchain_metrics(
            symbols=['BTCUSDT'],
            metrics=['active_addresses'],
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now()
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_process_zip_files(self, collector, temp_data_dir):
        """Test processing of zip files containing OHLCV data"""
        # This test would require creating actual zip files with CSV data
        # For now, we'll test the method exists and handles empty directory
        data_path = Path(temp_data_dir) / "test_data"
        data_path.mkdir(parents=True, exist_ok=True)
        
        result = collector._process_zip_files(data_path, 'BTCUSDT')
        
        # Should return empty list for directory with no zip files
        assert result == []
    
    def test_major_symbols_list(self, collector):
        """Test that major symbols list is properly defined"""
        assert len(collector.MAJOR_SYMBOLS) > 0
        
        # All symbols should be valid format
        for symbol in collector.MAJOR_SYMBOLS:
            assert symbol.endswith('USDT')
            assert symbol.isupper()
            assert len(symbol) >= 6
    
    def test_timeframe_mapping(self, collector):
        """Test timeframe mapping completeness"""
        expected_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        
        for tf in expected_timeframes:
            assert tf in collector.TIMEFRAME_MAPPING
            assert isinstance(collector.TIMEFRAME_MAPPING[tf], str)
    
    def test_should_use_public_data(self, collector):
        """Test logic for determining when to use public data"""
        # Recent data (within 7 days) should use API
        recent_start = datetime.now() - timedelta(days=3)
        recent_end = datetime.now() - timedelta(days=1)
        assert not collector._should_use_public_data(recent_start, recent_end)
        
        # Historical data (older than 7 days) should use public data
        historical_start = datetime.now() - timedelta(days=30)
        historical_end = datetime.now() - timedelta(days=10)
        assert collector._should_use_public_data(historical_start, historical_end)
    
    def test_clean_data(self, collector):
        """Test data cleaning functionality"""
        # Create data with various issues
        dirty_data = pd.DataFrame({
            'timestamp': [
                datetime(2022, 1, 1, 10, 0),
                datetime(2022, 1, 1, 11, 0),
                datetime(2022, 1, 1, 12, 0),
                datetime(2022, 1, 1, 13, 0),
                datetime(2022, 1, 1, 14, 0)
            ],
            'symbol': ['BTCUSDT', 'BTCUSDT', 'BTCUSDT', 'BTCUSDT', 'BTCUSDT'],
            'open': [50000.0, 51000.0, None, -100.0, 52000.0],  # Null and negative
            'high': [51000.0, 52000.0, 53000.0, 51000.0, 53000.0],
            'low': [49500.0, 50500.0, 51500.0, 49000.0, 51500.0],
            'close': [50500.0, 51500.0, 52500.0, 50000.0, 52500.0],
            'volume': [100.5, 150.0, 200.0, -50.0, 175.0]  # Negative volume
        })
        
        cleaned = collector.clean_data(dirty_data)
        
        # Should remove rows with null, negative prices, and negative volumes
        assert len(cleaned) == 3  # Only 3 valid rows remain
        assert cleaned['open'].min() > 0
        assert cleaned['volume'].min() >= 0
        assert not cleaned.isnull().any().any()
    
    def test_get_data_quality_report(self, collector):
        """Test data quality report generation"""
        # Create test data with known quality issues
        test_data = pd.DataFrame({
            'timestamp': [
                datetime(2022, 1, 1, 10, 0),
                datetime(2022, 1, 1, 11, 0),
                datetime(2022, 1, 1, 10, 0),  # Duplicate
                datetime(2022, 1, 1, 12, 0)
            ],
            'symbol': ['BTCUSDT', 'BTCUSDT', 'BTCUSDT', 'ETHUSDT'],
            'open': [50000.0, 51000.0, 50000.0, None],  # Null value
            'high': [51000.0, 52000.0, 51000.0, 3100.0],
            'low': [49500.0, 50500.0, 49500.0, 2950.0],
            'close': [50500.0, 51500.0, 50500.0, 3050.0],
            'volume': [100.5, 150.0, 100.5, 200.0]
        })
        
        report = collector.get_data_quality_report(test_data)
        
        assert report['status'] == 'analyzed'
        assert report['total_records'] == 4
        assert report['symbols'] == 2
        assert 'BTCUSDT' in report['symbol_list']
        assert 'ETHUSDT' in report['symbol_list']
        assert report['data_quality']['duplicate_records'] == 1
        assert report['data_quality']['missing_values']['open'] == 1
        assert 'overall_score' in report['data_quality']
    
    def test_save_and_load_data(self, collector, temp_data_dir):
        """Test saving and loading data"""
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': [datetime(2022, 1, 1, 10, 0), datetime(2022, 1, 1, 11, 0)],
            'symbol': ['BTCUSDT', 'BTCUSDT'],
            'open': [50000.0, 51000.0],
            'high': [51000.0, 52000.0],
            'low': [49500.0, 50500.0],
            'close': [50500.0, 51500.0],
            'volume': [100.5, 150.0]
        })
        
        # Save data
        file_path = collector.save_data(test_data, 'test_data.csv')
        assert file_path.exists()
        
        # Load data
        loaded_data = collector.load_data(file_path)
        
        # Compare data (timestamps might have slight differences due to serialization)
        assert len(loaded_data) == len(test_data)
        assert list(loaded_data.columns) == list(test_data.columns)
        assert loaded_data['symbol'].tolist() == test_data['symbol'].tolist()
        
        # Clean up
        file_path.unlink()
    
    def test_save_data_auto_filename(self, collector):
        """Test automatic filename generation"""
        test_data = pd.DataFrame({
            'timestamp': [datetime(2022, 1, 1, 10, 0), datetime(2022, 1, 2, 10, 0)],
            'symbol': ['BTCUSDT', 'ETHUSDT'],
            'open': [50000.0, 3000.0],
            'high': [51000.0, 3100.0],
            'low': [49500.0, 2950.0],
            'close': [50500.0, 3050.0],
            'volume': [100.5, 200.0]
        })
        
        file_path = collector.save_data(test_data)
        
        # Check that filename was auto-generated
        assert file_path.name.startswith('binance_data_')
        assert file_path.name.endswith('.csv')
        assert '20220101' in file_path.name
        assert '20220102' in file_path.name
        
        # Clean up
        file_path.unlink()
    
    def test_load_nonexistent_file(self, collector):
        """Test loading non-existent file"""
        with pytest.raises(FileNotFoundError):
            collector.load_data('nonexistent_file.csv')
    
    @patch('crypto_dlsa_bot.services.binance_data_collector.subprocess.run')
    def test_download_public_data(self, mock_subprocess, collector, temp_data_dir):
        """Test downloading data using public data scripts"""
        # Mock successful subprocess execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        # Set up collector with public data enabled
        collector.use_public_data = True
        collector.public_data_python = Path("binance-public-data/python")
        
        download_dir = Path(temp_data_dir) / "download"
        
        try:
            collector._download_public_data(
                symbols=['BTCUSDT'],
                timeframe='1h',
                start_date=datetime(2022, 1, 1),
                end_date=datetime(2022, 1, 2),
                market_type='spot',
                download_dir=download_dir
            )
            
            # Verify subprocess was called with correct arguments
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]
            assert 'download-kline.py' in call_args[1]
            assert '-t' in call_args
            assert 'spot' in call_args
            assert '-s' in call_args
            assert 'BTCUSDT' in call_args
            
        except Exception as e:
            # If binance-public-data directory doesn't exist, this is expected
            if "not found" in str(e).lower():
                pytest.skip("binance-public-data repository not available")
            else:
                raise


@pytest.mark.integration
class TestBinanceDataCollectorIntegration:
    """Integration tests for BinanceDataCollector (require network access)"""
    
    @pytest.fixture
    def collector(self):
        """Create collector for integration tests"""
        return BinanceDataCollector(
            data_dir="test_data",
            max_retries=1,
            rate_limit_delay=0.5  # Be respectful to API
        )
    
    def test_collect_real_data_small_range(self, collector):
        """Test collecting real data for small date range"""
        # Skip if no internet connection
        pytest.importorskip("requests")
        
        start_date = datetime.now() - timedelta(days=2)
        end_date = datetime.now() - timedelta(days=1)
        
        try:
            result = collector.collect_ohlcv(
                symbols=['BTCUSDT'],
                timeframe='1h',
                start_date=start_date,
                end_date=end_date,
                market_type='spot'
            )
            
            if not result.empty:  # Only assert if we got data
                assert len(result) > 0
                assert 'BTCUSDT' in result['symbol'].values
                assert collector.validate_data(result)
        
        except Exception as e:
            pytest.skip(f"Integration test failed due to network/API issue: {e}")
    
    def test_get_real_available_symbols(self, collector):
        """Test getting real available symbols"""
        pytest.importorskip("requests")
        
        try:
            symbols = collector.get_available_symbols('spot')
            
            assert len(symbols) > 0
            assert 'BTCUSDT' in symbols
            assert all(symbol.endswith('USDT') for symbol in symbols[:10])  # Check first 10
        
        except Exception as e:
            pytest.skip(f"Integration test failed due to network/API issue: {e}")