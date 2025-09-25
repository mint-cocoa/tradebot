"""
Unit tests for validation utilities
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from crypto_dlsa_bot.utils.validation import (
    validate_symbol,
    validate_timeframe,
    validate_date_range,
    validate_ohlcv_data,
    validate_factor_data,
    validate_model_predictions
)


class TestSymbolValidation:
    """Test cases for symbol validation"""
    
    def test_valid_symbols(self):
        """Test validation of valid symbols"""
        valid_symbols = [
            'BTCUSDT',
            'ETHUSDT', 
            'BNBUSDT',
            'ADABTC',
            'XRPETH',
            'DOGEUSDC',
            'SHIBBUSD'
        ]
        
        for symbol in valid_symbols:
            assert validate_symbol(symbol) is True
    
    def test_invalid_symbols(self):
        """Test validation of invalid symbols"""
        invalid_symbols = [
            'btcusdt',      # lowercase
            'BTC-USDT',     # contains dash
            'BTC_USDT',     # contains underscore
            'BTC USDT',     # contains space
            'BTC',          # too short
            'BTCUSD',       # doesn't end with valid quote
            '',             # empty string
            123,            # not string
            None            # None value
        ]
        
        for symbol in invalid_symbols:
            assert validate_symbol(symbol) is False


class TestTimeframeValidation:
    """Test cases for timeframe validation"""
    
    def test_valid_timeframes(self):
        """Test validation of valid timeframes"""
        valid_timeframes = [
            '1m', '3m', '5m', '15m', '30m',
            '1h', '2h', '4h', '6h', '8h', '12h',
            '1d', '3d', '1w', '1M'
        ]
        
        for timeframe in valid_timeframes:
            assert validate_timeframe(timeframe) is True
    
    def test_invalid_timeframes(self):
        """Test validation of invalid timeframes"""
        invalid_timeframes = [
            '1min',         # wrong format
            '1hour',        # wrong format
            '1day',         # wrong format
            '2m',           # not supported
            '7d',           # not supported
            '',             # empty string
            123,            # not string
            None            # None value
        ]
        
        for timeframe in invalid_timeframes:
            assert validate_timeframe(timeframe) is False


class TestDateRangeValidation:
    """Test cases for date range validation"""
    
    def test_valid_date_range(self):
        """Test validation of valid date range"""
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 31)
        
        result = validate_date_range(start_date, end_date)
        
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
    
    def test_invalid_date_order(self):
        """Test validation when start date is after end date"""
        start_date = datetime(2022, 1, 31)
        end_date = datetime(2022, 1, 1)
        
        result = validate_date_range(start_date, end_date)
        
        assert result['is_valid'] is False
        assert any('Start date must be before end date' in error for error in result['errors'])
    
    def test_future_start_date(self):
        """Test validation when start date is in the future"""
        start_date = datetime.now() + timedelta(days=1)
        end_date = datetime.now() + timedelta(days=2)
        
        result = validate_date_range(start_date, end_date)
        
        assert result['is_valid'] is False
        assert any('cannot be in the future' in error for error in result['errors'])
    
    def test_future_end_date(self):
        """Test validation when end date is in the future"""
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now() + timedelta(days=1)
        
        result = validate_date_range(start_date, end_date)
        
        assert result['is_valid'] is True
        assert any('in the future' in warning for warning in result['warnings'])
    
    def test_very_large_date_range(self):
        """Test validation of very large date range"""
        start_date = datetime(2017, 1, 1)
        end_date = datetime(2023, 1, 1)  # 6 years
        
        result = validate_date_range(start_date, end_date)
        
        assert result['is_valid'] is True
        assert any('very large' in warning for warning in result['warnings'])
    
    def test_date_before_binance_launch(self):
        """Test validation of date before Binance launch"""
        start_date = datetime(2017, 1, 1)  # Before Binance launch
        end_date = datetime(2017, 12, 31)
        
        result = validate_date_range(start_date, end_date)
        
        assert result['is_valid'] is True
        assert any('before Binance launch' in warning for warning in result['warnings'])


class TestOHLCVDataValidation:
    """Test cases for OHLCV data validation"""
    
    def test_valid_ohlcv_data(self):
        """Test validation of valid OHLCV data"""
        valid_data = pd.DataFrame({
            'timestamp': [datetime.now(), datetime.now() + timedelta(hours=1)],
            'symbol': ['BTCUSDT', 'BTCUSDT'],
            'open': [50000.0, 50500.0],
            'high': [51000.0, 51500.0],
            'low': [49500.0, 50000.0],
            'close': [50500.0, 51000.0],
            'volume': [100.5, 150.75]
        })
        
        result = validate_ohlcv_data(valid_data)
        
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
        assert result['stats']['total_rows'] == 2
        assert result['stats']['unique_symbols'] == 1
    
    def test_missing_columns(self):
        """Test validation with missing required columns"""
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTCUSDT'],
            'open': [50000.0]
            # Missing high, low, close, volume
        })
        
        result = validate_ohlcv_data(invalid_data)
        
        assert result['is_valid'] is False
        assert any('Missing required columns' in error for error in result['errors'])
    
    def test_invalid_ohlc_relationships(self):
        """Test validation with invalid OHLC relationships"""
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTCUSDT'],
            'open': [50000.0],
            'high': [49000.0],  # High < Open (invalid)
            'low': [49500.0],
            'close': [50500.0],
            'volume': [100.5]
        })
        
        result = validate_ohlcv_data(invalid_data)
        
        assert result['is_valid'] is False
        assert any('Invalid OHLC relationships' in error for error in result['errors'])
    
    def test_negative_volume(self):
        """Test validation with negative volume"""
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTCUSDT'],
            'open': [50000.0],
            'high': [51000.0],
            'low': [49500.0],
            'close': [50500.0],
            'volume': [-100.5]  # Negative volume
        })
        
        result = validate_ohlcv_data(invalid_data)
        
        assert result['is_valid'] is False
        assert any('Negative volume' in error for error in result['errors'])
    
    def test_null_values(self):
        """Test validation with null values"""
        data_with_nulls = pd.DataFrame({
            'timestamp': [datetime.now(), datetime.now() + timedelta(hours=1)],
            'symbol': ['BTCUSDT', 'BTCUSDT'],
            'open': [50000.0, None],  # Null value
            'high': [51000.0, 51500.0],
            'low': [49500.0, 50000.0],
            'close': [50500.0, 51000.0],
            'volume': [100.5, 150.75]
        })
        
        result = validate_ohlcv_data(data_with_nulls)
        
        assert any('Null values found' in warning for warning in result['warnings'])
    
    def test_extreme_price_movements(self):
        """Test validation with extreme price movements"""
        extreme_data = pd.DataFrame({
            'timestamp': [datetime.now(), datetime.now() + timedelta(hours=1)],
            'symbol': ['BTCUSDT', 'BTCUSDT'],
            'open': [50000.0, 50500.0],
            'high': [51000.0, 51500.0],
            'low': [49500.0, 50000.0],
            'close': [50500.0, 25000.0],  # 50% drop
            'volume': [100.5, 150.75]
        })
        
        result = validate_ohlcv_data(extreme_data)
        
        assert any('Extreme price movements' in warning for warning in result['warnings'])


class TestFactorDataValidation:
    """Test cases for factor data validation"""
    
    def test_valid_factor_data(self):
        """Test validation of valid factor data"""
        factor_names = ['market_factor', 'size_factor', 'momentum_factor']
        valid_data = pd.DataFrame({
            'timestamp': [datetime.now(), datetime.now() + timedelta(days=1)],
            'symbol': ['BTCUSDT', 'ETHUSDT'],
            'market_factor': [0.5, -0.3],
            'size_factor': [1.2, 0.8],
            'momentum_factor': [-0.1, 0.4]
        })
        
        result = validate_factor_data(valid_data, factor_names)
        
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
        assert result['stats']['total_rows'] == 2
        assert result['stats']['unique_symbols'] == 2
    
    def test_missing_factor_columns(self):
        """Test validation with missing factor columns"""
        factor_names = ['market_factor', 'size_factor', 'momentum_factor']
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTCUSDT'],
            'market_factor': [0.5]
            # Missing size_factor and momentum_factor
        })
        
        result = validate_factor_data(invalid_data, factor_names)
        
        assert result['is_valid'] is False
        assert any('Missing required columns' in error for error in result['errors'])
    
    def test_infinite_values(self):
        """Test validation with infinite values"""
        factor_names = ['market_factor', 'size_factor']
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTCUSDT'],
            'market_factor': [np.inf],  # Infinite value
            'size_factor': [1.2]
        })
        
        result = validate_factor_data(invalid_data, factor_names)
        
        assert result['is_valid'] is False
        assert any('Infinite values found' in error for error in result['errors'])
    
    def test_nan_values(self):
        """Test validation with NaN values"""
        factor_names = ['market_factor', 'size_factor']
        data_with_nan = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTCUSDT'],
            'market_factor': [np.nan],  # NaN value
            'size_factor': [1.2]
        })
        
        result = validate_factor_data(data_with_nan, factor_names)
        
        assert any('NaN values found' in warning for warning in result['warnings'])
    
    def test_extreme_factor_values(self):
        """Test validation with extreme factor values"""
        factor_names = ['market_factor']
        extreme_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTCUSDT'],
            'market_factor': [150.0]  # Extreme value
        })
        
        result = validate_factor_data(extreme_data, factor_names)
        
        assert any('Extreme values' in warning for warning in result['warnings'])


class TestModelPredictionValidation:
    """Test cases for model prediction validation"""
    
    def test_valid_predictions(self):
        """Test validation of valid model predictions"""
        valid_data = pd.DataFrame({
            'timestamp': [datetime.now(), datetime.now() + timedelta(hours=1)],
            'model_version': ['v1.0', 'v1.0'],
            'confidence_score': [0.85, 0.92],
            'BTCUSDT_weight': [0.6, 0.5],
            'ETHUSDT_weight': [0.4, 0.5]
        })
        
        result = validate_model_predictions(valid_data)
        
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
        assert result['stats']['total_predictions'] == 2
    
    def test_missing_required_columns(self):
        """Test validation with missing required columns"""
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'model_version': ['v1.0']
            # Missing confidence_score
        })
        
        result = validate_model_predictions(invalid_data)
        
        assert result['is_valid'] is False
        assert any('Missing required columns' in error for error in result['errors'])
    
    def test_invalid_confidence_scores(self):
        """Test validation with invalid confidence scores"""
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now(), datetime.now() + timedelta(hours=1)],
            'model_version': ['v1.0', 'v1.0'],
            'confidence_score': [1.5, -0.1]  # Invalid scores
        })
        
        result = validate_model_predictions(invalid_data)
        
        assert result['is_valid'] is False
        assert any('Invalid confidence scores' in error for error in result['errors'])
    
    def test_invalid_portfolio_weights(self):
        """Test validation with invalid portfolio weights"""
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'model_version': ['v1.0'],
            'confidence_score': [0.85],
            'BTCUSDT_weight': [0.7],
            'ETHUSDT_weight': [0.4]  # Weights sum to 1.1
        })
        
        result = validate_model_predictions(invalid_data)
        
        assert result['is_valid'] is False
        assert any("don't sum to 1" in error for error in result['errors'])
    
    def test_valid_portfolio_weights(self):
        """Test validation with valid portfolio weights"""
        valid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'model_version': ['v1.0'],
            'confidence_score': [0.85],
            'BTCUSDT_weight': [0.6],
            'ETHUSDT_weight': [0.4]  # Weights sum to 1.0
        })
        
        result = validate_model_predictions(valid_data)
        
        assert result['is_valid'] is True
        assert len(result['errors']) == 0