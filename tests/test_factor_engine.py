"""
Unit tests for the Factor Engine module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from crypto_dlsa_bot.ml.factor_engine import (
    CryptoFactorCalculator, 
    FactorEngine, 
    FactorCalculationConfig
)
from crypto_dlsa_bot.models.data_models import FactorExposure, OnChainMetrics


class TestFactorCalculationConfig:
    """Test FactorCalculationConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = FactorCalculationConfig()
        
        assert config.momentum_lookback_days == 21
        assert config.volatility_lookback_days == 30
        assert config.size_percentile_threshold == 0.5
        assert config.min_observations == 10
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = FactorCalculationConfig(
            momentum_lookback_days=14,
            volatility_lookback_days=20,
            size_percentile_threshold=0.3,
            min_observations=5
        )
        
        assert config.momentum_lookback_days == 14
        assert config.volatility_lookback_days == 20
        assert config.size_percentile_threshold == 0.3
        assert config.min_observations == 5


class TestCryptoFactorCalculator:
    """Test CryptoFactorCalculator class"""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        data = []
        np.random.seed(42)  # For reproducible tests
        
        for date in dates:
            for symbol in symbols:
                # Generate realistic crypto price data
                base_price = {'BTCUSDT': 20000, 'ETHUSDT': 1500, 'ADAUSDT': 0.5}[symbol]
                price_change = np.random.normal(0, 0.02)  # 2% daily volatility
                close_price = base_price * (1 + price_change)
                
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'open': close_price * 0.99,
                    'high': close_price * 1.02,
                    'low': close_price * 0.98,
                    'close': close_price,
                    'volume': np.random.uniform(1000000, 10000000)
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_onchain_data(self):
        """Create sample on-chain data for testing"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        data = []
        np.random.seed(42)
        
        for date in dates:
            for symbol in symbols:
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'active_addresses': np.random.randint(100000, 1000000),
                    'tvl': np.random.uniform(1e9, 10e9),
                    'nvt_ratio': np.random.uniform(10, 100),
                    'transaction_count': np.random.randint(100000, 500000)
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance for testing"""
        config = FactorCalculationConfig(min_observations=3)  # Lower for testing
        return CryptoFactorCalculator(config)
    
    def test_calculator_initialization(self):
        """Test calculator initialization"""
        calculator = CryptoFactorCalculator()
        assert calculator.config is not None
        assert isinstance(calculator.config, FactorCalculationConfig)
    
    def test_calculate_market_factor(self, calculator, sample_price_data):
        """Test market factor calculation"""
        market_factor = calculator.calculate_market_factor(sample_price_data)
        
        assert isinstance(market_factor, pd.Series)
        assert len(market_factor) > 0
        assert not market_factor.isna().all()
        
        # Market factor should be the mean return across assets
        # Check that values are reasonable (between -1 and 1 for daily returns)
        assert market_factor.abs().max() < 1.0
    
    def test_calculate_market_factor_invalid_data(self, calculator):
        """Test market factor calculation with invalid data"""
        # Missing required columns
        invalid_data = pd.DataFrame({'timestamp': [datetime.now()], 'symbol': ['BTC']})
        
        with pytest.raises(ValueError):
            calculator.calculate_market_factor(invalid_data)
    
    def test_calculate_size_factor(self, calculator, sample_price_data):
        """Test size factor calculation"""
        size_factor_df = calculator.calculate_size_factor(sample_price_data)
        
        assert isinstance(size_factor_df, pd.DataFrame)
        assert 'timestamp' in size_factor_df.columns
        assert 'symbol' in size_factor_df.columns
        assert 'size_factor' in size_factor_df.columns
        
        # Size factor should be standardized (mean around 0)
        if len(size_factor_df) > 0:
            assert size_factor_df['size_factor'].notna().any()
    
    def test_calculate_momentum_factor(self, calculator, sample_price_data):
        """Test momentum factor calculation"""
        momentum_factor_df = calculator.calculate_momentum_factor(sample_price_data)
        
        assert isinstance(momentum_factor_df, pd.DataFrame)
        assert 'timestamp' in momentum_factor_df.columns
        assert 'symbol' in momentum_factor_df.columns
        assert 'momentum_factor' in momentum_factor_df.columns
        
        # Should have fewer observations due to lookback period
        assert len(momentum_factor_df) <= len(sample_price_data)
    
    def test_calculate_volatility_factor(self, calculator, sample_price_data):
        """Test volatility factor calculation"""
        volatility_factor_df = calculator.calculate_volatility_factor(sample_price_data)
        
        assert isinstance(volatility_factor_df, pd.DataFrame)
        assert 'timestamp' in volatility_factor_df.columns
        assert 'symbol' in volatility_factor_df.columns
        assert 'volatility_factor' in volatility_factor_df.columns
        
        # Should have fewer observations due to rolling window
        assert len(volatility_factor_df) <= len(sample_price_data)
    
    def test_calculate_nvt_factor_with_onchain_data(self, calculator, sample_price_data, sample_onchain_data):
        """Test NVT factor calculation with on-chain data"""
        nvt_factor_df = calculator.calculate_nvt_factor(sample_price_data, sample_onchain_data)
        
        assert isinstance(nvt_factor_df, pd.DataFrame)
        assert 'timestamp' in nvt_factor_df.columns
        assert 'symbol' in nvt_factor_df.columns
        assert 'nvt_factor' in nvt_factor_df.columns
        
        if len(nvt_factor_df) > 0:
            assert nvt_factor_df['nvt_factor'].notna().any()
    
    def test_calculate_nvt_proxy_factor(self, calculator, sample_price_data):
        """Test NVT proxy factor calculation without on-chain data"""
        # Remove transaction_count to trigger proxy calculation
        onchain_data_no_tx = pd.DataFrame({
            'timestamp': sample_price_data['timestamp'].unique(),
            'symbol': 'BTCUSDT'
        })
        
        nvt_factor_df = calculator._calculate_nvt_proxy_factor(sample_price_data)
        
        assert isinstance(nvt_factor_df, pd.DataFrame)
        assert 'nvt_factor' in nvt_factor_df.columns
    
    def test_calculate_all_factors(self, calculator, sample_price_data, sample_onchain_data):
        """Test calculation of all factors together"""
        all_factors_df = calculator.calculate_all_factors(sample_price_data, sample_onchain_data)
        
        assert isinstance(all_factors_df, pd.DataFrame)
        
        expected_columns = [
            'timestamp', 'symbol', 'market_factor', 'size_factor',
            'momentum_factor', 'volatility_factor', 'nvt_factor'
        ]
        
        for col in expected_columns:
            assert col in all_factors_df.columns
        
        # Should have data for multiple symbols and timestamps
        assert all_factors_df['symbol'].nunique() > 1
        assert all_factors_df['timestamp'].nunique() > 1
    
    def test_calculate_all_factors_without_onchain(self, calculator, sample_price_data):
        """Test calculation of all factors without on-chain data"""
        all_factors_df = calculator.calculate_all_factors(sample_price_data, None)
        
        assert isinstance(all_factors_df, pd.DataFrame)
        assert 'nvt_factor' in all_factors_df.columns  # Should use proxy
    
    def test_validate_and_normalize_factors(self, calculator):
        """Test factor validation and normalization"""
        # Create test factor data with some issues
        factor_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='D'),
            'symbol': ['BTC'] * 10,
            'market_factor': [0.01, 0.02, np.nan, 0.05, -0.01, 0.03, 10.0, -5.0, 0.02, 0.01],
            'size_factor': [1.0, -1.0, 0.5, np.nan, 0.2, -0.3, 0.1, 0.4, -0.2, 0.0],
            'momentum_factor': [0.1, -0.1, 0.2, -0.2, np.nan, 0.0, 0.15, -0.15, 0.05, -0.05],
            'volatility_factor': [0.5, 0.3, 0.7, 0.2, 0.8, np.nan, 0.4, 0.6, 0.1, 0.9]
        })
        
        normalized_df = calculator.validate_and_normalize_factors(factor_data)
        
        assert isinstance(normalized_df, pd.DataFrame)
        assert len(normalized_df) == len(factor_data)
        
        # NaN values should be filled with 0
        assert not normalized_df[['market_factor', 'size_factor', 'momentum_factor', 'volatility_factor']].isna().any().any()
        
        # Extreme outliers should be capped
        for col in ['market_factor', 'size_factor', 'momentum_factor', 'volatility_factor']:
            col_data = normalized_df[col]
            mean_val = col_data.mean()
            std_val = col_data.std()
            if std_val > 0:
                assert col_data.max() <= mean_val + 3 * std_val
                assert col_data.min() >= mean_val - 3 * std_val
    
    def test_validate_and_normalize_factors_missing_columns(self, calculator):
        """Test validation with missing required columns"""
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTC']
            # Missing factor columns
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            calculator.validate_and_normalize_factors(invalid_data)
    
    def test_get_factor_value(self, calculator):
        """Test helper method for getting factor values"""
        factor_df = pd.DataFrame({
            'timestamp': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            'symbol': ['BTC', 'ETH'],
            'test_factor': [0.5, -0.3]
        })
        
        # Test existing value
        value = calculator._get_factor_value(
            factor_df, datetime(2023, 1, 1), 'BTC', 'test_factor'
        )
        assert value == 0.5
        
        # Test non-existing value
        value = calculator._get_factor_value(
            factor_df, datetime(2023, 1, 3), 'BTC', 'test_factor'
        )
        assert value is None


class TestFactorEngine:
    """Test FactorEngine class"""
    
    @pytest.fixture
    def factor_engine(self):
        """Create factor engine instance for testing"""
        return FactorEngine()
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        symbols = ['BTCUSDT', 'ETHUSDT']
        
        data = []
        np.random.seed(42)
        
        for date in dates:
            for symbol in symbols:
                base_price = {'BTCUSDT': 20000, 'ETHUSDT': 1500}[symbol]
                close_price = base_price * (1 + np.random.normal(0, 0.02))
                
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'close': close_price,
                    'volume': np.random.uniform(1000000, 10000000)
                })
        
        return pd.DataFrame(data)
    
    def test_factor_engine_initialization(self, factor_engine):
        """Test factor engine initialization"""
        assert factor_engine.calculator is not None
        assert isinstance(factor_engine.calculator, CryptoFactorCalculator)
    
    def test_calculate_market_factors(self, factor_engine, sample_price_data):
        """Test market factors calculation through engine interface"""
        factors_df = factor_engine.calculate_market_factors(sample_price_data)
        
        assert isinstance(factors_df, pd.DataFrame)
        assert len(factors_df) > 0
        
        expected_columns = [
            'timestamp', 'symbol', 'market_factor', 'size_factor',
            'momentum_factor', 'volatility_factor', 'nvt_factor'
        ]
        
        for col in expected_columns:
            assert col in factors_df.columns
    
    def test_fit_factor_model_not_implemented(self, factor_engine):
        """Test that factor model fitting raises NotImplementedError"""
        with pytest.raises(NotImplementedError):
            factor_engine.fit_factor_model(pd.DataFrame(), pd.DataFrame())
    
    def test_calculate_residuals_not_implemented(self, factor_engine):
        """Test that residual calculation raises NotImplementedError"""
        with pytest.raises(NotImplementedError):
            factor_engine.calculate_residuals(None, pd.DataFrame())


class TestIntegration:
    """Integration tests for factor calculation"""
    
    def test_end_to_end_factor_calculation(self):
        """Test complete factor calculation workflow"""
        # Create realistic test data
        dates = pd.date_range(start='2023-01-01', end='2023-02-28', freq='D')
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT']
        
        price_data = []
        np.random.seed(42)
        
        for i, date in enumerate(dates):
            for symbol in symbols:
                base_prices = {
                    'BTCUSDT': 25000, 'ETHUSDT': 1800, 
                    'ADAUSDT': 0.4, 'BNBUSDT': 300
                }
                
                # Add some trend and volatility
                trend = 0.001 * i  # Small upward trend
                volatility = np.random.normal(0, 0.03)
                price = base_prices[symbol] * (1 + trend + volatility)
                
                price_data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'open': price * 0.995,
                    'high': price * 1.02,
                    'low': price * 0.98,
                    'close': price,
                    'volume': np.random.uniform(5000000, 50000000)
                })
        
        price_df = pd.DataFrame(price_data)
        
        # Calculate factors with lower min_observations for testing
        config = FactorCalculationConfig(min_observations=3)
        calculator = CryptoFactorCalculator(config)
        factors_df = calculator.calculate_all_factors(price_df)
        
        # Validate results
        assert len(factors_df) > 0
        assert factors_df['symbol'].nunique() == len(symbols)
        
        # Check that factors have reasonable distributions
        for factor_col in ['market_factor', 'size_factor', 'momentum_factor', 'volatility_factor']:
            if factor_col in factors_df.columns:
                factor_values = factors_df[factor_col].dropna()
                if len(factor_values) > 0:
                    # Factors should be roughly standardized
                    assert factor_values.std() > 0  # Should have some variation
                    assert abs(factor_values.mean()) < 2  # Should be roughly centered
        
        # Validate and normalize
        normalized_df = calculator.validate_and_normalize_factors(factors_df)
        assert len(normalized_df) == len(factors_df)
        assert not normalized_df.isna().any().any()  # No NaN values after normalization


if __name__ == "__main__":
    pytest.main([__file__])