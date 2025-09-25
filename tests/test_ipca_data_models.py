"""
Unit tests for IPCA model data structures
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from crypto_dlsa_bot.models.data_models import (
    AssetCharacteristics,
    FactorLoadings,
    GammaMatrix,
    LatentFactors,
    IPCAModelState
)
from crypto_dlsa_bot.ml.characteristics_extractor import CryptoCharacteristicsExtractor


class TestAssetCharacteristics:
    """Test AssetCharacteristics data class"""
    
    def test_valid_characteristics(self):
        """Test creating valid asset characteristics"""
        timestamp = datetime(2023, 1, 1)
        
        char = AssetCharacteristics(
            timestamp=timestamp,
            symbol="BTC",
            market_cap=10.5,  # Log scale
            volume_ratio=0.1,
            nvt_ratio=15.0,
            momentum_1m=0.05,
            momentum_3m=0.12,
            momentum_6m=0.25,
            volatility_30d=0.8,
            rsi=65.0,
            beta_market=1.2
        )
        
        assert char.timestamp == timestamp
        assert char.symbol == "BTC"
        assert char.market_cap == 10.5
        assert char.volume_ratio == 0.1
        assert char.nvt_ratio == 15.0
        assert char.momentum_1m == 0.05
        assert char.rsi == 65.0
    
    def test_to_vector(self):
        """Test converting characteristics to numpy vector"""
        char = AssetCharacteristics(
            timestamp=datetime(2023, 1, 1),
            symbol="BTC",
            market_cap=10.5,
            volume_ratio=0.1,
            nvt_ratio=15.0,
            momentum_1m=0.05,
            momentum_3m=0.12,
            momentum_6m=0.25,
            volatility_30d=0.8,
            rsi=65.0,
            beta_market=1.2
        )
        
        vector = char.to_vector()
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 9  # Number of characteristics
        assert vector[0] == 10.5  # market_cap
        assert vector[1] == 0.1   # volume_ratio
        assert vector[2] == 15.0  # nvt_ratio
        assert vector[8] == 1.2   # beta_market
    
    def test_to_vector_with_none_values(self):
        """Test converting characteristics with None values"""
        char = AssetCharacteristics(
            timestamp=datetime(2023, 1, 1),
            symbol="BTC",
            market_cap=10.5,
            volume_ratio=0.1,
            nvt_ratio=None,
            momentum_1m=None,
            momentum_3m=0.12,
            momentum_6m=None,
            volatility_30d=None,
            rsi=None,
            beta_market=None
        )
        
        vector = char.to_vector()
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 9
        assert vector[0] == 10.5  # market_cap
        assert vector[1] == 0.1   # volume_ratio
        assert vector[2] == 0.0   # nvt_ratio (None -> 0.0)
        assert vector[3] == 0.0   # momentum_1m (None -> 0.0)
        assert vector[4] == 0.12  # momentum_3m
        assert vector[7] == 50.0  # rsi (None -> 50.0)
        assert vector[8] == 1.0   # beta_market (None -> 1.0)
    
    def test_invalid_market_cap(self):
        """Test validation with invalid market cap"""
        with pytest.raises(ValueError, match="Market cap must be positive"):
            AssetCharacteristics(
                timestamp=datetime(2023, 1, 1),
                symbol="BTC",
                market_cap=-1.0,
                volume_ratio=0.1
            )
    
    def test_invalid_volume_ratio(self):
        """Test validation with invalid volume ratio"""
        with pytest.raises(ValueError, match="Volume ratio cannot be negative"):
            AssetCharacteristics(
                timestamp=datetime(2023, 1, 1),
                symbol="BTC",
                market_cap=10.5,
                volume_ratio=-0.1
            )
    
    def test_invalid_rsi(self):
        """Test validation with invalid RSI"""
        with pytest.raises(ValueError, match="RSI must be between 0 and 100"):
            AssetCharacteristics(
                timestamp=datetime(2023, 1, 1),
                symbol="BTC",
                market_cap=10.5,
                volume_ratio=0.1,
                rsi=150.0
            )


class TestFactorLoadings:
    """Test FactorLoadings data class"""
    
    def test_valid_factor_loadings(self):
        """Test creating valid factor loadings"""
        timestamp = datetime(2023, 1, 1)
        loadings = np.array([0.5, -0.3, 0.8])
        
        fl = FactorLoadings(
            timestamp=timestamp,
            symbol="BTC",
            loadings=loadings,
            num_factors=3
        )
        
        assert fl.timestamp == timestamp
        assert fl.symbol == "BTC"
        assert np.array_equal(fl.loadings, loadings)
        assert fl.num_factors == 3
    
    def test_get_loading(self):
        """Test getting specific factor loading"""
        loadings = np.array([0.5, -0.3, 0.8])
        
        fl = FactorLoadings(
            timestamp=datetime(2023, 1, 1),
            symbol="BTC",
            loadings=loadings,
            num_factors=3
        )
        
        assert fl.get_loading(0) == 0.5
        assert fl.get_loading(1) == -0.3
        assert fl.get_loading(2) == 0.8
    
    def test_get_loading_invalid_index(self):
        """Test getting loading with invalid index"""
        loadings = np.array([0.5, -0.3, 0.8])
        
        fl = FactorLoadings(
            timestamp=datetime(2023, 1, 1),
            symbol="BTC",
            loadings=loadings,
            num_factors=3
        )
        
        with pytest.raises(IndexError, match="Factor index 3 out of range"):
            fl.get_loading(3)
        
        with pytest.raises(IndexError, match="Factor index -1 out of range"):
            fl.get_loading(-1)
    
    def test_invalid_loadings_length(self):
        """Test validation with mismatched loadings length"""
        loadings = np.array([0.5, -0.3])  # Length 2
        
        with pytest.raises(ValueError, match="Loadings length 2 != num_factors 3"):
            FactorLoadings(
                timestamp=datetime(2023, 1, 1),
                symbol="BTC",
                loadings=loadings,
                num_factors=3
            )


class TestGammaMatrix:
    """Test GammaMatrix data class"""
    
    def test_valid_gamma_matrix(self):
        """Test creating valid gamma matrix"""
        gamma = np.random.randn(5, 3)  # 5 characteristics, 3 factors
        estimation_date = datetime(2023, 1, 1)
        
        gm = GammaMatrix(
            gamma=gamma,
            num_characteristics=5,
            num_factors=3,
            estimation_date=estimation_date,
            convergence_achieved=True,
            iterations=10
        )
        
        assert np.array_equal(gm.gamma, gamma)
        assert gm.num_characteristics == 5
        assert gm.num_factors == 3
        assert gm.estimation_date == estimation_date
        assert gm.convergence_achieved is True
        assert gm.iterations == 10
    
    def test_compute_factor_loadings(self):
        """Test computing factor loadings β = Γ'z"""
        # Create simple gamma matrix
        gamma = np.array([
            [1.0, 0.0],  # Characteristic 1
            [0.0, 1.0],  # Characteristic 2
            [0.5, 0.5]   # Characteristic 3
        ])
        
        gm = GammaMatrix(
            gamma=gamma,
            num_characteristics=3,
            num_factors=2,
            estimation_date=datetime(2023, 1, 1)
        )
        
        # Test characteristics
        characteristics = np.array([2.0, 3.0, 1.0])
        
        loadings = gm.compute_factor_loadings(characteristics)
        
        # Expected: Γ'z = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]] @ [2.0, 3.0, 1.0]
        #                = [2.0 + 0.5, 3.0 + 0.5] = [2.5, 3.5]
        expected = np.array([2.5, 3.5])
        
        assert np.allclose(loadings, expected)
    
    def test_update_gamma(self):
        """Test updating gamma matrix"""
        initial_gamma = np.random.randn(3, 2)
        
        gm = GammaMatrix(
            gamma=initial_gamma,
            num_characteristics=3,
            num_factors=2,
            estimation_date=datetime(2023, 1, 1)
        )
        
        new_gamma = np.random.randn(3, 2)
        gm.update_gamma(new_gamma, iterations=15, converged=True)
        
        assert np.array_equal(gm.gamma, new_gamma)
        assert gm.iterations == 15
        assert gm.convergence_achieved is True
        # estimation_date should be updated to current time
        assert gm.estimation_date > datetime(2023, 1, 1)
    
    def test_invalid_gamma_shape(self):
        """Test validation with invalid gamma shape"""
        gamma = np.random.randn(3, 2)  # 3x2 matrix
        
        with pytest.raises(ValueError, match="Gamma shape \\(3, 2\\) != expected \\(5, 3\\)"):
            GammaMatrix(
                gamma=gamma,
                num_characteristics=5,
                num_factors=3,
                estimation_date=datetime(2023, 1, 1)
            )
    
    def test_invalid_characteristics_length(self):
        """Test computing loadings with invalid characteristics length"""
        gamma = np.random.randn(3, 2)
        
        gm = GammaMatrix(
            gamma=gamma,
            num_characteristics=3,
            num_factors=2,
            estimation_date=datetime(2023, 1, 1)
        )
        
        # Wrong length characteristics
        characteristics = np.array([1.0, 2.0])  # Length 2, expected 3
        
        with pytest.raises(ValueError, match="Characteristics length 2 != expected 3"):
            gm.compute_factor_loadings(characteristics)


class TestLatentFactors:
    """Test LatentFactors data class"""
    
    def test_valid_latent_factors(self):
        """Test creating valid latent factors"""
        timestamps = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)]
        factors = np.random.randn(5, 3)  # 5 periods, 3 factors
        factor_names = ["Market", "Size", "Momentum"]
        
        lf = LatentFactors(
            timestamps=timestamps,
            factors=factors,
            num_factors=3,
            factor_names=factor_names
        )
        
        assert lf.timestamps == timestamps
        assert np.array_equal(lf.factors, factors)
        assert lf.num_factors == 3
        assert lf.factor_names == factor_names
    
    def test_default_factor_names(self):
        """Test default factor names generation"""
        timestamps = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(3)]
        factors = np.random.randn(3, 2)
        
        lf = LatentFactors(
            timestamps=timestamps,
            factors=factors,
            num_factors=2
        )
        
        assert lf.factor_names == ["Factor_1", "Factor_2"]
    
    def test_get_factor_at_time(self):
        """Test getting factor values at specific timestamp"""
        timestamps = [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)]
        factors = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        
        lf = LatentFactors(
            timestamps=timestamps,
            factors=factors,
            num_factors=2
        )
        
        # Test existing timestamp
        result = lf.get_factor_at_time(datetime(2023, 1, 2))
        expected = np.array([3.0, 4.0])
        assert np.array_equal(result, expected)
        
        # Test non-existing timestamp
        result = lf.get_factor_at_time(datetime(2023, 1, 4))
        assert result is None
    
    def test_get_factor_series(self):
        """Test getting time series for specific factor"""
        timestamps = [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)]
        factors = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        
        lf = LatentFactors(
            timestamps=timestamps,
            factors=factors,
            num_factors=2
        )
        
        # Test factor 0
        series = lf.get_factor_series(0)
        expected = np.array([1.0, 3.0, 5.0])
        assert np.array_equal(series, expected)
        
        # Test factor 1
        series = lf.get_factor_series(1)
        expected = np.array([2.0, 4.0, 6.0])
        assert np.array_equal(series, expected)
    
    def test_get_factor_series_invalid_index(self):
        """Test getting factor series with invalid index"""
        timestamps = [datetime(2023, 1, 1)]
        factors = np.array([[1.0, 2.0]])
        
        lf = LatentFactors(
            timestamps=timestamps,
            factors=factors,
            num_factors=2
        )
        
        with pytest.raises(IndexError, match="Factor index 2 out of range"):
            lf.get_factor_series(2)
    
    def test_to_dataframe(self):
        """Test converting to pandas DataFrame"""
        timestamps = [datetime(2023, 1, 1), datetime(2023, 1, 2)]
        factors = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])
        factor_names = ["Market", "Size"]
        
        lf = LatentFactors(
            timestamps=timestamps,
            factors=factors,
            num_factors=2,
            factor_names=factor_names
        )
        
        df = lf.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["Market", "Size"]
        assert list(df.index) == timestamps
        assert df.iloc[0, 0] == 1.0
        assert df.iloc[1, 1] == 4.0
    
    def test_invalid_timestamps_length(self):
        """Test validation with mismatched timestamps length"""
        timestamps = [datetime(2023, 1, 1), datetime(2023, 1, 2)]  # Length 2
        factors = np.random.randn(3, 2)  # 3 periods
        
        with pytest.raises(ValueError, match="Number of timestamps must match number of periods"):
            LatentFactors(
                timestamps=timestamps,
                factors=factors,
                num_factors=2
            )
    
    def test_invalid_factors_shape(self):
        """Test validation with invalid factors shape"""
        timestamps = [datetime(2023, 1, 1), datetime(2023, 1, 2)]
        factors = np.random.randn(2, 3)  # 3 factors
        
        with pytest.raises(ValueError, match="Factor matrix columns 3 != num_factors 2"):
            LatentFactors(
                timestamps=timestamps,
                factors=factors,
                num_factors=2
            )


class TestIPCAModelState:
    """Test IPCAModelState data class"""
    
    @pytest.fixture
    def sample_gamma_matrix(self):
        """Create sample gamma matrix"""
        gamma = np.random.randn(9, 3)  # 9 characteristics to match AssetCharacteristics.to_vector()
        return GammaMatrix(
            gamma=gamma,
            num_characteristics=9,
            num_factors=3,
            estimation_date=datetime(2023, 1, 1)
        )
    
    @pytest.fixture
    def sample_latent_factors(self):
        """Create sample latent factors"""
        timestamps = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
        factors = np.random.randn(10, 3)
        return LatentFactors(
            timestamps=timestamps,
            factors=factors,
            num_factors=3
        )
    
    def test_valid_model_state(self, sample_gamma_matrix, sample_latent_factors):
        """Test creating valid IPCA model state"""
        model_state = IPCAModelState(
            gamma_matrix=sample_gamma_matrix,
            latent_factors=sample_latent_factors,
            estimation_window_start=datetime(2023, 1, 1),
            estimation_window_end=datetime(2023, 1, 10),
            assets_in_model=["BTC", "ETH", "ADA"],
            model_id="ipca_model_v1"
        )
        
        assert model_state.gamma_matrix == sample_gamma_matrix
        assert model_state.latent_factors == sample_latent_factors
        assert model_state.estimation_window_start == datetime(2023, 1, 1)
        assert model_state.estimation_window_end == datetime(2023, 1, 10)
        assert model_state.assets_in_model == ["BTC", "ETH", "ADA"]
        assert model_state.model_id == "ipca_model_v1"
    
    def test_predict_factor_loadings(self, sample_gamma_matrix, sample_latent_factors):
        """Test predicting factor loadings"""
        model_state = IPCAModelState(
            gamma_matrix=sample_gamma_matrix,
            latent_factors=sample_latent_factors,
            estimation_window_start=datetime(2023, 1, 1),
            estimation_window_end=datetime(2023, 1, 10),
            assets_in_model=["BTC"],
            model_id="test_model"
        )
        
        # Create sample characteristics
        characteristics = AssetCharacteristics(
            timestamp=datetime(2023, 1, 5),
            symbol="BTC",
            market_cap=10.0,
            volume_ratio=0.1,
            nvt_ratio=15.0,
            momentum_1m=0.05,
            momentum_3m=0.1,
            momentum_6m=0.2,
            volatility_30d=0.8,
            rsi=60.0,
            beta_market=1.1
        )
        
        loadings = model_state.predict_factor_loadings(characteristics)
        
        assert isinstance(loadings, FactorLoadings)
        assert loadings.timestamp == characteristics.timestamp
        assert loadings.symbol == characteristics.symbol
        assert loadings.num_factors == 3
        assert len(loadings.loadings) == 3
    
    def test_compute_residual_return(self, sample_gamma_matrix, sample_latent_factors):
        """Test computing residual return"""
        model_state = IPCAModelState(
            gamma_matrix=sample_gamma_matrix,
            latent_factors=sample_latent_factors,
            estimation_window_start=datetime(2023, 1, 1),
            estimation_window_end=datetime(2023, 1, 10),
            assets_in_model=["BTC"],
            model_id="test_model"
        )
        
        # Create sample factor loadings
        loadings = FactorLoadings(
            timestamp=datetime(2023, 1, 5),
            symbol="BTC",
            loadings=np.array([0.5, -0.3, 0.8]),
            num_factors=3
        )
        
        # Sample data
        asset_return = 0.02
        factor_returns = np.array([0.01, -0.005, 0.015])
        
        residual = model_state.compute_residual_return(
            asset_return, loadings, factor_returns
        )
        
        # Expected: 0.02 - (0.5*0.01 + (-0.3)*(-0.005) + 0.8*0.015)
        #         = 0.02 - (0.005 + 0.0015 + 0.012)
        #         = 0.02 - 0.0185 = 0.0015
        expected = 0.02 - (0.5 * 0.01 + (-0.3) * (-0.005) + 0.8 * 0.015)
        
        assert abs(residual - expected) < 1e-10
    
    def test_invalid_factor_consistency(self, sample_latent_factors):
        """Test validation with inconsistent number of factors"""
        # Create gamma matrix with different number of factors
        gamma = np.random.randn(5, 2)  # 2 factors
        gamma_matrix = GammaMatrix(
            gamma=gamma,
            num_characteristics=5,
            num_factors=2,  # Different from latent_factors (3)
            estimation_date=datetime(2023, 1, 1)
        )
        
        with pytest.raises(ValueError, match="Number of factors must be consistent"):
            IPCAModelState(
                gamma_matrix=gamma_matrix,
                latent_factors=sample_latent_factors,  # 3 factors
                estimation_window_start=datetime(2023, 1, 1),
                estimation_window_end=datetime(2023, 1, 10),
                assets_in_model=["BTC"],
                model_id="test_model"
            )
    
    def test_invalid_estimation_window(self, sample_gamma_matrix, sample_latent_factors):
        """Test validation with invalid estimation window"""
        with pytest.raises(ValueError, match="Estimation window start must be before end"):
            IPCAModelState(
                gamma_matrix=sample_gamma_matrix,
                latent_factors=sample_latent_factors,
                estimation_window_start=datetime(2023, 1, 10),
                estimation_window_end=datetime(2023, 1, 1),  # End before start
                assets_in_model=["BTC"],
                model_id="test_model"
            )


class TestCryptoCharacteristicsExtractor:
    """Test CryptoCharacteristicsExtractor"""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Generate sample OHLCV data"""
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        initial_price = 100.0
        returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices)
        
        # Generate OHLC from close prices
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        
        volumes = np.random.uniform(1000, 10000, len(dates))
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })
    
    def test_extract_characteristics(self, sample_ohlcv_data):
        """Test extracting characteristics from OHLCV data"""
        extractor = CryptoCharacteristicsExtractor()
        
        timestamp = datetime(2023, 6, 1)
        symbol = "BTC"
        
        characteristics = extractor.extract_characteristics(
            sample_ohlcv_data, symbol, timestamp
        )
        
        assert isinstance(characteristics, AssetCharacteristics)
        assert characteristics.timestamp == timestamp
        assert characteristics.symbol == symbol
        assert characteristics.market_cap > 0
        assert characteristics.volume_ratio >= 0
        assert characteristics.momentum_1m is not None
        assert characteristics.volatility_30d is not None
        assert characteristics.rsi is not None
        assert 0 <= characteristics.rsi <= 100
    
    def test_extract_characteristics_insufficient_data(self):
        """Test extracting characteristics with insufficient data"""
        extractor = CryptoCharacteristicsExtractor()
        
        # Create minimal data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        })
        
        with pytest.raises(ValueError, match="Insufficient data"):
            extractor.extract_characteristics(data, "BTC", datetime(2023, 1, 10))
    
    def test_validate_characteristics(self, sample_ohlcv_data):
        """Test characteristics validation"""
        extractor = CryptoCharacteristicsExtractor()
        
        # Extract valid characteristics
        characteristics = extractor.extract_characteristics(
            sample_ohlcv_data, "BTC", datetime(2023, 6, 1)
        )
        
        # Should be valid
        assert extractor.validate_characteristics(characteristics) is True
        
        # Test invalid characteristics - create manually without triggering __post_init__
        # We'll test the validation method directly
        valid_char = AssetCharacteristics(
            timestamp=datetime(2023, 1, 1),
            symbol="BTC",
            market_cap=10.0,  # Valid
            volume_ratio=0.1
        )
        
        # Manually set invalid value to test validation
        valid_char.market_cap = -1.0
        assert extractor.validate_characteristics(valid_char) is False
        
        # Test invalid RSI
        valid_char.market_cap = 10.0  # Reset to valid
        valid_char.rsi = 150.0  # Invalid RSI
        assert extractor.validate_characteristics(valid_char) is False


if __name__ == '__main__':
    pytest.main([__file__])