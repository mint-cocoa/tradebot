"""
Unit tests for PCA and IPCA factor models
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os

from crypto_dlsa_bot.ml.factor_engine import (
    PCAFactorModel, 
    IPCAFactorModel, 
    FactorModelConfig,
    FactorEngine
)


class TestFactorModelConfig:
    """Test FactorModelConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = FactorModelConfig()
        
        assert config.n_components == 5
        assert config.standardize is True
        assert config.min_observations == 50
        assert config.explained_variance_threshold == 0.8
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = FactorModelConfig(
            n_components=3,
            standardize=False,
            min_observations=30,
            explained_variance_threshold=0.7
        )
        
        assert config.n_components == 3
        assert config.standardize is False
        assert config.min_observations == 30
        assert config.explained_variance_threshold == 0.7


class TestPCAFactorModel:
    """Test PCA factor model implementation"""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        assets = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']
        
        # Generate correlated returns
        n_factors = 2
        factor_loadings = np.random.randn(len(assets), n_factors)
        factor_returns = np.random.randn(len(dates), n_factors) * 0.02
        idiosyncratic = np.random.randn(len(dates), len(assets)) * 0.01
        
        returns = factor_returns @ factor_loadings.T + idiosyncratic
        
        return pd.DataFrame(returns, index=dates, columns=assets)
    
    @pytest.fixture
    def small_returns(self):
        """Generate small sample for testing edge cases"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        assets = ['BTC', 'ETH']
        
        returns = np.random.randn(len(dates), len(assets)) * 0.02
        return pd.DataFrame(returns, index=dates, columns=assets)
    
    def test_initialization(self):
        """Test PCA model initialization"""
        model = PCAFactorModel()
        
        assert model.config.n_components == 5
        assert model.is_fitted is False
        assert model.asset_names is None
        assert model.factor_names is None
    
    def test_initialization_with_config(self):
        """Test PCA model initialization with custom config"""
        config = FactorModelConfig(n_components=3, standardize=False)
        model = PCAFactorModel(config)
        
        assert model.config.n_components == 3
        assert model.config.standardize is False
        assert model.scaler is None
    
    def test_fit_valid_data(self, sample_returns):
        """Test fitting PCA model with valid data"""
        config = FactorModelConfig(n_components=2, min_observations=50)
        model = PCAFactorModel(config)
        
        fitted_model = model.fit(sample_returns)
        
        assert fitted_model.is_fitted is True
        assert len(fitted_model.asset_names) == 5
        assert len(fitted_model.factor_names) == 2
        assert fitted_model.factor_names == ['PC1', 'PC2']
    
    def test_fit_insufficient_data(self, small_returns):
        """Test fitting with insufficient data"""
        config = FactorModelConfig(min_observations=50)
        model = PCAFactorModel(config)
        
        with pytest.raises(ValueError, match="Insufficient observations"):
            model.fit(small_returns)
    
    def test_fit_empty_data(self):
        """Test fitting with empty data"""
        model = PCAFactorModel()
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Returns data cannot be empty"):
            model.fit(empty_df)
    
    def test_fit_single_asset(self):
        """Test fitting with single asset (should fail)"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        returns = pd.DataFrame(
            np.random.randn(100, 1) * 0.02,
            index=dates,
            columns=['BTC']
        )
        
        model = PCAFactorModel()
        
        with pytest.raises(ValueError, match="Need at least 2 valid assets"):
            model.fit(returns)
    
    def test_transform_before_fit(self, sample_returns):
        """Test transform before fitting (should fail)"""
        model = PCAFactorModel()
        
        with pytest.raises(ValueError, match="Model must be fitted before transformation"):
            model.transform(sample_returns)
    
    def test_transform_after_fit(self, sample_returns):
        """Test transform after fitting"""
        config = FactorModelConfig(n_components=2, min_observations=50)
        model = PCAFactorModel(config)
        model.fit(sample_returns)
        
        factor_exposures, residuals = model.transform(sample_returns)
        
        assert isinstance(factor_exposures, pd.DataFrame)
        assert isinstance(residuals, pd.DataFrame)
        assert factor_exposures.shape == (100, 2)  # 100 dates, 2 factors
        assert residuals.shape == (100, 5)  # 100 dates, 5 assets
        assert list(factor_exposures.columns) == ['PC1', 'PC2']
        assert list(residuals.columns) == sample_returns.columns.tolist()
    
    def test_get_factor_loadings(self, sample_returns):
        """Test getting factor loadings"""
        config = FactorModelConfig(n_components=2, min_observations=50)
        model = PCAFactorModel(config)
        model.fit(sample_returns)
        
        loadings = model.get_factor_loadings()
        
        assert isinstance(loadings, pd.DataFrame)
        assert loadings.shape == (2, 5)  # 2 factors, 5 assets
        assert list(loadings.index) == ['PC1', 'PC2']
        assert list(loadings.columns) == sample_returns.columns.tolist()
    
    def test_get_explained_variance_ratio(self, sample_returns):
        """Test getting explained variance ratio"""
        config = FactorModelConfig(n_components=2, min_observations=50)
        model = PCAFactorModel(config)
        model.fit(sample_returns)
        
        explained_var = model.get_explained_variance_ratio()
        
        assert isinstance(explained_var, pd.Series)
        assert len(explained_var) == 2
        assert list(explained_var.index) == ['PC1', 'PC2']
        assert all(explained_var >= 0)
        assert all(explained_var <= 1)
    
    def test_save_and_load_model(self, sample_returns):
        """Test saving and loading model"""
        config = FactorModelConfig(n_components=2, min_observations=50)
        model = PCAFactorModel(config)
        model.fit(sample_returns)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            model.save_model(tmp_file.name)
            
            # Load model
            new_model = PCAFactorModel()
            loaded_model = new_model.load_model(tmp_file.name)
            
            assert loaded_model.is_fitted is True
            assert loaded_model.asset_names == model.asset_names
            assert loaded_model.factor_names == model.factor_names
            
            # Test that loaded model can transform
            factor_exposures, residuals = loaded_model.transform(sample_returns)
            assert factor_exposures.shape == (100, 2)
            assert residuals.shape == (100, 5)
            
            # Clean up
            os.unlink(tmp_file.name)
    
    def test_save_unfitted_model(self):
        """Test saving unfitted model (should fail)"""
        model = PCAFactorModel()
        
        with pytest.raises(ValueError, match="Cannot save unfitted model"):
            model.save_model("dummy_path.pkl")


class TestIPCAFactorModel:
    """Test Incremental PCA factor model implementation"""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        assets = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']
        
        # Generate correlated returns
        n_factors = 2
        factor_loadings = np.random.randn(len(assets), n_factors)
        factor_returns = np.random.randn(len(dates), n_factors) * 0.02
        idiosyncratic = np.random.randn(len(dates), len(assets)) * 0.01
        
        returns = factor_returns @ factor_loadings.T + idiosyncratic
        
        return pd.DataFrame(returns, index=dates, columns=assets)
    
    def test_initialization(self):
        """Test IPCA model initialization"""
        model = IPCAFactorModel()
        
        assert model.config.n_components == 5
        assert model.batch_size == 100
        assert model.is_fitted is False
        assert model.asset_names is None
        assert model.factor_names is None
        assert model._n_samples_seen == 0
    
    def test_initialization_with_batch_size(self):
        """Test IPCA model initialization with custom batch size"""
        model = IPCAFactorModel(batch_size=50)
        
        assert model.batch_size == 50
    
    def test_fit_valid_data(self, sample_returns):
        """Test fitting IPCA model with valid data"""
        config = FactorModelConfig(n_components=2, min_observations=50)
        model = IPCAFactorModel(config, batch_size=25)
        
        fitted_model = model.fit(sample_returns)
        
        assert fitted_model.is_fitted is True
        assert len(fitted_model.asset_names) == 5
        assert len(fitted_model.factor_names) == 2
        assert fitted_model.factor_names == ['IPC1', 'IPC2']
        assert fitted_model._n_samples_seen == 100
    
    def test_partial_fit(self, sample_returns):
        """Test partial fitting with new data"""
        config = FactorModelConfig(n_components=2, min_observations=50)
        model = IPCAFactorModel(config, batch_size=25)
        
        # Initial fit
        initial_data = sample_returns.iloc[:60]
        model.fit(initial_data)
        initial_samples = model._n_samples_seen
        
        # Partial fit with new data
        new_data = sample_returns.iloc[60:]
        model.partial_fit(new_data)
        
        assert model._n_samples_seen == initial_samples + len(new_data)
        assert model.is_fitted is True
    
    def test_transform_after_fit(self, sample_returns):
        """Test transform after fitting"""
        config = FactorModelConfig(n_components=2, min_observations=50)
        model = IPCAFactorModel(config, batch_size=25)
        model.fit(sample_returns)
        
        factor_exposures, residuals = model.transform(sample_returns)
        
        assert isinstance(factor_exposures, pd.DataFrame)
        assert isinstance(residuals, pd.DataFrame)
        assert factor_exposures.shape == (100, 2)  # 100 dates, 2 factors
        assert residuals.shape == (100, 5)  # 100 dates, 5 assets
        assert list(factor_exposures.columns) == ['IPC1', 'IPC2']
    
    def test_get_factor_loadings(self, sample_returns):
        """Test getting factor loadings"""
        config = FactorModelConfig(n_components=2, min_observations=50)
        model = IPCAFactorModel(config, batch_size=25)
        model.fit(sample_returns)
        
        loadings = model.get_factor_loadings()
        
        assert isinstance(loadings, pd.DataFrame)
        assert loadings.shape == (2, 5)  # 2 factors, 5 assets
        assert list(loadings.index) == ['IPC1', 'IPC2']
        assert list(loadings.columns) == sample_returns.columns.tolist()
    
    def test_save_and_load_model(self, sample_returns):
        """Test saving and loading IPCA model"""
        config = FactorModelConfig(n_components=2, min_observations=50)
        model = IPCAFactorModel(config, batch_size=25)
        model.fit(sample_returns)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            model.save_model(tmp_file.name)
            
            # Load model
            new_model = IPCAFactorModel()
            loaded_model = new_model.load_model(tmp_file.name)
            
            assert loaded_model.is_fitted is True
            assert loaded_model.asset_names == model.asset_names
            assert loaded_model.factor_names == model.factor_names
            assert loaded_model.batch_size == model.batch_size
            assert loaded_model._n_samples_seen == model._n_samples_seen
            
            # Clean up
            os.unlink(tmp_file.name)


class TestFactorEngine:
    """Test FactorEngine integration with factor models"""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        assets = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']
        
        returns = np.random.randn(len(dates), len(assets)) * 0.02
        return pd.DataFrame(returns, index=dates, columns=assets)
    
    def test_fit_pca_model(self, sample_returns):
        """Test fitting PCA model through FactorEngine"""
        engine = FactorEngine()
        config = FactorModelConfig(n_components=2, min_observations=50)
        
        model = engine.fit_factor_model(sample_returns, model_type='pca', config=config)
        
        assert isinstance(model, PCAFactorModel)
        assert model.is_fitted is True
        assert len(model.factor_names) == 2
    
    def test_fit_ipca_model(self, sample_returns):
        """Test fitting IPCA model through FactorEngine"""
        engine = FactorEngine()
        config = FactorModelConfig(n_components=2, min_observations=50)
        
        model = engine.fit_factor_model(sample_returns, model_type='ipca', config=config)
        
        assert isinstance(model, IPCAFactorModel)
        assert model.is_fitted is True
        assert len(model.factor_names) == 2
    
    def test_fit_invalid_model_type(self, sample_returns):
        """Test fitting with invalid model type"""
        engine = FactorEngine()
        
        with pytest.raises(ValueError, match="Unsupported model type"):
            engine.fit_factor_model(sample_returns, model_type='invalid')
    
    def test_fit_default_pca_model(self, sample_returns):
        """Test fitting with default PCA model"""
        engine = FactorEngine()
        config = FactorModelConfig(min_observations=50)
        
        model = engine.fit_factor_model(sample_returns, config=config)
        
        assert isinstance(model, PCAFactorModel)
        assert model.is_fitted is True


if __name__ == '__main__':
    pytest.main([__file__])