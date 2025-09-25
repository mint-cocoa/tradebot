"""
Cryptocurrency Factor Engine Implementation

This module implements factor calculation for cryptocurrency markets,
including Market, Size, NVT Ratio, Momentum, and Volatility factors.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
import joblib
from ipca import InstrumentedPCA
from crypto_dlsa_bot.services.interfaces import FactorEngineInterface
from crypto_dlsa_bot.utils.validation import validate_dataframe, validate_numeric_column
from crypto_dlsa_bot.ml.ipca_preprocessor import IPCAPreprocessorConfig
from .crypto_ipca_model import CryptoIPCAModel as _RealCryptoIPCAModel


logger = logging.getLogger(__name__)


class FactorModel(ABC):
    """
    Abstract base class for factor models
    """
    
    @abstractmethod
    def fit(self, returns: pd.DataFrame, characteristics: Optional[pd.DataFrame] = None) -> 'FactorModel':
        """
        Fit the factor model to return data
        
        Args:
            returns: DataFrame with asset returns (assets as columns, time as rows)
            characteristics: Optional DataFrame with asset characteristics
            
        Returns:
            Fitted factor model
        """
        pass
    
    @abstractmethod
    def transform(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform returns into factor exposures and residuals
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            Tuple of (factor_exposures, residuals)
        """
        pass
    
    @abstractmethod
    def get_factor_loadings(self) -> pd.DataFrame:
        """
        Get factor loadings for each asset
        
        Returns:
            DataFrame with factor loadings
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted model to disk
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> 'FactorModel':
        """
        Load a fitted model from disk
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded factor model
        """
        pass


@dataclass
class FactorModelConfig:
    """Configuration for factor models"""
    n_components: int = 5  # Number of factors to extract
    standardize: bool = True  # Whether to standardize returns before fitting
    min_observations: int = 50  # Minimum observations required for fitting
    explained_variance_threshold: float = 0.8  # Minimum explained variance
    

class PCAFactorModel(FactorModel):
    """
    PCA-based factor model for cryptocurrency returns
    
    Uses Principal Component Analysis to extract common factors from asset returns.
    Suitable for batch processing of historical data.
    """
    
    def __init__(self, config: Optional[FactorModelConfig] = None):
        """
        Initialize PCA factor model
        
        Args:
            config: Configuration for the factor model
        """
        self.config = config or FactorModelConfig()
        self.pca = PCA(n_components=self.config.n_components, random_state=42)
        self.scaler = StandardScaler() if self.config.standardize else None
        self.is_fitted = False
        self.asset_names = None
        self.factor_names = None
        
        logger.info("Initialized PCAFactorModel with %d components", self.config.n_components)
    
    def fit(self, returns: pd.DataFrame, characteristics: Optional[pd.DataFrame] = None) -> 'PCAFactorModel':
        """
        Fit PCA factor model to return data
        
        Args:
            returns: DataFrame with asset returns (time as index, assets as columns)
            characteristics: Optional characteristics (not used in PCA)
            
        Returns:
            Fitted PCA factor model
            
        Raises:
            ValueError: If insufficient data or invalid input
        """
        logger.info("Fitting PCA factor model to returns data")
        
        # Validate input
        if returns.empty:
            raise ValueError("Returns data cannot be empty")
        
        if len(returns) < self.config.min_observations:
            raise ValueError(f"Insufficient observations: {len(returns)} < {self.config.min_observations}")
        
        # Remove columns with all NaN or zero variance
        valid_assets = []
        for col in returns.columns:
            col_data = returns[col].dropna()
            if len(col_data) >= self.config.min_observations and col_data.std() > 1e-8:
                valid_assets.append(col)
        
        if len(valid_assets) < 2:
            raise ValueError("Need at least 2 valid assets for factor model")
        
        # Filter to valid assets and remove NaN rows
        returns_clean = returns[valid_assets].dropna()
        
        if len(returns_clean) < self.config.min_observations:
            raise ValueError(f"Insufficient clean observations: {len(returns_clean)} < {self.config.min_observations}")
        
        self.asset_names = list(returns_clean.columns)
        
        # Standardize if configured
        if self.scaler is not None:
            returns_scaled = self.scaler.fit_transform(returns_clean)
        else:
            returns_scaled = returns_clean.values
        
        # Fit PCA
        self.pca.fit(returns_scaled)
        
        # Generate factor names
        self.factor_names = [f"PC{i+1}" for i in range(self.pca.n_components_)]
        
        # Check explained variance
        total_explained_variance = np.sum(self.pca.explained_variance_ratio_)
        logger.info("PCA explained variance: %.3f", total_explained_variance)
        
        if total_explained_variance < self.config.explained_variance_threshold:
            logger.warning("Low explained variance: %.3f < %.3f", 
                         total_explained_variance, self.config.explained_variance_threshold)
        
        self.is_fitted = True
        logger.info("PCA factor model fitted successfully with %d factors for %d assets", 
                   len(self.factor_names), len(self.asset_names))
        
        return self
    
    def transform(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform returns into factor exposures and residuals
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            Tuple of (factor_exposures, residuals)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transformation")
        
        logger.info("Transforming returns using fitted PCA model")
        
        # Align returns with fitted assets
        if self.asset_names is None:
            raise ValueError("Model must be fitted before transformation")
        common_assets = [asset for asset in self.asset_names if asset in returns.columns]
        if not common_assets:
            raise ValueError("No common assets between fitted model and input returns")
        
        returns_aligned = returns[common_assets].copy()
        
        # Handle missing values by forward filling then backward filling
        returns_aligned = returns_aligned.ffill().bfill()
        
        # Standardize if configured
        if self.scaler is not None:
            returns_scaled = self.scaler.transform(returns_aligned)
        else:
            returns_scaled = returns_aligned.values
        
        # Transform to factor space
        factor_scores = self.pca.transform(returns_scaled)
        
        # Create factor exposures DataFrame
        factor_exposures = pd.DataFrame(
            factor_scores,
            index=returns_aligned.index,
            columns=self.factor_names
        )
        
        # Calculate residuals by reconstructing returns and subtracting from original
        reconstructed = self.pca.inverse_transform(factor_scores)
        
        if self.scaler is not None:
            reconstructed = self.scaler.inverse_transform(reconstructed)
        
        reconstructed_df = pd.DataFrame(
            reconstructed,
            index=returns_aligned.index,
            columns=common_assets
        )
        
        residuals = returns_aligned - reconstructed_df
        
        logger.info("Transformation completed: %d factor exposures, %d residuals", 
                   len(factor_exposures), len(residuals))
        
        return factor_exposures, residuals
    
    def get_factor_loadings(self) -> pd.DataFrame:
        """
        Get factor loadings (components) for each asset
        
        Returns:
            DataFrame with factor loadings (factors as rows, assets as columns)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting loadings")
        
        loadings = pd.DataFrame(
            self.pca.components_,
            index=self.factor_names,
            columns=self.asset_names
        )
        
        return loadings
    
    def get_explained_variance_ratio(self) -> pd.Series:
        """
        Get explained variance ratio for each factor
        
        Returns:
            Series with explained variance ratios
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting explained variance")
        
        return pd.Series(
            self.pca.explained_variance_ratio_,
            index=self.factor_names
        )
    
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted PCA model to disk
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'pca': self.pca,
            'scaler': self.scaler,
            'config': self.config,
            'asset_names': self.asset_names,
            'factor_names': self.factor_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info("PCA model saved to %s", filepath)
    
    def load_model(self, filepath: str) -> 'PCAFactorModel':
        """
        Load a fitted PCA model from disk
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded PCA factor model
        """
        model_data = joblib.load(filepath)
        
        self.pca = model_data['pca']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.asset_names = model_data['asset_names']
        self.factor_names = model_data['factor_names']
        self.is_fitted = model_data['is_fitted']
        
        logger.info("PCA model loaded from %s", filepath)
        return self


class IPCAFactorModel(FactorModel):
    """
    Incremental PCA-based factor model for cryptocurrency returns
    
    Uses Incremental Principal Component Analysis for online learning.
    Suitable for streaming data and large datasets that don't fit in memory.
    """
    
    def __init__(self, config: Optional[FactorModelConfig] = None, batch_size: int = 100):
        """
        Initialize Incremental PCA factor model
        
        Args:
            config: Configuration for the factor model
            batch_size: Batch size for incremental learning
        """
        self.config = config or FactorModelConfig()
        self.batch_size = batch_size
        self.ipca = IncrementalPCA(n_components=self.config.n_components, batch_size=batch_size)
        self.scaler = StandardScaler() if self.config.standardize else None
        self.is_fitted = False
        self.asset_names = None
        self.factor_names = None
        self._n_samples_seen = 0
        
        logger.info("Initialized IPCAFactorModel with %d components, batch_size=%d", 
                   self.config.n_components, batch_size)
    
    def fit(self, returns: pd.DataFrame, characteristics: Optional[pd.DataFrame] = None) -> 'IPCAFactorModel':
        """
        Fit Incremental PCA factor model to return data
        
        Args:
            returns: DataFrame with asset returns (time as index, assets as columns)
            characteristics: Optional characteristics (not used in IPCA)
            
        Returns:
            Fitted IPCA factor model
        """
        logger.info("Fitting Incremental PCA factor model to returns data")
        
        # Validate input
        if returns.empty:
            raise ValueError("Returns data cannot be empty")
        
        # Remove columns with all NaN or zero variance
        valid_assets = []
        for col in returns.columns:
            col_data = returns[col].dropna()
            if len(col_data) >= self.config.min_observations and col_data.std() > 1e-8:
                valid_assets.append(col)
        
        if len(valid_assets) < 2:
            raise ValueError("Need at least 2 valid assets for factor model")
        
        # Filter to valid assets and remove NaN rows
        returns_clean = returns[valid_assets].dropna()
        
        if len(returns_clean) < self.config.min_observations:
            raise ValueError(f"Insufficient clean observations: {len(returns_clean)} < {self.config.min_observations}")
        
        self.asset_names = list(returns_clean.columns)
        
        # Fit scaler if configured
        if self.scaler is not None:
            self.scaler.fit(returns_clean)
            returns_scaled = self.scaler.transform(returns_clean)
        else:
            returns_scaled = returns_clean.values
        
        # Fit IPCA in batches
        n_samples = len(returns_scaled)
        for i in range(0, n_samples, self.batch_size):
            batch_end = min(i + self.batch_size, n_samples)
            batch_data = returns_scaled[i:batch_end]
            
            if len(batch_data) > 0:
                self.ipca.partial_fit(batch_data)
        
        # Generate factor names
        self.factor_names = [f"IPC{i+1}" for i in range(self.ipca.n_components_)]
        
        # Check explained variance
        if hasattr(self.ipca, 'explained_variance_ratio_'):
            total_explained_variance = np.sum(self.ipca.explained_variance_ratio_)
            logger.info("IPCA explained variance: %.3f", total_explained_variance)
            
            if total_explained_variance < self.config.explained_variance_threshold:
                logger.warning("Low explained variance: %.3f < %.3f", 
                             total_explained_variance, self.config.explained_variance_threshold)
        
        self.is_fitted = True
        self._n_samples_seen = n_samples
        
        logger.info("IPCA factor model fitted successfully with %d factors for %d assets", 
                   len(self.factor_names), len(self.asset_names))
        
        return self
    
    def partial_fit(self, returns: pd.DataFrame) -> 'IPCAFactorModel':
        """
        Incrementally fit the model with new data
        
        Args:
            returns: DataFrame with new asset returns
            
        Returns:
            Updated IPCA factor model
        """
        logger.info("Partial fitting IPCA model with %d new observations", len(returns))
        
        if self.asset_names is None:
            # First time fitting
            return self.fit(returns)
        
        # Align returns with existing assets
        common_assets = [asset for asset in self.asset_names if asset in returns.columns]
        if not common_assets:
            logger.warning("No common assets for partial fit, skipping update")
            return self
        
        returns_aligned = returns[common_assets].dropna()
        
        if len(returns_aligned) == 0:
            logger.warning("No valid data for partial fit, skipping update")
            return self
        
        # Standardize if configured
        if self.scaler is not None:
            returns_scaled = self.scaler.transform(returns_aligned)
        else:
            returns_scaled = returns_aligned.values
        
        # Partial fit in batches
        n_samples = len(returns_scaled)
        for i in range(0, n_samples, self.batch_size):
            batch_end = min(i + self.batch_size, n_samples)
            batch_data = returns_scaled[i:batch_end]
            
            if len(batch_data) > 0:
                self.ipca.partial_fit(batch_data)
        
        self._n_samples_seen += n_samples
        logger.info("IPCA model updated with %d samples (total: %d)", n_samples, self._n_samples_seen)
        
        return self
    
    def transform(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform returns into factor exposures and residuals
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            Tuple of (factor_exposures, residuals)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transformation")
        
        logger.info("Transforming returns using fitted IPCA model")
        
        # Align returns with fitted assets
        if self.asset_names is None:
            raise ValueError("Model must be fitted before transformation")
        common_assets = [asset for asset in self.asset_names if asset in returns.columns]
        if not common_assets:
            raise ValueError("No common assets between fitted model and input returns")
        
        returns_aligned = returns[common_assets].copy()
        
        # Handle missing values
        returns_aligned = returns_aligned.ffill().bfill()
        
        # Standardize if configured
        if self.scaler is not None:
            returns_scaled = self.scaler.transform(returns_aligned)
        else:
            returns_scaled = returns_aligned.values
        
        # Transform to factor space
        factor_scores = self.ipca.transform(returns_scaled)
        
        # Create factor exposures DataFrame
        factor_exposures = pd.DataFrame(
            factor_scores,
            index=returns_aligned.index,
            columns=self.factor_names
        )
        
        # Calculate residuals
        reconstructed = self.ipca.inverse_transform(factor_scores)
        
        if self.scaler is not None:
            reconstructed = self.scaler.inverse_transform(reconstructed)
        
        reconstructed_df = pd.DataFrame(
            reconstructed,
            index=returns_aligned.index,
            columns=common_assets
        )
        
        residuals = returns_aligned - reconstructed_df
        
        logger.info("IPCA transformation completed: %d factor exposures, %d residuals", 
                   len(factor_exposures), len(residuals))
        
        return factor_exposures, residuals
    
    def get_factor_loadings(self) -> pd.DataFrame:
        """
        Get factor loadings (components) for each asset
        
        Returns:
            DataFrame with factor loadings (factors as rows, assets as columns)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting loadings")
        
        loadings = pd.DataFrame(
            self.ipca.components_,
            index=self.factor_names,
            columns=self.asset_names
        )
        
        return loadings
    
    def get_explained_variance_ratio(self) -> pd.Series:
        """
        Get explained variance ratio for each factor
        
        Returns:
            Series with explained variance ratios
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting explained variance")
        
        if hasattr(self.ipca, 'explained_variance_ratio_'):
            return pd.Series(
                self.ipca.explained_variance_ratio_,
                index=self.factor_names
            )
        else:
            logger.warning("Explained variance ratio not available for IPCA")
            count = len(self.factor_names) if self.factor_names is not None else 0
            return pd.Series([np.nan] * count, index=self.factor_names if self.factor_names is not None else [])
    
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted IPCA model to disk
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'ipca': self.ipca,
            'scaler': self.scaler,
            'config': self.config,
            'batch_size': self.batch_size,
            'asset_names': self.asset_names,
            'factor_names': self.factor_names,
            'is_fitted': self.is_fitted,
            'n_samples_seen': self._n_samples_seen
        }
        
        joblib.dump(model_data, filepath)
        logger.info("IPCA model saved to %s", filepath)
    
    def load_model(self, filepath: str) -> 'IPCAFactorModel':
        """
        Load a fitted IPCA model from disk
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded IPCA factor model
        """
        model_data = joblib.load(filepath)
        
        self.ipca = model_data['ipca']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.batch_size = model_data['batch_size']
        self.asset_names = model_data['asset_names']
        self.factor_names = model_data['factor_names']
        self.is_fitted = model_data['is_fitted']
        self._n_samples_seen = model_data['n_samples_seen']
        
        logger.info("IPCA model loaded from %s", filepath)
        return self


# Note: The IPCA model implementation is provided in crypto_ipca_model.CryptoIPCAModel.
# For backward compatibility with older constructors that accepted `preprocessor_config`,
# provide a thin wrapper that drops unknown kwargs and delegates to the real class.
class CryptoIPCAModel(_RealCryptoIPCAModel):
    def __init__(self, *args, **kwargs):
        # Discard deprecated/unknown kwargs to maintain compatibility
        kwargs.pop('preprocessor_config', None)
        super().__init__(*args, **kwargs)


@dataclass
class FactorCalculationConfig:
    """Configuration for factor calculations"""
    momentum_lookback_days: int = 21  # 21-day momentum
    volatility_lookback_days: int = 30  # 30-day volatility
    size_percentile_threshold: float = 0.5  # Median split for size factor
    min_observations: int = 10  # Minimum observations for calculation
    

class CryptoFactorCalculator:
    """
    Cryptocurrency-specific factor calculator
    
    Implements calculation of key factors for crypto markets:
    - Market Factor: Overall market return
    - Size Factor: Market capitalization based factor
    - NVT Ratio Factor: Network Value to Transaction ratio
    - Momentum Factor: Price momentum over specified period
    - Volatility Factor: Realized volatility measure
    """
    
    def __init__(self, config: Optional[FactorCalculationConfig] = None):
        """
        Initialize factor calculator
        
        Args:
            config: Configuration for factor calculations
        """
        self.config = config or FactorCalculationConfig()
        logger.info("Initialized CryptoFactorCalculator with config: %s", self.config)
    
    def calculate_market_factor(self, price_data: pd.DataFrame) -> pd.Series:
        """
        Calculate market factor (equal-weighted market return)
        
        Args:
            price_data: DataFrame with columns ['timestamp', 'symbol', 'close', 'volume']
            
        Returns:
            Series with market factor values indexed by timestamp
            
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        logger.info("Calculating market factor for %d symbols", price_data['symbol'].nunique())
        
        # Validate input data
        required_columns = ['timestamp', 'symbol', 'close', 'volume']
        validate_dataframe(price_data, required_columns)
        validate_numeric_column(price_data, 'close')
        validate_numeric_column(price_data, 'volume')
        
        # Calculate returns for each symbol
        price_pivot = price_data.pivot(index='timestamp', columns='symbol', values='close')
        returns = price_pivot.pct_change().dropna()
        
        # Equal-weighted market return
        market_factor = returns.mean(axis=1)
        
        logger.info("Market factor calculated for %d time periods", len(market_factor))
        return market_factor
    
    def calculate_size_factor(self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate size factor based on market capitalization proxy
        
        Since market cap data might not be available, we use price * volume as a proxy
        
        Args:
            price_data: DataFrame with price data
            volume_data: Optional volume data (if not in price_data)
            
        Returns:
            DataFrame with size factor for each symbol and timestamp
        """
        logger.info("Calculating size factor")
        
        # Use volume from price_data if volume_data not provided
        if volume_data is None:
            if 'volume' not in price_data.columns:
                raise ValueError("Volume data required for size factor calculation")
            volume_data = price_data[['timestamp', 'symbol', 'volume']]
        
        # Calculate market cap proxy (price * volume)
        price_pivot = price_data.pivot(index='timestamp', columns='symbol', values='close')
        volume_pivot = volume_data.pivot(index='timestamp', columns='symbol', values='volume')
        
        market_cap_proxy = price_pivot * volume_pivot
        
        # Calculate size factor as log of market cap proxy relative to median
        size_factors = []
        
        for timestamp in market_cap_proxy.index:
            row_data = market_cap_proxy.loc[timestamp].dropna()
            if len(row_data) < self.config.min_observations:
                # If not enough observations for cross-sectional standardization,
                # still include individual values with 0 as size factor
                for symbol in row_data.index:
                    size_factors.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'size_factor': 0.0
                    })
                continue
                
            median_cap = row_data.median()
            if median_cap > 0:
                log_size = np.log(row_data / median_cap)
                
                for symbol in log_size.index:
                    if not np.isnan(log_size[symbol]) and not np.isinf(log_size[symbol]):
                        size_factors.append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'size_factor': log_size[symbol]
                        })
                    else:
                        size_factors.append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'size_factor': 0.0
                        })
            else:
                # If median is 0 or negative, assign 0 to all
                for symbol in row_data.index:
                    size_factors.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'size_factor': 0.0
                    })
        
        result_df = pd.DataFrame(size_factors)
        logger.info("Size factor calculated for %d observations", len(result_df))
        return result_df
    
    def calculate_momentum_factor(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum factor based on past returns
        
        Args:
            price_data: DataFrame with price data
            
        Returns:
            DataFrame with momentum factor for each symbol and timestamp
        """
        logger.info("Calculating momentum factor with %d-day lookback", 
                   self.config.momentum_lookback_days)
        
        price_pivot = price_data.pivot(index='timestamp', columns='symbol', values='close')
        
        # Calculate rolling returns over momentum period
        momentum_returns = price_pivot.pct_change(periods=self.config.momentum_lookback_days)
        
        momentum_factors = []
        
        for timestamp in momentum_returns.index:
            row_data = momentum_returns.loc[timestamp].dropna()
            if len(row_data) < self.config.min_observations:
                continue
            
            # Standardize momentum across assets at each timestamp
            mean_momentum = row_data.mean()
            std_momentum = row_data.std()
            
            if std_momentum > 0:
                standardized_momentum = (row_data - mean_momentum) / std_momentum
                
                for symbol in standardized_momentum.index:
                    momentum_factors.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'momentum_factor': standardized_momentum[symbol]
                    })
        
        result_df = pd.DataFrame(momentum_factors)
        logger.info("Momentum factor calculated for %d observations", len(result_df))
        return result_df
    
    def calculate_volatility_factor(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility factor based on realized volatility
        
        Args:
            price_data: DataFrame with price data
            
        Returns:
            DataFrame with volatility factor for each symbol and timestamp
        """
        logger.info("Calculating volatility factor with %d-day lookback", 
                   self.config.volatility_lookback_days)
        
        price_pivot = price_data.pivot(index='timestamp', columns='symbol', values='close')
        returns = price_pivot.pct_change().dropna()
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=self.config.volatility_lookback_days).std()
        
        volatility_factors = []
        
        for timestamp in rolling_vol.index:
            row_data = rolling_vol.loc[timestamp].dropna()
            if len(row_data) < self.config.min_observations:
                continue
            
            # Standardize volatility across assets
            mean_vol = row_data.mean()
            std_vol = row_data.std()
            
            if std_vol > 0:
                standardized_vol = (row_data - mean_vol) / std_vol
                
                for symbol in standardized_vol.index:
                    volatility_factors.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'volatility_factor': standardized_vol[symbol]
                    })
        
        result_df = pd.DataFrame(volatility_factors)
        logger.info("Volatility factor calculated for %d observations", len(result_df))
        return result_df
    
    def calculate_nvt_factor(self, price_data: pd.DataFrame, onchain_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate NVT (Network Value to Transaction) ratio factor
        
        Args:
            price_data: DataFrame with price data
            onchain_data: DataFrame with on-chain metrics including transaction data
            
        Returns:
            DataFrame with NVT factor for each symbol and timestamp
        """
        logger.info("Calculating NVT factor")
        
        # Validate on-chain data
        if 'transaction_count' not in onchain_data.columns:
            logger.warning("Transaction count not available, using volume as proxy")
            return self._calculate_nvt_proxy_factor(price_data)
        
        # Merge price and on-chain data
        merged_data = pd.merge(
            price_data[['timestamp', 'symbol', 'close', 'volume']],
            onchain_data[['timestamp', 'symbol', 'transaction_count']],
            on=['timestamp', 'symbol'],
            how='inner'
        )
        
        nvt_factors = []
        
        # Group by timestamp and calculate NVT ratios
        for timestamp, group in merged_data.groupby('timestamp'):
            if len(group) < self.config.min_observations:
                continue
            
            # Calculate NVT ratio: Market Cap Proxy / Transaction Volume
            group = group.copy()
            group['market_cap_proxy'] = group['close'] * group['volume']
            group['nvt_ratio'] = group['market_cap_proxy'] / (group['transaction_count'] + 1e-8)
            
            # Standardize NVT ratios
            nvt_values = group['nvt_ratio'].dropna()
            if len(nvt_values) > 1:
                mean_nvt = nvt_values.mean()
                std_nvt = nvt_values.std()
                
                if std_nvt > 0:
                    group['nvt_factor'] = (group['nvt_ratio'] - mean_nvt) / std_nvt
                    
                    for _, row in group.iterrows():
                        if not pd.isna(row['nvt_factor']):
                            nvt_factors.append({
                                'timestamp': timestamp,
                                'symbol': row['symbol'],
                                'nvt_factor': row['nvt_factor']
                            })
        
        result_df = pd.DataFrame(nvt_factors)
        logger.info("NVT factor calculated for %d observations", len(result_df))
        return result_df
    
    def _calculate_nvt_proxy_factor(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate NVT proxy factor using volume data when transaction count unavailable
        
        Args:
            price_data: DataFrame with price and volume data
            
        Returns:
            DataFrame with NVT proxy factor
        """
        logger.info("Calculating NVT proxy factor using volume data")
        
        price_pivot = price_data.pivot(index='timestamp', columns='symbol', values='close')
        volume_pivot = price_data.pivot(index='timestamp', columns='symbol', values='volume')
        
        # Use price/volume ratio as NVT proxy
        nvt_proxy = price_pivot / (volume_pivot + 1e-8)
        
        nvt_factors = []
        
        for timestamp in nvt_proxy.index:
            row_data = nvt_proxy.loc[timestamp].dropna()
            if len(row_data) < self.config.min_observations:
                continue
            
            # Standardize NVT proxy values
            mean_nvt = row_data.mean()
            std_nvt = row_data.std()
            
            if std_nvt > 0:
                standardized_nvt = (row_data - mean_nvt) / std_nvt
                
                for symbol in standardized_nvt.index:
                    nvt_factors.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'nvt_factor': standardized_nvt[symbol]
                    })
        
        result_df = pd.DataFrame(nvt_factors)
        logger.info("NVT proxy factor calculated for %d observations", len(result_df))
        return result_df
    
    def calculate_all_factors(
        self, 
        price_data: pd.DataFrame, 
        onchain_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate all factors for the given data
        
        Args:
            price_data: DataFrame with OHLCV data
            onchain_data: Optional on-chain metrics data
            
        Returns:
            DataFrame with all calculated factors
        """
        logger.info("Calculating all factors for cryptocurrency data")
        
        # Calculate individual factors
        market_factor = self.calculate_market_factor(price_data)
        size_factor_df = self.calculate_size_factor(price_data)
        momentum_factor_df = self.calculate_momentum_factor(price_data)
        volatility_factor_df = self.calculate_volatility_factor(price_data)
        
        # Calculate NVT factor if on-chain data available
        if onchain_data is not None:
            nvt_factor_df = self.calculate_nvt_factor(price_data, onchain_data)
        else:
            nvt_factor_df = self._calculate_nvt_proxy_factor(price_data)
        
        # Merge all factors
        all_factors = []
        
        # Get unique timestamp-symbol combinations from price data
        timestamps = sorted(price_data['timestamp'].unique())
        symbols = sorted(price_data['symbol'].unique())
        
        for timestamp in timestamps:
            # Get market factor value for this timestamp
            market_val = 0.0
            if timestamp in market_factor.index:
                market_val = market_factor.loc[timestamp]
            
            for symbol in symbols:
                # Get factor values for this timestamp-symbol combination
                size_val = self._get_factor_value(size_factor_df, timestamp, symbol, 'size_factor')
                momentum_val = self._get_factor_value(momentum_factor_df, timestamp, symbol, 'momentum_factor')
                volatility_val = self._get_factor_value(volatility_factor_df, timestamp, symbol, 'volatility_factor')
                nvt_val = self._get_factor_value(nvt_factor_df, timestamp, symbol, 'nvt_factor')
                
                # Include all timestamp-symbol combinations from price data
                # Even if some factors are missing, we still want the record
                all_factors.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'market_factor': market_val,
                    'size_factor': size_val or 0.0,
                    'momentum_factor': momentum_val or 0.0,
                    'volatility_factor': volatility_val or 0.0,
                    'nvt_factor': nvt_val or 0.0
                })
        
        result_df = pd.DataFrame(all_factors)
        logger.info("All factors calculated for %d observations across %d symbols", 
                   len(result_df), len(symbols))
        
        return result_df
    
    def _get_factor_value(self, factor_df: pd.DataFrame, timestamp: datetime, 
                         symbol: str, factor_column: str) -> Optional[float]:
        """
        Helper method to get factor value for specific timestamp and symbol
        
        Args:
            factor_df: DataFrame with factor data
            timestamp: Target timestamp
            symbol: Target symbol
            factor_column: Name of the factor column
            
        Returns:
            Factor value or None if not found
        """
        # Check if DataFrame is empty or missing required columns
        if factor_df.empty or 'timestamp' not in factor_df.columns or 'symbol' not in factor_df.columns:
            return None
        
        if factor_column not in factor_df.columns:
            return None
        
        mask = (factor_df['timestamp'] == timestamp) & (factor_df['symbol'] == symbol)
        matching_rows = factor_df[mask]
        
        if len(matching_rows) > 0:
            return matching_rows.iloc[0][factor_column]
        return None
    
    def validate_and_normalize_factors(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and normalize factor data
        
        Args:
            factor_data: DataFrame with calculated factors
            
        Returns:
            Validated and normalized factor DataFrame
            
        Raises:
            ValueError: If factor data is invalid
        """
        logger.info("Validating and normalizing factor data")
        
        # Check required columns
        required_columns = ['timestamp', 'symbol', 'market_factor', 'size_factor', 
                          'momentum_factor', 'volatility_factor']
        missing_columns = [col for col in required_columns if col not in factor_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove rows with all NaN factor values
        factor_columns = ['market_factor', 'size_factor', 'momentum_factor', 
                         'volatility_factor', 'nvt_factor']
        
        # Fill NaN values with 0 for factor columns that exist
        existing_factor_columns = [col for col in factor_columns if col in factor_data.columns]
        factor_data[existing_factor_columns] = factor_data[existing_factor_columns].fillna(0.0)
        
        # Remove extreme outliers (beyond 3 standard deviations)
        for col in existing_factor_columns:
            if col in factor_data.columns:
                mean_val = factor_data[col].mean()
                std_val = factor_data[col].std()
                
                if std_val > 0:
                    # Cap extreme values
                    lower_bound = mean_val - 3 * std_val
                    upper_bound = mean_val + 3 * std_val
                    factor_data[col] = factor_data[col].clip(lower_bound, upper_bound)
        
        # Sort by timestamp and symbol
        factor_data = factor_data.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        
        logger.info("Factor data validated and normalized: %d observations", len(factor_data))
        return factor_data


class FactorEngine(FactorEngineInterface):
    """
    Main factor engine implementing the FactorEngineInterface
    """
    
    def __init__(self, config: Optional[FactorCalculationConfig] = None):
        """
        Initialize factor engine
        
        Args:
            config: Configuration for factor calculations
        """
        self.calculator = CryptoFactorCalculator(config)
        logger.info("FactorEngine initialized")
    
    def calculate_market_factors(
        self, 
        price_data: pd.DataFrame,
        onchain_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate market factors for all assets
        
        Args:
            price_data: OHLCV price data
            onchain_data: Optional on-chain metrics data
            
        Returns:
            DataFrame with calculated factors
        """
        return self.calculator.calculate_all_factors(price_data, onchain_data)
    
    def fit_factor_model(
        self, 
        returns: pd.DataFrame, 
        characteristics: Optional[pd.DataFrame] = None,
        model_type: str = 'pca',
        config: Optional[FactorModelConfig] = None
    ) -> FactorModel:
        """
        Fit factor model to return and characteristic data
        
        Args:
            returns: DataFrame with asset returns (time as index, assets as columns)
            characteristics: Optional DataFrame with asset characteristics
            model_type: Type of factor model ('pca' or 'ipca')
            config: Optional configuration for the factor model
            
        Returns:
            Fitted factor model
            
        Raises:
            ValueError: If invalid model type or insufficient data
        """
        logger.info("Fitting %s factor model to returns data", model_type.upper())
        
        if model_type.lower() == 'pca':
            model = PCAFactorModel(config)
        elif model_type.lower() == 'ipca':
            model = IPCAFactorModel(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'pca' or 'ipca'")
        
        # Fit the model
        fitted_model = model.fit(returns, characteristics)
        
        logger.info("Factor model fitted successfully")
        return fitted_model
    
    def calculate_residuals(
        self, 
        model: 'FactorModel', 
        returns: pd.DataFrame,
        window_size: int = 252
    ) -> pd.DataFrame:
        """
        Calculate residual returns using rolling window
        
        This method will be implemented in the next task (3.3)
        """
        raise NotImplementedError("Residual calculation will be implemented in task 3.3")
    
    def fit_ipca_model(
        self,
        returns: pd.DataFrame,
        characteristics: Optional[pd.DataFrame] = None,
        n_factors: int = 3,
        intercept: bool = False,
        max_iter: int = 1000,
        iter_tol: float = 1e-6,
        alpha: float = 0.0,
        preprocessor_config: Optional[IPCAPreprocessorConfig] = None
    ) -> 'CryptoIPCAModel':
        """
        Fit IPCA (Instrumented PCA) model to cryptocurrency data
        
        Args:
            returns: Asset return data with symbol, date, return columns
            characteristics: Optional asset characteristics data
            n_factors: Number of factors to extract
            intercept: Whether to include intercept
            max_iter: Maximum number of iterations
            iter_tol: Convergence tolerance
            alpha: Regularization parameter
            preprocessor_config: Configuration for data preprocessing
            
        Returns:
            Fitted IPCA model
        """
        logger.info(f"Fitting IPCA model with {n_factors} factors")
        
        # Create IPCA model
        ipca_model = CryptoIPCAModel(
            n_factors=n_factors,
            intercept=intercept,
            max_iter=max_iter,
            iter_tol=iter_tol,
            alpha=alpha
        )
        
        # Fit the model
        ipca_model.fit(returns, characteristics)
        
        logger.info("IPCA model fitted successfully")
        return ipca_model
    
    def calculate_ipca_residuals(
        self,
        ipca_model: CryptoIPCAModel,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate residuals using fitted IPCA model
        
        Args:
            ipca_model: Fitted IPCA model
            returns: Asset return data
            
        Returns:
            DataFrame with residual returns
        """
        logger.info("Calculating IPCA residuals")
        
        if not ipca_model.is_fitted:
            raise ValueError("IPCA model must be fitted before calculating residuals")
        
        # Calculate residuals using the model
        residuals_df = ipca_model.calculate_residuals(returns)
        
        logger.info(f"IPCA residuals calculated for {len(residuals_df)} observations")
        return residuals_df

