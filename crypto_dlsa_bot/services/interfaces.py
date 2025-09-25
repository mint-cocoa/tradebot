"""
Service interfaces for the Crypto-DLSA Bot system
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd

from crypto_dlsa_bot.models.data_models import (
    OHLCVData, OnChainMetrics, FactorExposure, 
    ResidualData, ModelPrediction, BacktestResult
)


class DataCollectorInterface(ABC):
    """Interface for data collection services"""
    
    @abstractmethod
    def collect_ohlcv(
        self, 
        symbols: List[str], 
        timeframe: str,
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Collect OHLCV data for specified symbols and time range
        
        Args:
            symbols: List of trading symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
            timeframe: Time interval (e.g., '1h', '1d')
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def collect_onchain_metrics(
        self, 
        symbols: List[str],
        metrics: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Collect on-chain metrics data
        
        Args:
            symbols: List of cryptocurrency symbols
            metrics: List of metrics to collect
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with on-chain metrics
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate collected data quality"""
        pass
    
    @abstractmethod
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize and standardize data"""
        pass


class FactorEngineInterface(ABC):
    """Interface for factor calculation and residual generation"""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def fit_factor_model(
        self, 
        returns: pd.DataFrame, 
        characteristics: pd.DataFrame
    ) -> 'FactorModel':
        """
        Fit factor model to return and characteristic data
        
        Args:
            returns: Asset return data
            characteristics: Asset characteristic data
            
        Returns:
            Fitted factor model
        """
        pass
    
    @abstractmethod
    def calculate_residuals(
        self, 
        model: 'FactorModel', 
        returns: pd.DataFrame,
        window_size: int = 252
    ) -> pd.DataFrame:
        """
        Calculate residual returns using rolling window
        
        Args:
            model: Fitted factor model
            returns: Asset return data
            window_size: Rolling window size in periods
            
        Returns:
            DataFrame with residual returns
        """
        pass
    
    @abstractmethod
    def fit_ipca_model(
        self,
        returns: pd.DataFrame,
        characteristics: Optional[pd.DataFrame] = None,
        n_factors: int = 3
    ) -> 'CryptoIPCAModel':
        """
        Fit IPCA (Instrumented PCA) model to cryptocurrency data
        
        Args:
            returns: Asset return data with symbol, date, return columns
            characteristics: Optional asset characteristics data
            n_factors: Number of factors to extract
            
        Returns:
            Fitted IPCA model
        """
        pass
    
    @abstractmethod
    def calculate_ipca_residuals(
        self,
        ipca_model: 'CryptoIPCAModel',
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
        pass


class MLModelInterface(ABC):
    """Interface for machine learning models"""
    
    @abstractmethod
    def train(
        self, 
        features: pd.DataFrame, 
        targets: pd.DataFrame,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the ML model
        
        Args:
            features: Input features (residual returns, factors, etc.)
            targets: Target values (future returns)
            validation_split: Fraction of data for validation
            
        Returns:
            Training metrics dictionary
        """
        pass
    
    @abstractmethod
    def predict(
        self, 
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate predictions from trained model
        
        Args:
            features: Input features for prediction
            
        Returns:
            DataFrame with predictions
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """Save trained model to file"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """Load trained model from file"""
        pass


class BacktestEngineInterface(ABC):
    """Interface for backtesting engine"""
    
    @abstractmethod
    def run_backtest(
        self,
        predictions: pd.DataFrame,
        price_data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
        rebalance_frequency: str = 'daily'
    ) -> BacktestResult:
        """
        Run backtesting simulation
        
        Args:
            predictions: Model predictions with portfolio weights
            price_data: Historical price data
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital amount
            rebalance_frequency: Portfolio rebalancing frequency
            
        Returns:
            Backtesting results
        """
        pass
    
    @abstractmethod
    def calculate_performance_metrics(
        self, 
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Optional benchmark return series
            
        Returns:
            Dictionary of performance metrics
        """
        pass
    
    @abstractmethod
    def calculate_transaction_costs(
        self,
        trades: pd.DataFrame,
        cost_model: 'TransactionCostModel'
    ) -> pd.Series:
        """
        Calculate transaction costs for trades
        
        Args:
            trades: DataFrame with trade information
            cost_model: Transaction cost model
            
        Returns:
            Series with transaction costs
        """
        pass


class RiskManagerInterface(ABC):
    """Interface for risk management"""
    
    @abstractmethod
    def calculate_var(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.05
    ) -> float:
        """Calculate Value at Risk"""
        pass
    
    @abstractmethod
    def calculate_cvar(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.05
    ) -> float:
        """Calculate Conditional Value at Risk"""
        pass
    
    @abstractmethod
    def check_risk_limits(
        self, 
        portfolio_weights: Dict[str, float],
        risk_limits: Dict[str, float]
    ) -> bool:
        """Check if portfolio satisfies risk limits"""
        pass


class DataStorageInterface(ABC):
    """Interface for data storage operations"""
    
    @abstractmethod
    def save_ohlcv_data(self, data: List[OHLCVData]) -> None:
        """Save OHLCV data to storage"""
        pass
    
    @abstractmethod
    def load_ohlcv_data(
        self, 
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Load OHLCV data from storage"""
        pass
    
    @abstractmethod
    def save_factor_data(self, data: List[FactorExposure]) -> None:
        """Save factor exposure data"""
        pass
    
    @abstractmethod
    def load_factor_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Load factor exposure data"""
        pass