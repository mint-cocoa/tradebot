"""
Data validation utilities
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional
from datetime import datetime


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate that DataFrame has required columns
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Raises:
        ValueError: If DataFrame is invalid or missing required columns
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def validate_numeric_column(df: pd.DataFrame, column: str) -> None:
    """
    Validate that a column contains numeric data
    
    Args:
        df: DataFrame containing the column
        column: Name of the column to validate
        
    Raises:
        ValueError: If column is not numeric or contains invalid values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must contain numeric data")
    
    # Check for infinite values
    if np.isinf(df[column]).any():
        raise ValueError(f"Column '{column}' contains infinite values")


def validate_ohlcv_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate OHLCV data quality
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        results['is_valid'] = False
        results['errors'].append(f"Missing required columns: {missing_columns}")
        return results
    
    # Check for null values
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        results['warnings'].append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
    
    # Validate OHLC relationships
    invalid_ohlc = df[
        (df['high'] < df[['open', 'close']].max(axis=1)) |
        (df['low'] > df[['open', 'close']].min(axis=1))
    ]
    
    if len(invalid_ohlc) > 0:
        results['errors'].append(f"Invalid OHLC relationships found in {len(invalid_ohlc)} rows")
        results['is_valid'] = False
    
    # Check for negative volumes
    negative_volume = df[df['volume'] < 0]
    if len(negative_volume) > 0:
        results['errors'].append(f"Negative volume found in {len(negative_volume)} rows")
        results['is_valid'] = False
    
    # Check for extreme price movements (>50% in single period)
    df_sorted = df.sort_values(['symbol', 'timestamp'])
    df_sorted['price_change'] = df_sorted.groupby('symbol')['close'].pct_change()
    extreme_moves = df_sorted[abs(df_sorted['price_change']) > 0.5]
    
    if len(extreme_moves) > 0:
        results['warnings'].append(f"Extreme price movements (>50%) found in {len(extreme_moves)} rows")
    
    # Calculate statistics
    results['stats'] = {
        'total_rows': len(df),
        'unique_symbols': df['symbol'].nunique(),
        'date_range': {
            'start': df['timestamp'].min(),
            'end': df['timestamp'].max()
        },
        'null_percentage': (df[required_columns].isnull().sum().sum() / (len(df) * len(required_columns))) * 100
    }
    
    return results


def validate_factor_data(df: pd.DataFrame, factor_names: List[str]) -> Dict[str, Any]:
    """
    Validate factor exposure data
    
    Args:
        df: DataFrame with factor data
        factor_names: List of expected factor names
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    required_columns = ['timestamp', 'symbol'] + factor_names
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        results['is_valid'] = False
        results['errors'].append(f"Missing required columns: {missing_columns}")
        return results
    
    # Check for infinite or NaN values in factors
    factor_df = df[factor_names]
    
    inf_counts = np.isinf(factor_df).sum()
    if inf_counts.any():
        results['errors'].append(f"Infinite values found in factors: {inf_counts[inf_counts > 0].to_dict()}")
        results['is_valid'] = False
    
    nan_counts = factor_df.isnull().sum()
    if nan_counts.any():
        results['warnings'].append(f"NaN values found in factors: {nan_counts[nan_counts > 0].to_dict()}")
    
    # Check factor value ranges (should be reasonable)
    for factor in factor_names:
        factor_values = df[factor].dropna()
        if len(factor_values) > 0:
            if abs(factor_values).max() > 100:  # Arbitrary threshold
                results['warnings'].append(f"Extreme values in {factor}: max={factor_values.max():.2f}, min={factor_values.min():.2f}")
    
    # Calculate statistics
    results['stats'] = {
        'total_rows': len(df),
        'unique_symbols': df['symbol'].nunique(),
        'factor_correlations': factor_df.corr().to_dict(),
        'factor_means': factor_df.mean().to_dict(),
        'factor_stds': factor_df.std().to_dict()
    }
    
    return results


def validate_model_predictions(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate model prediction data
    
    Args:
        df: DataFrame with model predictions
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    required_columns = ['timestamp', 'model_version', 'confidence_score']
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        results['is_valid'] = False
        results['errors'].append(f"Missing required columns: {missing_columns}")
        return results
    
    # Check confidence scores are between 0 and 1
    invalid_confidence = df[
        (df['confidence_score'] < 0) | (df['confidence_score'] > 1)
    ]
    
    if len(invalid_confidence) > 0:
        results['errors'].append(f"Invalid confidence scores found in {len(invalid_confidence)} rows")
        results['is_valid'] = False
    
    # Check for portfolio weight columns
    weight_columns = [col for col in df.columns if col.endswith('_weight')]
    if weight_columns:
        # Check if weights sum to approximately 1 for each row
        weight_sums = df[weight_columns].sum(axis=1)
        invalid_weights = abs(weight_sums - 1.0) > 1e-6
        
        if invalid_weights.any():
            results['errors'].append(f"Portfolio weights don't sum to 1 in {invalid_weights.sum()} rows")
            results['is_valid'] = False
    
    # Calculate statistics
    results['stats'] = {
        'total_predictions': len(df),
        'unique_models': df['model_version'].nunique(),
        'confidence_stats': {
            'mean': df['confidence_score'].mean(),
            'std': df['confidence_score'].std(),
            'min': df['confidence_score'].min(),
            'max': df['confidence_score'].max()
        }
    }
    
    return results


def validate_symbol(symbol: str) -> bool:
    """
    Validate trading symbol format
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(symbol, str):
        return False
    
    # Basic format check: should be uppercase letters/numbers, typically ending with USDT/BTC/ETH
    pattern = r'^[A-Z0-9]+$'
    if not re.match(pattern, symbol):
        return False
    
    # Should be at least 6 characters (e.g., BTCUSDT)
    if len(symbol) < 6:
        return False
    
    # Common quote currencies
    valid_quotes = ['USDT', 'BTC', 'ETH', 'BNB', 'BUSD', 'USDC']
    if not any(symbol.endswith(quote) for quote in valid_quotes):
        return False
    
    return True


def validate_timeframe(timeframe: str) -> bool:
    """
    Validate timeframe format
    
    Args:
        timeframe: Timeframe string (e.g., '1h', '1d')
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(timeframe, str):
        return False
    
    # Valid timeframes for Binance
    valid_timeframes = [
        '1m', '3m', '5m', '15m', '30m',
        '1h', '2h', '4h', '6h', '8h', '12h',
        '1d', '3d', '1w', '1M'
    ]
    
    return timeframe in valid_timeframes


def validate_date_range(start_date: datetime, end_date: datetime) -> Dict[str, Any]:
    """
    Validate date range for data collection
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if start_date is before end_date
    if start_date >= end_date:
        results['is_valid'] = False
        results['errors'].append("Start date must be before end date")
    
    # Check if dates are not too far in the future
    now = datetime.now()
    if start_date > now:
        results['is_valid'] = False
        results['errors'].append("Start date cannot be in the future")
    
    if end_date > now:
        results['warnings'].append("End date is in the future, will be adjusted to current time")
    
    # Check if date range is reasonable (not too large)
    date_diff = end_date - start_date
    if date_diff.days > 365 * 5:  # 5 years
        results['warnings'].append("Date range is very large (>5 years), consider splitting the request")
    
    # Check if dates are not too old (Binance data availability)
    binance_start = datetime(2017, 7, 14)  # Binance launch date
    if start_date < binance_start:
        results['warnings'].append(f"Start date is before Binance launch ({binance_start})")
    
    return results