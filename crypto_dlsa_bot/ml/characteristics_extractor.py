"""
Cryptocurrency characteristics extraction module for IPCA model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from ..models.data_models import AssetCharacteristics, OHLCVData
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CryptoCharacteristicsExtractor:
    """
    Extracts cryptocurrency characteristics for IPCA model
    """
    
    def __init__(self):
        self.logger = logger
    
    def extract_characteristics(self, 
                             ohlcv_data: pd.DataFrame,
                             symbol: str,
                             timestamp: datetime,
                             lookback_days: int = 180) -> AssetCharacteristics:
        """
        Extract comprehensive characteristics for a cryptocurrency at given timestamp
        
        Args:
            ohlcv_data: OHLCV data DataFrame with columns [timestamp, open, high, low, close, volume]
            symbol: Cryptocurrency symbol
            timestamp: Target timestamp for characteristics extraction
            lookback_days: Number of days to look back for calculations
            
        Returns:
            AssetCharacteristics object
        """
        try:
            # Filter data up to timestamp
            data = ohlcv_data[ohlcv_data['timestamp'] <= timestamp].copy()
            
            if len(data) < 30:  # Minimum data requirement
                raise ValueError(f"Insufficient data for {symbol}: {len(data)} rows")
            
            # Sort by timestamp
            data = data.sort_values('timestamp')
            
            # Calculate returns
            data['returns'] = data['close'].pct_change()
            
            # Get latest values
            latest_close = data['close'].iloc[-1]
            latest_volume = data['volume'].iloc[-1]
            
            # Calculate market cap (using close price as proxy, volume as liquidity measure)
            market_cap = np.log(latest_close * latest_volume + 1)  # Log scale
            
            # Volume ratio (volume to price ratio as liquidity measure)
            volume_ratio = latest_volume / latest_close if latest_close > 0 else 0
            
            # Calculate momentum factors
            momentum_1m = self._calculate_momentum(data, 30)
            momentum_3m = self._calculate_momentum(data, 90)
            momentum_6m = self._calculate_momentum(data, 180)
            
            # Calculate volatility
            volatility_30d = self._calculate_volatility(data, 30)
            
            # Calculate RSI
            rsi = self._calculate_rsi(data, 14)
            
            # Calculate market beta (using simple market proxy)
            beta_market = self._calculate_market_beta(data)
            
            # NVT ratio calculation (simplified version)
            nvt_ratio = self._calculate_nvt_ratio(data)
            
            return AssetCharacteristics(
                timestamp=timestamp,
                symbol=symbol,
                market_cap=market_cap,
                volume_ratio=volume_ratio,
                nvt_ratio=nvt_ratio,
                momentum_1m=momentum_1m,
                momentum_3m=momentum_3m,
                momentum_6m=momentum_6m,
                volatility_30d=volatility_30d,
                rsi=rsi,
                beta_market=beta_market
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting characteristics for {symbol}: {str(e)}")
            raise
    
    def _calculate_momentum(self, data: pd.DataFrame, days: int) -> Optional[float]:
        """Calculate momentum over specified days"""
        try:
            if len(data) < days + 1:
                return None
            
            current_price = data['close'].iloc[-1]
            past_price = data['close'].iloc[-(days + 1)]
            
            if past_price <= 0:
                return None
            
            momentum = (current_price - past_price) / past_price
            return momentum
            
        except Exception:
            return None
    
    def _calculate_volatility(self, data: pd.DataFrame, days: int) -> Optional[float]:
        """Calculate rolling volatility"""
        try:
            if len(data) < days:
                return None
            
            returns = data['returns'].dropna()
            if len(returns) < days:
                return None
            
            volatility = returns.tail(days).std() * np.sqrt(365)  # Annualized
            return volatility
            
        except Exception:
            return None
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        try:
            if len(data) < period + 1:
                return None
            
            returns = data['returns'].dropna()
            if len(returns) < period:
                return None
            
            # Calculate gains and losses
            gains = returns.where(returns > 0, 0)
            losses = -returns.where(returns < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
            
            # Calculate RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None
            
        except Exception:
            return None
    
    def _calculate_market_beta(self, data: pd.DataFrame) -> Optional[float]:
        """Calculate market beta (simplified using own volatility as proxy)"""
        try:
            if len(data) < 60:  # Need at least 60 days
                return None
            
            returns = data['returns'].dropna()
            if len(returns) < 60:
                return None
            
            # Simplified beta calculation using volatility relative to average
            volatility = returns.std()
            avg_volatility = 0.02  # Assumed average market volatility
            
            beta = volatility / avg_volatility if avg_volatility > 0 else 1.0
            return min(max(beta, 0.1), 3.0)  # Cap between 0.1 and 3.0
            
        except Exception:
            return 1.0  # Default to market beta
    
    def _calculate_nvt_ratio(self, data: pd.DataFrame) -> Optional[float]:
        """Calculate Network Value to Transaction ratio (simplified)"""
        try:
            if len(data) < 30:
                return None
            
            # Simplified NVT using price and volume
            recent_data = data.tail(30)
            avg_price = recent_data['close'].mean()
            avg_volume = recent_data['volume'].mean()
            
            if avg_volume <= 0:
                return None
            
            # Simplified NVT ratio
            nvt_ratio = avg_price / avg_volume
            return nvt_ratio
            
        except Exception:
            return None
    
    def extract_batch_characteristics(self,
                                   ohlcv_data: Dict[str, pd.DataFrame],
                                   symbols: List[str],
                                   timestamp: datetime) -> Dict[str, AssetCharacteristics]:
        """
        Extract characteristics for multiple assets at once
        
        Args:
            ohlcv_data: Dictionary mapping symbol to OHLCV DataFrame
            symbols: List of symbols to process
            timestamp: Target timestamp
            
        Returns:
            Dictionary mapping symbol to AssetCharacteristics
        """
        characteristics = {}
        
        for symbol in symbols:
            try:
                if symbol not in ohlcv_data:
                    self.logger.warning(f"No data available for {symbol}")
                    continue
                
                char = self.extract_characteristics(
                    ohlcv_data[symbol], 
                    symbol, 
                    timestamp
                )
                characteristics[symbol] = char
                
            except Exception as e:
                self.logger.error(f"Failed to extract characteristics for {symbol}: {str(e)}")
                continue
        
        self.logger.info(f"Extracted characteristics for {len(characteristics)} assets")
        return characteristics
    
    def validate_characteristics(self, characteristics: AssetCharacteristics) -> bool:
        """
        Validate extracted characteristics for quality
        
        Args:
            characteristics: AssetCharacteristics to validate
            
        Returns:
            True if characteristics are valid, False otherwise
        """
        try:
            # Check for required fields
            if characteristics.market_cap <= 0:
                return False
            
            if characteristics.volume_ratio < 0:
                return False
            
            # Check for reasonable ranges
            if characteristics.rsi is not None:
                if not (0 <= characteristics.rsi <= 100):
                    return False
            
            if characteristics.volatility_30d is not None:
                if characteristics.volatility_30d < 0 or characteristics.volatility_30d > 10:  # 1000% annualized
                    return False
            
            # Check momentum values are reasonable
            momentum_values = [
                characteristics.momentum_1m,
                characteristics.momentum_3m,
                characteristics.momentum_6m
            ]
            
            for momentum in momentum_values:
                if momentum is not None:
                    if abs(momentum) > 10:  # 1000% change seems unreasonable
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating characteristics: {str(e)}")
            return False