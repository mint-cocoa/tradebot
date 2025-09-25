"""
Configuration settings for Crypto-DLSA Bot
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import yaml
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class DataConfig:
    """Data collection and processing configuration"""
    # Binance configuration
    binance_api_key: str = field(default_factory=lambda: os.getenv('BINANCE_API_KEY', ''))
    binance_secret_key: str = field(default_factory=lambda: os.getenv('BINANCE_SECRET_KEY', ''))
    
    # CoinGecko configuration
    coingecko_api_key: str = field(default_factory=lambda: os.getenv('COINGECKO_API_KEY', ''))
    
    # On-chain data APIs
    glassnode_api_key: str = field(default_factory=lambda: os.getenv('GLASSNODE_API_KEY', ''))
    dune_api_key: str = field(default_factory=lambda: os.getenv('DUNE_API_KEY', ''))
    
    # Data collection settings
    default_symbols: List[str] = field(default_factory=lambda: [
        'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT',
        'LTCUSDT', 'BCHUSDT', 'XLMUSDT', 'EOSUSDT', 'TRXUSDT'
    ])
    default_timeframes: List[str] = field(default_factory=lambda: ['1h', '4h', '1d'])
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.1
    
    # Data storage
    data_directory: str = field(default_factory=lambda: os.getenv('DATA_DIR', './data'))
    use_parquet: bool = True
    compression: str = 'snappy'


@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    # CNN+Transformer architecture
    input_dim: int = 10  # Number of input features
    seq_length: int = 60  # Sequence length for time series
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout_rate: float = 0.1
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    
    # Loss function
    loss_function: str = 'sharpe_ratio'  # 'sharpe_ratio', 'mean_variance'
    risk_free_rate: float = 0.02
    
    # Model paths
    model_directory: str = field(default_factory=lambda: os.getenv('MODEL_DIR', './models'))
    checkpoint_frequency: int = 10  # Save checkpoint every N epochs


@dataclass
class FactorConfig:
    """Factor model configuration"""
    # Factor calculation parameters
    lookback_window: int = 252  # 1 year for daily data
    min_periods: int = 60  # Minimum periods for calculation
    
    # PCA parameters
    n_components: int = 5
    use_incremental_pca: bool = True
    pca_batch_size: int = 1000
    
    # Factor definitions
    factor_names: List[str] = field(default_factory=lambda: [
        'market_factor', 'size_factor', 'momentum_factor', 
        'volatility_factor', 'nvt_factor'
    ])
    
    # Rolling window for residual calculation
    residual_window: int = 252
    min_residual_periods: int = 60


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    # Portfolio parameters
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # Maximum 10% per asset
    min_position_size: float = 0.01  # Minimum 1% per asset
    
    # Rebalancing
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    
    # Transaction costs
    trading_fee_rate: float = 0.001  # 0.1% trading fee
    slippage_rate: float = 0.0005   # 0.05% slippage
    gas_cost_usd: float = 5.0       # $5 gas cost for on-chain trades
    
    # Risk management
    max_leverage: float = 1.0
    stop_loss_threshold: float = -0.05  # 5% stop loss
    max_drawdown_threshold: float = -0.20  # 20% max drawdown
    
    # Performance metrics
    benchmark_symbol: str = 'BTCUSDT'
    risk_free_rate: float = 0.02


@dataclass
class SystemConfig:
    """System-wide configuration"""
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    log_directory: str = field(default_factory=lambda: os.getenv('LOG_DIR', './logs'))
    
    # Database
    database_url: str = field(default_factory=lambda: os.getenv('DATABASE_URL', 'sqlite:///crypto_dlsa.db'))
    
    # Redis cache
    redis_host: str = field(default_factory=lambda: os.getenv('REDIS_HOST', 'localhost'))
    redis_port: int = field(default_factory=lambda: int(os.getenv('REDIS_PORT', '6379')))
    redis_db: int = field(default_factory=lambda: int(os.getenv('REDIS_DB', '0')))
    
    # Performance
    num_workers: int = field(default_factory=lambda: int(os.getenv('NUM_WORKERS', '4')))
    use_gpu: bool = field(default_factory=lambda: os.getenv('USE_GPU', 'true').lower() == 'true')
    gpu_device: str = field(default_factory=lambda: os.getenv('GPU_DEVICE', 'cuda:0'))
    
    # Security
    encrypt_api_keys: bool = True
    encryption_key: Optional[str] = field(default_factory=lambda: os.getenv('ENCRYPTION_KEY'))


@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    factor: FactorConfig = field(default_factory=FactorConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            factor=FactorConfig(**config_dict.get('factor', {})),
            backtest=BacktestConfig(**config_dict.get('backtest', {})),
            system=SystemConfig(**config_dict.get('system', {}))
        )
    
    @classmethod
    def from_json(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            factor=FactorConfig(**config_dict.get('factor', {})),
            backtest=BacktestConfig(**config_dict.get('backtest', {})),
            system=SystemConfig(**config_dict.get('system', {}))
        )
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'factor': self.factor.__dict__,
            'backtest': self.backtest.__dict__,
            'system': self.system.__dict__
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate API keys (only in production mode)
        import os
        if os.getenv('ENVIRONMENT', 'development') == 'production' and not self.data.binance_api_key:
            errors.append("Binance API key is required")
        
        # Validate directories
        for directory in [self.data.data_directory, self.model.model_directory, self.system.log_directory]:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create directory {directory}: {e}")
        
        # Validate model parameters
        if self.model.seq_length <= 0:
            errors.append("Sequence length must be positive")
        
        if not (0 < self.model.validation_split < 1):
            errors.append("Validation split must be between 0 and 1")
        
        # Validate backtest parameters
        if self.backtest.initial_capital <= 0:
            errors.append("Initial capital must be positive")
        
        if not (0 < self.backtest.max_position_size <= 1):
            errors.append("Max position size must be between 0 and 1")
        
        return errors


# Global configuration instance
config = Config()


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or environment variables
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        
    Returns:
        Loaded configuration object
    """
    global config
    
    if config_path:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = Config.from_yaml(config_path)
        elif config_path.endswith('.json'):
            config = Config.from_json(config_path)
        else:
            raise ValueError("Configuration file must be YAML or JSON")
    else:
        # Use default configuration with environment variables
        config = Config()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    return config


def get_config() -> Config:
    """Get current configuration instance"""
    return config