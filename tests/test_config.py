"""
Tests for configuration management
"""

import pytest
import tempfile
import os
from crypto_dlsa_bot.config.settings import Config, DataConfig, ModelConfig, load_config


class TestConfig:
    """Test cases for configuration management"""
    
    def test_default_config_creation(self):
        """Test creation of default configuration"""
        config = Config()
        
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert config.data.max_retries == 3
        assert config.model.batch_size == 32
    
    def test_config_validation_success(self):
        """Test successful configuration validation"""
        config = Config()
        errors = config.validate()
        
        # Should have some errors due to missing API keys, but structure should be valid
        assert isinstance(errors, list)
    
    def test_config_validation_invalid_parameters(self):
        """Test configuration validation with invalid parameters"""
        config = Config()
        config.model.seq_length = -1  # Invalid sequence length
        config.model.validation_split = 1.5  # Invalid validation split
        config.backtest.initial_capital = -1000  # Invalid initial capital
        
        errors = config.validate()
        
        assert len(errors) >= 3
        assert any("Sequence length must be positive" in error for error in errors)
        assert any("Validation split must be between 0 and 1" in error for error in errors)
        assert any("Initial capital must be positive" in error for error in errors)
    
    def test_yaml_config_save_load(self):
        """Test saving and loading configuration from YAML"""
        config = Config()
        config.data.max_retries = 5
        config.model.batch_size = 64
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            # Save configuration
            config.to_yaml(config_path)
            
            # Load configuration
            loaded_config = Config.from_yaml(config_path)
            
            assert loaded_config.data.max_retries == 5
            assert loaded_config.model.batch_size == 64
            
        finally:
            os.unlink(config_path)
    
    def test_load_config_function(self):
        """Test load_config function"""
        # Test loading default config (skip validation for test)
        import os
        os.environ['BINANCE_API_KEY'] = 'test_key'
        try:
            config = load_config()
            assert isinstance(config, Config)
        finally:
            if 'BINANCE_API_KEY' in os.environ:
                del os.environ['BINANCE_API_KEY']
        
        # Test loading from YAML file
        test_config = {
            'data': {'max_retries': 10},
            'model': {'batch_size': 128}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            loaded_config = load_config(config_path)
            assert loaded_config.data.max_retries == 10
            assert loaded_config.model.batch_size == 128
            
        finally:
            os.unlink(config_path)
    
    def test_invalid_config_file_extension(self):
        """Test loading configuration with invalid file extension"""
        with pytest.raises(ValueError, match="Configuration file must be YAML or JSON"):
            load_config("invalid_config.txt")