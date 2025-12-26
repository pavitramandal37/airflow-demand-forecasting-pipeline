"""
Tests for common utilities.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / 'scripts'))

from common.config_loader import load_config, merge_configs
from common.metrics import calculate_all_metrics, compare_models
from common.model_versioning import create_model_version, calculate_data_hash


class TestConfigLoader:
    """Test configuration loading."""
    
    def test_load_prophet_config(self):
        """Test loading Prophet configuration."""
        config = load_config('prophet')
        
        assert 'model' in config
        assert config['model']['type'] == 'prophet'
        assert 'prophet' in config['model']
    
    def test_merge_configs(self):
        """Test configuration merging."""
        base = {'a': 1, 'b': {'c': 2}}
        override = {'b': {'d': 3}, 'e': 4}
        
        merged = merge_configs(base, override)
        
        assert merged['a'] == 1
        assert merged['b']['c'] == 2
        assert merged['b']['d'] == 3
        assert merged['e'] == 4


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_calculate_metrics(self):
        """Test metric calculation."""
        y_true = np.array([100, 110, 120, 130, 140])
        y_pred = np.array([102, 108, 125, 128, 142])
        
        metrics = calculate_all_metrics(y_true, y_pred)
        
        assert 'mape' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'smape' in metrics
        
        assert metrics['mape'] > 0
        assert metrics['rmse'] > 0
    
    def test_compare_models(self):
        """Test model comparison."""
        y_true = np.array([100, 110, 120])
        predictions = {
            'model1': np.array([102, 108, 125]),
            'model2': np.array([101, 111, 119])
        }
        
        comparison = compare_models(y_true, predictions)
        
        assert 'model1' in comparison
        assert 'model2' in comparison
        assert 'mape' in comparison['model1']


class TestModelVersioning:
    """Test model versioning."""
    
    def test_create_model_version(self):
        """Test version creation."""
        version = create_model_version('prophet', 'a1b2c3d4', 42)
        
        assert 'prophet' in version
        assert 'a1b2c3d4' in version
        assert 'seed42' in version
    
    def test_calculate_data_hash(self):
        """Test data hash calculation."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        hash1 = calculate_data_hash(df)
        hash2 = calculate_data_hash(df)
        
        # Same data should produce same hash
        assert hash1 == hash2
        
        # Different data should produce different hash
        df2 = pd.DataFrame({'a': [1, 2, 4], 'b': [4, 5, 6]})
        hash3 = calculate_data_hash(df2)
        
        assert hash1 != hash3


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
