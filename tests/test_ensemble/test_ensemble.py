"""
Tests for ensemble combiner.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / 'scripts'))

from ensemble.model_combiner import EnsembleCombiner


class TestEnsembleCombiner:
    """Test ensemble combination logic."""
    
    def test_inverse_error_weights(self):
        """Test inverse error weight calculation."""
        combiner = EnsembleCombiner()
        
        predictions = {
            'model1': np.array([100, 110, 120]),
            'model2': np.array([102, 108, 125]),
            'model3': np.array([101, 111, 119])
        }
        
        y_true = np.array([100, 110, 120])
        
        weights = combiner.calculate_weights_inverse_error(predictions, y_true)
        
        # Weights should sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        
        # All weights should be positive
        assert all(w > 0 for w in weights.values())
        
        # Model with lowest error should have highest weight
        # (model1 has perfect predictions)
        assert weights['model1'] > weights['model2']
        assert weights['model1'] > weights['model3']
    
    def test_optimize_weights(self):
        """Test weight optimization."""
        combiner = EnsembleCombiner()
        
        predictions = {
            'model1': np.array([100, 110, 120, 130, 140]),
            'model2': np.array([102, 108, 125, 128, 142])
        }
        
        y_true = np.array([100, 110, 120, 130, 140])
        
        weights = combiner.optimize_weights(predictions, y_true)
        
        # Weights should sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        
        # All weights should be between 0 and 1
        assert all(0 <= w <= 1 for w in weights.values())
    
    def test_combine_predictions(self):
        """Test prediction combination."""
        combiner = EnsembleCombiner()
        
        # Create sample predictions
        dates = pd.date_range('2024-01-01', periods=5)
        
        predictions = {
            'model1': pd.DataFrame({'ds': dates, 'forecast': [100, 110, 120, 130, 140]}),
            'model2': pd.DataFrame({'ds': dates, 'forecast': [102, 108, 125, 128, 142]})
        }
        
        weights = {'model1': 0.6, 'model2': 0.4}
        
        ensemble = combiner.combine_predictions(predictions, weights)
        
        # Check output structure
        assert 'ds' in ensemble.columns
        assert 'ensemble_forecast' in ensemble.columns
        assert len(ensemble) == 5
        
        # Check weighted average calculation
        expected_first = 0.6 * 100 + 0.4 * 102
        assert abs(ensemble['ensemble_forecast'].iloc[0] - expected_first) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
