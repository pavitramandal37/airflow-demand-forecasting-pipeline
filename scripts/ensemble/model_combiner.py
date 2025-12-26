"""
Ensemble model combiner with weighted average and optimization.

Combines predictions from Prophet, SARIMA, and DeepAR models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from scipy.optimize import minimize

import sys
sys.path.append(str(Path(__file__).parent.parent))

from common.config_loader import load_config
from common.metrics import calculate_all_metrics, compare_models, get_best_model
from common.utils import setup_logging

logger = logging.getLogger(__name__)


class EnsembleCombiner:
    """
    Ensemble model combiner with multiple strategies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ensemble combiner.
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = load_config('ensemble')
        
        self.config = config
        self.ensemble_config = config['ensemble']
        self.strategy = self.ensemble_config.get('strategy', 'weighted_average')
        
        self.weights = {}
        self.comparison_results = {}
    
    def load_predictions(
        self,
        prophet_path: Optional[Path] = None,
        sarima_path: Optional[Path] = None,
        deepar_path: Optional[Path] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load predictions from all models.
        
        Args:
            prophet_path: Path to Prophet predictions
            sarima_path: Path to SARIMA predictions
            deepar_path: Path to DeepAR predictions
        
        Returns:
            Dictionary of model predictions
        """
        predictions = {}
        
        models_config = self.ensemble_config.get('models', {})
        
        # Load Prophet
        if models_config.get('prophet', {}).get('enabled', True) and prophet_path:
            logger.info(f"Loading Prophet predictions from {prophet_path}")
            predictions['prophet'] = pd.read_csv(prophet_path)
        
        # Load SARIMA
        if models_config.get('sarima', {}).get('enabled', True) and sarima_path:
            logger.info(f"Loading SARIMA predictions from {sarima_path}")
            predictions['sarima'] = pd.read_csv(sarima_path)
        
        # Load DeepAR
        if models_config.get('deepar', {}).get('enabled', True) and deepar_path:
            logger.info(f"Loading DeepAR predictions from {deepar_path}")
            predictions['deepar'] = pd.read_csv(deepar_path)
        
        logger.info(f"Loaded predictions from {len(predictions)} models")
        
        return predictions
    
    def optimize_weights(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        metric: str = 'mape'
    ) -> Dict[str, float]:
        """
        Optimize ensemble weights using validation data.
        
        Args:
            predictions: Dictionary of model predictions
            y_true: True values
            metric: Metric to optimize
        
        Returns:
            Optimized weights dictionary
        """
        logger.info(f"Optimizing weights using {metric}")
        
        model_names = list(predictions.keys())
        n_models = len(model_names)
        
        # Objective function: minimize metric
        def objective(weights):
            # Ensure weights sum to 1
            weights = weights / weights.sum()
            
            # Calculate ensemble prediction
            ensemble_pred = sum(
                weights[i] * predictions[model_names[i]]
                for i in range(n_models)
            )
            
            # Calculate metric
            from common.metrics import mean_absolute_percentage_error
            if metric.lower() == 'mape':
                return mean_absolute_percentage_error(y_true, ensemble_pred)
            else:
                # Add other metrics as needed
                return mean_absolute_percentage_error(y_true, ensemble_pred)
        
        # Initial weights (equal)
        initial_weights = np.ones(n_models) / n_models
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        
        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Extract optimized weights
        optimized_weights = result.x / result.x.sum()
        
        weights_dict = {
            model_names[i]: float(optimized_weights[i])
            for i in range(n_models)
        }
        
        logger.info(f"Optimized weights: {weights_dict}")
        
        return weights_dict
    
    def calculate_weights_inverse_error(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        metric: str = 'mape'
    ) -> Dict[str, float]:
        """
        Calculate weights using inverse error method.
        
        Models with lower error get higher weight.
        
        Args:
            predictions: Dictionary of model predictions
            y_true: True values
            metric: Metric to use
        
        Returns:
            Weights dictionary
        """
        logger.info(f"Calculating weights using inverse {metric}")
        
        # Calculate errors for each model
        errors = {}
        for model_name, y_pred in predictions.items():
            metrics = calculate_all_metrics(y_true, y_pred)
            errors[model_name] = metrics[metric.lower()]
        
        # Calculate inverse error weights
        inverse_errors = {
            model: 1 / error if error > 0 else 1e10
            for model, error in errors.items()
        }
        
        # Normalize to sum to 1
        total_inverse_error = sum(inverse_errors.values())
        weights = {
            model: inv_error / total_inverse_error
            for model, inv_error in inverse_errors.items()
        }
        
        logger.info(f"Inverse error weights: {weights}")
        logger.info(f"Model errors: {errors}")
        
        return weights
    
    def get_weights(
        self,
        predictions: Optional[Dict[str, np.ndarray]] = None,
        y_true: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Get ensemble weights based on configured strategy.
        
        Args:
            predictions: Dictionary of model predictions (for auto optimization)
            y_true: True values (for auto optimization)
        
        Returns:
            Weights dictionary
        """
        weight_config = self.ensemble_config.get('weighted_average', {})
        weight_strategy = weight_config.get('weight_strategy', 'auto')
        
        if weight_strategy == 'manual':
            # Use manual weights from config
            models_config = self.ensemble_config.get('models', {})
            weights = {
                model: config.get('weight', 1.0 / 3)
                for model, config in models_config.items()
                if config.get('enabled', True)
            }
            
            # Normalize
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
            
            logger.info(f"Using manual weights: {weights}")
        
        elif weight_strategy == 'equal':
            # Equal weights
            models_config = self.ensemble_config.get('models', {})
            enabled_models = [
                model for model, config in models_config.items()
                if config.get('enabled', True)
            ]
            
            weight = 1.0 / len(enabled_models)
            weights = {model: weight for model in enabled_models}
            
            logger.info(f"Using equal weights: {weights}")
        
        elif weight_strategy == 'auto':
            # Auto-optimize weights
            if predictions is None or y_true is None:
                raise ValueError("predictions and y_true required for auto weight optimization")
            
            optimization_config = weight_config.get('optimization', {})
            method = optimization_config.get('method', 'inverse_error')
            metric = optimization_config.get('metric', 'mape')
            
            if method == 'inverse_error':
                weights = self.calculate_weights_inverse_error(predictions, y_true, metric)
            elif method == 'optimization':
                weights = self.optimize_weights(predictions, y_true, metric)
            else:
                logger.warning(f"Unknown optimization method: {method}. Using inverse_error.")
                weights = self.calculate_weights_inverse_error(predictions, y_true, metric)
        
        else:
            raise ValueError(f"Unknown weight strategy: {weight_strategy}")
        
        self.weights = weights
        return weights
    
    def combine_predictions(
        self,
        predictions: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None,
        forecast_column: str = 'forecast'
    ) -> pd.DataFrame:
        """
        Combine predictions using weighted average.
        
        Args:
            predictions: Dictionary of model prediction DataFrames
            weights: Weights dictionary (uses stored weights if None)
            forecast_column: Name of forecast column
        
        Returns:
            Combined ensemble predictions
        """
        if weights is None:
            weights = self.weights
        
        if not weights:
            raise ValueError("No weights available. Call get_weights() first.")
        
        logger.info("Combining predictions using weighted average")
        
        # Get common dates
        dates = None
        for model_name, pred_df in predictions.items():
            if dates is None:
                dates = pred_df['ds'].values
            else:
                # Ensure all models have same dates
                assert len(dates) == len(pred_df['ds'].values), \
                    f"Date mismatch for {model_name}"
        
        # Combine forecasts
        ensemble_forecast = np.zeros(len(dates))
        
        for model_name, pred_df in predictions.items():
            weight = weights.get(model_name, 0)
            
            # Handle different column names
            if forecast_column in pred_df.columns:
                forecast_values = pred_df[forecast_column].values
            elif 'yhat' in pred_df.columns:
                forecast_values = pred_df['yhat'].values
            elif 'prediction' in pred_df.columns:
                forecast_values = pred_df['prediction'].values
            else:
                logger.warning(f"No forecast column found for {model_name}")
                continue
            
            ensemble_forecast += weight * forecast_values
            logger.debug(f"Added {model_name} with weight {weight:.3f}")
        
        # Create result DataFrame
        result = pd.DataFrame({
            'ds': dates,
            'ensemble_forecast': ensemble_forecast,
            'model': 'ensemble'
        })
        
        # Add individual model predictions if configured
        if self.ensemble_config.get('output', {}).get('save_individual_predictions', True):
            for model_name, pred_df in predictions.items():
                if forecast_column in pred_df.columns:
                    result[f'{model_name}_forecast'] = pred_df[forecast_column].values
        
        # Add weights as metadata
        for model_name, weight in weights.items():
            result[f'{model_name}_weight'] = weight
        
        logger.info(f"Generated {len(result)} ensemble predictions")
        
        return result
    
    def evaluate_ensemble(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        ensemble_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate ensemble performance vs individual models.
        
        Args:
            predictions: Dictionary of individual model predictions
            y_true: True values
            ensemble_pred: Ensemble predictions
        
        Returns:
            Evaluation results
        """
        logger.info("Evaluating ensemble performance")
        
        # Add ensemble to predictions
        all_predictions = {**predictions, 'ensemble': ensemble_pred}
        
        # Compare all models
        comparison = compare_models(y_true, all_predictions)
        
        # Find best model
        best_model = get_best_model(comparison, metric='mape')
        
        # Calculate improvement
        ensemble_mape = comparison['ensemble']['mape']
        best_individual_mape = min(
            comparison[model]['mape']
            for model in predictions.keys()
        )
        
        improvement = ((best_individual_mape - ensemble_mape) / best_individual_mape) * 100
        
        results = {
            'comparison': comparison,
            'best_model': best_model,
            'ensemble_mape': ensemble_mape,
            'best_individual_mape': best_individual_mape,
            'improvement_percent': improvement
        }
        
        logger.info(f"Ensemble MAPE: {ensemble_mape:.2f}%")
        logger.info(f"Best individual MAPE: {best_individual_mape:.2f}%")
        logger.info(f"Improvement: {improvement:.2f}%")
        
        self.comparison_results = results
        
        return results


def create_ensemble_forecast(
    prophet_predictions_path: Path,
    sarima_predictions_path: Path,
    deepar_predictions_path: Optional[Path] = None,
    validation_data_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Main function to create ensemble forecast.
    
    Args:
        prophet_predictions_path: Path to Prophet predictions
        sarima_predictions_path: Path to SARIMA predictions
        deepar_predictions_path: Path to DeepAR predictions
        validation_data_path: Path to validation data (for weight optimization)
        config: Configuration dictionary
    
    Returns:
        Ensemble predictions DataFrame
    """
    setup_logging()
    logger.info("=" * 60)
    logger.info("ENSEMBLE FORECAST GENERATION")
    logger.info("=" * 60)
    
    # Initialize combiner
    combiner = EnsembleCombiner(config)
    
    # Load predictions
    predictions = combiner.load_predictions(
        prophet_path=prophet_predictions_path,
        sarima_path=sarima_predictions_path,
        deepar_path=deepar_predictions_path
    )
    
    # Get weights
    if validation_data_path and validation_data_path.exists():
        # Load validation data for weight optimization
        val_df = pd.read_csv(validation_data_path)
        
        # Extract predictions as arrays
        pred_arrays = {
            model: df['forecast'].values if 'forecast' in df.columns else df['yhat'].values
            for model, df in predictions.items()
        }
        
        y_true = val_df['sales_quantity'].values if 'sales_quantity' in val_df.columns else val_df['y'].values
        
        weights = combiner.get_weights(pred_arrays, y_true)
    else:
        # Use configured weights
        weights = combiner.get_weights()
    
    # Combine predictions
    ensemble_forecast = combiner.combine_predictions(predictions, weights)
    
    logger.info("=" * 60)
    logger.info("ENSEMBLE FORECAST COMPLETE")
    logger.info(f"Weights: {weights}")
    logger.info("=" * 60)
    
    return ensemble_forecast


if __name__ == "__main__":
    # Test ensemble
    import argparse
    
    parser = argparse.ArgumentParser(description='Create ensemble forecast')
    parser.add_argument('--prophet', type=str, required=True, help='Prophet predictions path')
    parser.add_argument('--sarima', type=str, required=True, help='SARIMA predictions path')
    parser.add_argument('--deepar', type=str, help='DeepAR predictions path')
    parser.add_argument('--validation', type=str, help='Validation data path')
    
    args = parser.parse_args()
    
    ensemble_forecast = create_ensemble_forecast(
        prophet_predictions_path=Path(args.prophet),
        sarima_predictions_path=Path(args.sarima),
        deepar_predictions_path=Path(args.deepar) if args.deepar else None,
        validation_data_path=Path(args.validation) if args.validation else None
    )
    
    print("\n✅ Ensemble forecast created successfully")
    print(f"\nFirst 5 predictions:")
    print(ensemble_forecast.head())
