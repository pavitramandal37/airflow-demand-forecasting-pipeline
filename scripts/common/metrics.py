"""
Evaluation metrics for forecasting models.

Provides standard metrics: MAPE, RMSE, MAE, SMAPE
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    MAPE = (1/n) * Σ|((y_true - y_pred) / y_true)| * 100
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        MAPE as percentage
    
    Note:
        Returns np.inf if any y_true values are zero
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    if np.any(y_true == 0):
        logger.warning("MAPE: Division by zero detected. Returning inf.")
        return np.inf
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return float(mape)


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    SMAPE = (100/n) * Σ(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2))
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        SMAPE as percentage
    
    Note:
        More robust than MAPE when y_true contains zeros
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Avoid division by zero
    mask = denominator != 0
    if not np.any(mask):
        logger.warning("SMAPE: All denominators are zero. Returning 0.")
        return 0.0
    
    smape = np.mean(numerator[mask] / denominator[mask]) * 100
    return float(smape)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    RMSE = sqrt((1/n) * Σ(y_true - y_pred)²)
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        RMSE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return float(rmse)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    MAE = (1/n) * Σ|y_true - y_pred|
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        MAE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = np.mean(np.abs(y_true - y_pred))
    return float(mae)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate specified evaluation metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        metrics: List of metrics to calculate. If None, calculates all.
                 Options: ['mape', 'smape', 'rmse', 'mae']
    
    Returns:
        Dictionary of metric names and values
    
    Example:
        >>> y_true = np.array([100, 110, 120])
        >>> y_pred = np.array([102, 108, 125])
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> print(metrics['mape'])
        2.42
    """
    if metrics is None:
        metrics = ['mape', 'smape', 'rmse', 'mae']
    
    results = {}
    
    metric_functions = {
        'mape': mean_absolute_percentage_error,
        'smape': symmetric_mean_absolute_percentage_error,
        'rmse': root_mean_squared_error,
        'mae': mean_absolute_error,
    }
    
    for metric in metrics:
        metric_lower = metric.lower()
        if metric_lower in metric_functions:
            try:
                results[metric_lower] = metric_functions[metric_lower](y_true, y_pred)
                logger.debug(f"Calculated {metric_lower}: {results[metric_lower]:.4f}")
            except Exception as e:
                logger.error(f"Error calculating {metric_lower}: {e}")
                results[metric_lower] = np.nan
        else:
            logger.warning(f"Unknown metric: {metric}. Skipping.")
    
    return results


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all available metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        Dictionary of all metric names and values
    """
    return calculate_metrics(y_true, y_pred, metrics=['mape', 'smape', 'rmse', 'mae'])


def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    metrics: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models using specified metrics.
    
    Args:
        y_true: Actual values
        predictions: Dictionary of model names and their predictions
        metrics: List of metrics to calculate
    
    Returns:
        Nested dictionary: {model_name: {metric_name: value}}
    
    Example:
        >>> y_true = np.array([100, 110, 120])
        >>> predictions = {
        ...     'prophet': np.array([102, 108, 125]),
        ...     'sarima': np.array([101, 111, 119])
        ... }
        >>> results = compare_models(y_true, predictions)
        >>> print(results['prophet']['mape'])
        2.42
    """
    comparison = {}
    
    for model_name, y_pred in predictions.items():
        comparison[model_name] = calculate_metrics(y_true, y_pred, metrics)
        logger.info(f"Calculated metrics for {model_name}")
    
    return comparison


def get_best_model(
    comparison: Dict[str, Dict[str, float]],
    metric: str = 'mape',
    lower_is_better: bool = True
) -> str:
    """
    Determine the best model based on a specific metric.
    
    Args:
        comparison: Results from compare_models()
        metric: Metric to use for comparison
        lower_is_better: If True, lower values are better
    
    Returns:
        Name of the best model
    
    Example:
        >>> comparison = {
        ...     'prophet': {'mape': 2.5, 'rmse': 5.0},
        ...     'sarima': {'mape': 2.0, 'rmse': 4.5}
        ... }
        >>> get_best_model(comparison, metric='mape')
        'sarima'
    """
    metric_lower = metric.lower()
    
    # Extract metric values for all models
    model_scores = {
        model: scores.get(metric_lower, np.inf if lower_is_better else -np.inf)
        for model, scores in comparison.items()
    }
    
    # Find best model
    if lower_is_better:
        best_model = min(model_scores, key=model_scores.get)
    else:
        best_model = max(model_scores, key=model_scores.get)
    
    logger.info(f"Best model by {metric}: {best_model} ({model_scores[best_model]:.4f})")
    
    return best_model


if __name__ == "__main__":
    # Test metrics
    logging.basicConfig(level=logging.INFO)
    
    y_true = np.array([100, 110, 120, 130, 140])
    y_pred_prophet = np.array([102, 108, 125, 128, 142])
    y_pred_sarima = np.array([101, 111, 119, 131, 139])
    
    print("\n=== Testing Metrics ===")
    metrics = calculate_all_metrics(y_true, y_pred_prophet)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print("\n=== Comparing Models ===")
    predictions = {
        'prophet': y_pred_prophet,
        'sarima': y_pred_sarima
    }
    comparison = compare_models(y_true, predictions)
    
    for model, scores in comparison.items():
        print(f"\n{model.upper()}:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nBest model: {get_best_model(comparison, metric='mape')}")
