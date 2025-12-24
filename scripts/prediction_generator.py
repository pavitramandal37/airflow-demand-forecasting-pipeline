#!/usr/bin/env python3
"""
Prediction Generator Module
===========================

Generates demand forecasts using trained Prophet models. This module handles
model loading, prediction generation, and forecast output formatting.

Key Features:
- Automatic latest model detection
- Configurable forecast horizon
- Confidence interval generation (upper/lower bounds)
- Forecast metadata for monitoring and analysis

Output includes:
- Point forecasts (yhat)
- Confidence intervals (yhat_lower, yhat_upper)
- Trend and seasonality components

Usage:
------
    from scripts.prediction_generator import generate_predictions

    metadata = generate_predictions(
        models_dir="models/saved_models",
        output_path="data/predictions/forecast.csv",
        config=config
    )
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml
from prophet import Prophet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load pipeline configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "pipeline_config.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_latest_model(models_dir: str) -> Tuple[str, Dict[str, Any]]:
    """
    Find and load the most recently trained model.

    Args:
        models_dir: Directory containing saved models

    Returns:
        Tuple of (model path, metadata dictionary)

    Raises:
        FileNotFoundError: If no models are found
    """
    models_path = Path(models_dir)

    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # Find all model files
    model_files = list(models_path.glob("prophet_model_v*.pkl"))

    if not model_files:
        raise FileNotFoundError(
            f"No trained models found in {models_dir}. "
            "Run model training first."
        )

    # Sort by modification time (most recent first)
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_model_path = model_files[0]

    # Load corresponding metadata
    metadata_file = models_path / f"{latest_model_path.stem}_metadata.json"
    metadata = {}

    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata for model version: {metadata.get('version', 'unknown')}")
    else:
        logger.warning(f"No metadata file found for {latest_model_path}")

    logger.info(f"Using latest model: {latest_model_path}")

    return str(latest_model_path), metadata


def load_model(model_path: str) -> Prophet:
    """
    Load a trained Prophet model from disk.

    Args:
        model_path: Path to pickled model file

    Returns:
        Loaded Prophet model

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If file is not a valid Prophet model
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        with open(path, "rb") as f:
            model = pickle.load(f)

        if not isinstance(model, Prophet):
            raise ValueError(f"Loaded object is not a Prophet model: {type(model)}")

        logger.info(f"Successfully loaded model from {model_path}")
        return model

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def generate_future_dataframe(
    model: Prophet,
    periods: int,
    freq: str = "D"
) -> pd.DataFrame:
    """
    Generate a DataFrame of future dates for prediction.

    Args:
        model: Trained Prophet model
        periods: Number of future periods to generate
        freq: Frequency of predictions (D=daily, W=weekly, M=monthly)

    Returns:
        DataFrame with future dates in 'ds' column
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    logger.info(f"Generated future dataframe with {len(future)} dates")
    return future


def format_forecast(forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Format Prophet forecast output for cleaner storage.

    Selects key columns and formats for business consumption:
    - ds: Forecast date
    - yhat: Point forecast
    - yhat_lower: Lower confidence bound
    - yhat_upper: Upper confidence bound
    - trend: Trend component
    - weekly: Weekly seasonality (if available)
    - yearly: Yearly seasonality (if available)

    Args:
        forecast: Raw Prophet forecast DataFrame

    Returns:
        Formatted forecast DataFrame
    """
    # Core columns always present
    columns = ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]

    # Optional seasonality columns
    if "weekly" in forecast.columns:
        columns.append("weekly")
    if "yearly" in forecast.columns:
        columns.append("yearly")

    formatted = forecast[columns].copy()

    # Round numeric columns for cleaner output
    numeric_cols = formatted.select_dtypes(include=["float64"]).columns
    formatted[numeric_cols] = formatted[numeric_cols].round(2)

    # Ensure non-negative forecasts (demand can't be negative)
    formatted["yhat"] = formatted["yhat"].clip(lower=0)
    formatted["yhat_lower"] = formatted["yhat_lower"].clip(lower=0)

    return formatted


def calculate_forecast_statistics(
    forecast: pd.DataFrame,
    future_only: bool = True
) -> Dict[str, Any]:
    """
    Calculate summary statistics for the forecast.

    Args:
        forecast: Forecast DataFrame
        future_only: If True, calculate stats only for future dates

    Returns:
        Dictionary of forecast statistics
    """
    # Determine the split point (last historical date)
    if future_only and len(forecast) > 0:
        # Assume last 30 rows are future predictions (based on typical horizon)
        # In practice, you'd compare with historical data dates
        future_df = forecast.tail(30)
    else:
        future_df = forecast

    stats = {
        "forecast_days": len(future_df),
        "mean_forecast": round(float(future_df["yhat"].mean()), 2),
        "min_forecast": round(float(future_df["yhat"].min()), 2),
        "max_forecast": round(float(future_df["yhat"].max()), 2),
        "total_forecast": round(float(future_df["yhat"].sum()), 2),
        "avg_confidence_width": round(
            float((future_df["yhat_upper"] - future_df["yhat_lower"]).mean()), 2
        )
    }

    return stats


def generate_predictions(
    models_dir: str,
    output_path: str,
    config: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None,
    horizon_days: Optional[int] = None
) -> Dict[str, Any]:
    """
    Main entry point: generate forecasts using trained model.

    Complete prediction workflow:
    1. Load latest model (or specified model)
    2. Generate future dates
    3. Create predictions with confidence intervals
    4. Format and save forecasts
    5. Return comprehensive metadata for XCom

    Args:
        models_dir: Directory containing saved models
        output_path: Path to save forecast CSV
        config: Optional configuration dictionary
        model_path: Optional specific model path (default: use latest)
        horizon_days: Optional forecast horizon (default: from config)

    Returns:
        Metadata dictionary containing:
        - model_version: Version of model used
        - model_path: Path to model file
        - output_path: Path to saved forecasts
        - horizon_days: Forecast horizon
        - forecast_statistics: Summary stats
        - generated_at: Generation timestamp
    """
    if config is None:
        config = load_config()

    model_config = config.get("model", {})

    if horizon_days is None:
        horizon_days = model_config.get("horizon_days", 30)

    logger.info(f"Starting prediction generation with horizon: {horizon_days} days")

    # Get model path
    if model_path is None:
        model_path, model_metadata = get_latest_model(models_dir)
    else:
        # Load metadata if model_path provided
        metadata_file = Path(model_path).with_suffix(".pkl").stem + "_metadata.json"
        metadata_path = Path(models_dir) / f"{Path(model_path).stem}_metadata.json"
        model_metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                model_metadata = json.load(f)

    # Load model
    model = load_model(model_path)

    # Generate future dates
    future = generate_future_dataframe(model, periods=horizon_days)

    # Generate predictions
    logger.info("Generating predictions...")
    forecast = model.predict(future)

    # Format forecast
    formatted_forecast = format_forecast(forecast)

    # Calculate statistics (for future dates only)
    forecast_stats = calculate_forecast_statistics(formatted_forecast, future_only=True)

    # Save forecast
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    formatted_forecast.to_csv(output_path, index=False)
    logger.info(f"Saved forecast to {output_path}")

    # Prepare metadata
    metadata = {
        "model_version": model_metadata.get("version", "unknown"),
        "model_path": str(model_path),
        "output_path": str(output_path),
        "horizon_days": horizon_days,
        "total_predictions": len(formatted_forecast),
        "forecast_range": {
            "start": formatted_forecast["ds"].min().strftime("%Y-%m-%d"),
            "end": formatted_forecast["ds"].max().strftime("%Y-%m-%d")
        },
        "forecast_statistics": forecast_stats,
        "generated_at": datetime.now().isoformat()
    }

    logger.info(
        f"Prediction generation complete. "
        f"Generated {len(formatted_forecast)} predictions for {horizon_days} day horizon"
    )

    return metadata


def main() -> None:
    """
    Main entry point for standalone prediction generation.
    """
    try:
        config = load_config()
        project_root = Path(__file__).parent.parent

        models_path = config.get("paths", {}).get("models", "models/saved_models")
        predictions_path = config.get("paths", {}).get("predictions", "data/predictions")

        models_dir = project_root / models_path
        output_path = project_root / predictions_path / "forecast.csv"

        metadata = generate_predictions(str(models_dir), str(output_path), config)

        logger.info("Prediction generation completed successfully")
        logger.info(f"Model used: {metadata['model_version']}")
        logger.info(f"Forecast saved to: {metadata['output_path']}")
        logger.info(f"Forecast statistics: {metadata['forecast_statistics']}")

    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
