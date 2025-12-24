#!/usr/bin/env python3
"""
Model Training Module
=====================

Trains Prophet forecasting models with proper versioning and metadata tracking.
This module implements MLOps best practices for model management.

Key Features:
- Versioned model naming: prophet_model_vYYYYMMDD_<datahash>.pkl
- Comprehensive metadata JSON alongside each model
- Configurable Prophet hyperparameters
- Training duration tracking for performance monitoring

The versioning scheme enables:
- Model rollback capabilities
- Training reproducibility verification
- Data drift detection through hash comparison

Usage:
------
    from scripts.model_training import train_and_save_model

    metadata = train_and_save_model(
        data_path="data/processed/features.csv",
        models_path="models/saved_models",
        config=config
    )
"""

import hashlib
import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml

# Suppress cmdstanpy logger to prevent Prophet initialization issues
import cmdstanpy
cmdstanpy_logger = logging.getLogger('cmdstanpy')
cmdstanpy_logger.setLevel(logging.WARNING)

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


def compute_data_hash(df: pd.DataFrame) -> str:
    """
    Compute a hash of the training data for versioning.

    The hash captures the data state at training time, enabling:
    - Detection of data drift between training runs
    - Verification of model-data correspondence
    - Reproducibility auditing

    Args:
        df: Training DataFrame

    Returns:
        8-character hexadecimal hash string
    """
    data_str = df.to_csv(index=False)
    full_hash = hashlib.md5(data_str.encode()).hexdigest()
    return full_hash[:8]


def create_model_version(data_hash: str) -> str:
    """
    Create a versioned model identifier.

    Format: prophet_model_vYYYYMMDD_<datahash>

    This naming convention provides:
    - Date-based ordering for easy identification
    - Data hash for traceability
    - Unique identifiers for concurrent training runs

    Args:
        data_hash: Hash of the training data

    Returns:
        Version string for model naming
    """
    date_str = datetime.now().strftime("%Y%m%d")
    return f"prophet_model_v{date_str}_{data_hash}"


def configure_prophet(config: Dict[str, Any]) -> Prophet:
    """
    Configure Prophet model with parameters from config.

    Applies hyperparameters from the configuration file to create
    a properly tuned Prophet model for retail demand forecasting.

    Args:
        config: Configuration dictionary with model.prophet section

    Returns:
        Configured Prophet model instance
    """
    prophet_config = config.get("model", {}).get("prophet", {})

    model = Prophet(
        changepoint_prior_scale=prophet_config.get("changepoint_prior_scale", 0.05),
        seasonality_prior_scale=prophet_config.get("seasonality_prior_scale", 10.0),
        seasonality_mode=prophet_config.get("seasonality_mode", "multiplicative"),
        weekly_seasonality=prophet_config.get("weekly_seasonality", True),
        yearly_seasonality=prophet_config.get("yearly_seasonality", True),
        interval_width=prophet_config.get("interval_width", 0.95)
    )

    logger.info(
        f"Configured Prophet with: "
        f"changepoint_prior_scale={prophet_config.get('changepoint_prior_scale', 0.05)}, "
        f"seasonality_mode={prophet_config.get('seasonality_mode', 'multiplicative')}"
    )

    return model


def train_prophet_model(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[Prophet, float]:
    """
    Train a Prophet model on the provided data.

    The training data must have:
    - 'ds' column: datestamp (datetime)
    - 'y' column: target variable (numeric)

    Args:
        df: Training DataFrame with ds and y columns
        config: Configuration dictionary

    Returns:
        Tuple of (trained Prophet model, training duration in seconds)
    """
    logger.info(f"Training Prophet model on {len(df)} records")

    # Validate required columns
    if "ds" not in df.columns or "y" not in df.columns:
        raise ValueError("Training data must have 'ds' and 'y' columns")

    # Configure model
    model = configure_prophet(config)

    # Prepare training data (Prophet only needs ds and y)
    train_df = df[["ds", "y"]].copy()

    # Train with timing
    start_time = time.time()
    model.fit(train_df)
    training_duration = time.time() - start_time

    logger.info(f"Training complete in {training_duration:.2f} seconds")

    return model, training_duration


def save_model_with_metadata(
    model: Prophet,
    models_dir: str,
    model_version: str,
    metadata: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Save trained model and metadata to disk.

    Creates two files:
    1. {model_version}.pkl - Pickled Prophet model
    2. {model_version}_metadata.json - Training metadata

    Args:
        model: Trained Prophet model
        models_dir: Directory to save models
        model_version: Version string for naming
        metadata: Metadata dictionary to save

    Returns:
        Tuple of (model file path, metadata file path)
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_file = models_path / f"{model_version}.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to {model_file}")

    # Save metadata
    metadata_file = models_path / f"{model_version}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved metadata to {metadata_file}")

    return str(model_file), str(metadata_file)


def train_and_save_model(
    data_path: str,
    models_dir: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main entry point: train Prophet model and save with versioning.

    Complete training workflow:
    1. Load feature-engineered data
    2. Compute data hash for versioning
    3. Create version identifier
    4. Configure and train Prophet model
    5. Save model and metadata
    6. Return comprehensive metadata for XCom

    Args:
        data_path: Path to feature-engineered CSV
        models_dir: Directory to save models
        config: Optional configuration dictionary

    Returns:
        Metadata dictionary containing:
        - version: Model version string
        - model_path: Path to saved model file
        - metadata_path: Path to saved metadata file
        - created_at: Training timestamp
        - data_hash: Hash of training data
        - model_type: Type of model (prophet)
        - training_duration_sec: Training time
        - horizon_days: Forecast horizon
        - training_records: Number of training records
    """
    if config is None:
        config = load_config()

    model_config = config.get("model", {})
    horizon_days = model_config.get("horizon_days", 30)

    logger.info(f"Starting model training from {data_path}")

    # Load training data
    df = pd.read_csv(data_path, parse_dates=["ds"])
    logger.info(f"Loaded {len(df)} training records")

    # Compute data hash for versioning
    data_hash = compute_data_hash(df)
    logger.info(f"Data hash: {data_hash}")

    # Create model version
    model_version = create_model_version(data_hash)
    logger.info(f"Model version: {model_version}")

    # Train model
    model, training_duration = train_prophet_model(df, config)

    # Prepare metadata
    metadata = {
        "version": model_version,
        "created_at": datetime.now().isoformat(),
        "data_hash": data_hash,
        "model_type": model_config.get("type", "prophet"),
        "training_duration_sec": round(training_duration, 2),
        "horizon_days": horizon_days,
        "training_records": len(df),
        "date_range": {
            "start": df["ds"].min().strftime("%Y-%m-%d"),
            "end": df["ds"].max().strftime("%Y-%m-%d")
        },
        "prophet_params": {
            "changepoint_prior_scale": model.changepoint_prior_scale,
            "seasonality_prior_scale": model.seasonality_prior_scale,
            "seasonality_mode": model.seasonality_mode
        }
    }

    # Save model and metadata
    model_path, metadata_path = save_model_with_metadata(
        model, models_dir, model_version, metadata
    )

    # Add paths to return metadata
    metadata["model_path"] = model_path
    metadata["metadata_path"] = metadata_path

    logger.info(
        f"Model training complete. Version: {model_version}, "
        f"Duration: {training_duration:.2f}s"
    )

    return metadata


def get_latest_model(models_dir: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Find the most recently trained model in the models directory.

    Args:
        models_dir: Directory containing saved models

    Returns:
        Tuple of (model path, metadata dict) or (None, None) if no models found
    """
    models_path = Path(models_dir)

    if not models_path.exists():
        return None, None

    # Find all model files
    model_files = list(models_path.glob("prophet_model_v*.pkl"))

    if not model_files:
        return None, None

    # Sort by modification time (most recent first)
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_model = model_files[0]

    # Load corresponding metadata
    metadata_file = models_path / f"{latest_model.stem}_metadata.json"
    metadata = None

    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

    logger.info(f"Found latest model: {latest_model}")

    return str(latest_model), metadata


def main() -> None:
    """
    Main entry point for standalone model training.
    """
    try:
        config = load_config()
        project_root = Path(__file__).parent.parent

        processed_path = config.get("paths", {}).get("processed_data", "data/processed")
        models_path = config.get("paths", {}).get("models", "models/saved_models")

        data_path = project_root / processed_path / "features.csv"
        models_dir = project_root / models_path

        metadata = train_and_save_model(str(data_path), str(models_dir), config)

        logger.info("Model training completed successfully")
        logger.info(f"Model version: {metadata['version']}")
        logger.info(f"Model path: {metadata['model_path']}")

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


if __name__ == "__main__":
    main()
