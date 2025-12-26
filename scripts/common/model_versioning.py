"""
Model versioning and metadata management.

Handles version creation, metadata storage, and reproducibility.
"""

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def create_model_version(
    model_type: str,
    data_hash: Optional[str] = None,
    seed: Optional[int] = None,
    include_timestamp: bool = True
) -> str:
    """
    Create a versioned model name.
    
    Format: {model_type}_v{timestamp}_{data_hash}_seed{seed}
    
    Args:
        model_type: Type of model ('prophet', 'sarima', 'deepar')
        data_hash: Hash of training data (first 8 chars)
        seed: Random seed used for training
        include_timestamp: Whether to include timestamp
    
    Returns:
        Versioned model name
    
    Example:
        >>> create_model_version('prophet', 'a1b2c3d4', 42)
        'prophet_v20241226_a1b2c3d4_seed42'
    """
    parts = [model_type]
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d")
        parts.append(f"v{timestamp}")
    
    if data_hash:
        # Use first 8 characters of hash
        short_hash = data_hash[:8] if len(data_hash) > 8 else data_hash
        parts.append(short_hash)
    
    if seed is not None:
        parts.append(f"seed{seed}")
    
    version = "_".join(parts)
    logger.info(f"Created model version: {version}")
    
    return version


def calculate_data_hash(data: Any, algorithm: str = 'md5') -> str:
    """
    Calculate hash of training data for versioning.
    
    Args:
        data: Data to hash (DataFrame, array, etc.)
        algorithm: Hash algorithm ('md5', 'sha256')
    
    Returns:
        Hexadecimal hash string
    
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': [1, 2, 3]})
        >>> hash_val = calculate_data_hash(df)
        >>> len(hash_val)
        32
    """
    # Convert data to bytes
    if hasattr(data, 'to_csv'):
        # DataFrame
        data_bytes = data.to_csv(index=False).encode('utf-8')
    elif hasattr(data, 'tobytes'):
        # NumPy array
        data_bytes = data.tobytes()
    else:
        # Generic object
        data_bytes = str(data).encode('utf-8')
    
    # Calculate hash
    if algorithm == 'md5':
        hash_obj = hashlib.md5(data_bytes)
    elif algorithm == 'sha256':
        hash_obj = hashlib.sha256(data_bytes)
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    hash_value = hash_obj.hexdigest()
    logger.debug(f"Calculated {algorithm} hash: {hash_value[:16]}...")
    
    return hash_value


def save_model_metadata(
    model_version: str,
    model_type: str,
    metadata: Dict[str, Any],
    save_dir: Path,
    product_id: Optional[str] = None
) -> Path:
    """
    Save model metadata to JSON file.
    
    Args:
        model_version: Version string from create_model_version()
        model_type: Type of model
        metadata: Dictionary of metadata to save
        save_dir: Directory to save metadata
        product_id: Optional product ID for per-product models (SARIMA)
    
    Returns:
        Path to saved metadata file
    
    Example:
        >>> metadata = {
        ...     'seed': 42,
        ...     'data_hash': 'a1b2c3d4',
        ...     'hyperparams': {'changepoint_prior_scale': 0.05},
        ...     'validation_mape': 2.5
        ... }
        >>> save_model_metadata('prophet_v20241226', 'prophet', metadata, Path('models/prophet'))
    """
    save_dir = Path(save_dir)
    
    # Create directory if needed
    if product_id:
        # Per-product structure (SARIMA)
        save_dir = save_dir / product_id
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Add standard fields
    full_metadata = {
        'model_version': model_version,
        'model_type': model_type,
        'created_at': datetime.now().isoformat(),
        **metadata
    }
    
    if product_id:
        full_metadata['product_id'] = product_id
    
    # Save to JSON
    metadata_file = save_dir / f"{model_version}_metadata.json"
    
    with open(metadata_file, 'w') as f:
        json.dump(full_metadata, f, indent=2, default=str)
    
    logger.info(f"Saved metadata to {metadata_file}")
    
    return metadata_file


def load_model_metadata(metadata_file: Path) -> Dict[str, Any]:
    """
    Load model metadata from JSON file.
    
    Args:
        metadata_file: Path to metadata JSON file
    
    Returns:
        Metadata dictionary
    """
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded metadata from {metadata_file}")
    
    return metadata


def save_model(
    model: Any,
    model_version: str,
    save_dir: Path,
    product_id: Optional[str] = None
) -> Path:
    """
    Save trained model to pickle file.
    
    Args:
        model: Trained model object
        model_version: Version string
        save_dir: Directory to save model
        product_id: Optional product ID for per-product models
    
    Returns:
        Path to saved model file
    """
    save_dir = Path(save_dir)
    
    # Create directory if needed
    if product_id:
        save_dir = save_dir / product_id
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_file = save_dir / f"{model_version}.pkl"
    
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Saved model to {model_file}")
    
    return model_file


def load_model(model_file: Path) -> Any:
    """
    Load trained model from pickle file.
    
    Args:
        model_file: Path to model pickle file
    
    Returns:
        Loaded model object
    """
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Loaded model from {model_file}")
    
    return model


def get_latest_model_version(
    model_dir: Path,
    model_type: str,
    product_id: Optional[str] = None
) -> Optional[Path]:
    """
    Get the latest model version from a directory.
    
    Args:
        model_dir: Directory containing models
        model_type: Type of model
        product_id: Optional product ID
    
    Returns:
        Path to latest model file, or None if no models found
    """
    model_dir = Path(model_dir)
    
    if product_id:
        model_dir = model_dir / product_id
    
    if not model_dir.exists():
        logger.warning(f"Model directory does not exist: {model_dir}")
        return None
    
    # Find all model files
    model_files = list(model_dir.glob(f"{model_type}_*.pkl"))
    
    if not model_files:
        logger.warning(f"No models found in {model_dir}")
        return None
    
    # Sort by modification time (most recent first)
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    
    logger.info(f"Latest model: {latest_model}")
    
    return latest_model


def create_full_model_metadata(
    model_type: str,
    seed: int,
    data_hash: str,
    hyperparams: Dict[str, Any],
    validation_metrics: Dict[str, float],
    training_time_seconds: float,
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create comprehensive model metadata.
    
    Args:
        model_type: Type of model
        seed: Random seed
        data_hash: Hash of training data
        hyperparams: Model hyperparameters
        validation_metrics: Validation performance metrics
        training_time_seconds: Time taken to train
        additional_info: Any additional information
    
    Returns:
        Complete metadata dictionary
    """
    metadata = {
        'model_type': model_type,
        'seed': seed,
        'data_hash': data_hash,
        'hyperparameters': hyperparams,
        'validation_metrics': validation_metrics,
        'training_time_seconds': training_time_seconds,
        'created_at': datetime.now().isoformat(),
    }
    
    if additional_info:
        metadata.update(additional_info)
    
    return metadata


if __name__ == "__main__":
    # Test versioning
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Testing Model Versioning ===")
    
    # Create version
    version = create_model_version('prophet', 'a1b2c3d4e5f6', 42)
    print(f"Version: {version}")
    
    # Create metadata
    metadata = create_full_model_metadata(
        model_type='prophet',
        seed=42,
        data_hash='a1b2c3d4e5f6',
        hyperparams={'changepoint_prior_scale': 0.05},
        validation_metrics={'mape': 2.5, 'rmse': 10.2},
        training_time_seconds=45.3
    )
    
    print("\nMetadata:")
    print(json.dumps(metadata, indent=2, default=str))
