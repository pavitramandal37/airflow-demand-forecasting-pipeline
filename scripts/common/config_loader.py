"""
Configuration loader for multi-model forecasting pipeline.

Loads and merges hierarchical YAML configurations:
- base_config.yaml (shared settings)
- model-specific configs (prophet, sarima, deepar, ensemble)
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(model_type: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration for a specific model.
    
    Merges base_config with model-specific config.
    
    Args:
        model_type: Type of model ('prophet', 'sarima', 'deepar', 'ensemble')
        config_dir: Path to config directory (defaults to project config/)
    
    Returns:
        Merged configuration dictionary
    
    Example:
        >>> config = load_config('prophet')
        >>> print(config['model']['prophet']['changepoint_prior_scale'])
        0.05
    """
    if config_dir is None:
        # Default to project config directory
        config_dir = Path(__file__).parent.parent.parent / "config"
    
    logger.info(f"Loading configuration for model: {model_type}")
    
    # Load base config
    base_config_path = config_dir / "base_config.yaml"
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    logger.debug(f"Loaded base config from {base_config_path}")
    
    # Load model-specific config
    model_config_file = f"{model_type}_config.yaml"
    model_config_path = config_dir / model_config_file
    
    if not model_config_path.exists():
        logger.warning(f"Model config not found: {model_config_path}. Using base config only.")
        return base_config
    
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    logger.debug(f"Loaded model config from {model_config_path}")
    
    # Merge configs (model config takes precedence)
    merged_config = merge_configs(base_config, model_config)
    
    logger.info(f"Successfully loaded configuration for {model_type}")
    
    return merged_config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    
    Override values take precedence over base values.
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
    
    Returns:
        Merged configuration dictionary
    
    Example:
        >>> base = {'a': 1, 'b': {'c': 2}}
        >>> override = {'b': {'d': 3}}
        >>> merge_configs(base, override)
        {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    merged = base.copy()
    
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value
    
    return merged


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to value (e.g., 'model.prophet.changepoint_prior_scale')
        default: Default value if key not found
    
    Returns:
        Configuration value or default
    
    Example:
        >>> config = {'model': {'prophet': {'changepoint_prior_scale': 0.05}}}
        >>> get_config_value(config, 'model.prophet.changepoint_prior_scale')
        0.05
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def validate_config(config: Dict[str, Any], model_type: str) -> bool:
    """
    Validate configuration for a specific model type.
    
    Args:
        config: Configuration dictionary
        model_type: Type of model
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check required base fields
    required_base_fields = ['pipeline', 'paths', 'data_quality', 'evaluation']
    for field in required_base_fields:
        if field not in config:
            raise ValueError(f"Missing required base field: {field}")
    
    # Check model-specific fields
    if model_type in ['prophet', 'sarima', 'deepar']:
        if 'model' not in config:
            raise ValueError(f"Missing 'model' field for {model_type}")
        
        if 'type' not in config['model']:
            raise ValueError(f"Missing 'model.type' field")
        
        if config['model']['type'] != model_type:
            raise ValueError(
                f"Model type mismatch: expected {model_type}, got {config['model']['type']}"
            )
    
    elif model_type == 'ensemble':
        if 'ensemble' not in config:
            raise ValueError("Missing 'ensemble' field for ensemble config")
    
    logger.info(f"Configuration validation passed for {model_type}")
    return True


if __name__ == "__main__":
    # Test configuration loading
    logging.basicConfig(level=logging.DEBUG)
    
    for model in ['prophet', 'sarima', 'deepar', 'ensemble']:
        try:
            config = load_config(model)
            validate_config(config, model)
            print(f"✅ {model.upper()} config loaded successfully")
        except Exception as e:
            print(f"❌ {model.upper()} config failed: {e}")
