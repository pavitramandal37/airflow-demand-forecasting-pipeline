"""
Common utilities for forecasting pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List
import logging

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Union[str, Path, None] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional path to log file
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def ensure_datetime(df: pd.DataFrame, date_column: str = 'ds') -> pd.DataFrame:
    """
    Ensure date column is datetime type.
    
    Args:
        df: DataFrame
        date_column: Name of date column
    
    Returns:
        DataFrame with datetime column
    """
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
        logger.debug(f"Converted '{date_column}' to datetime")
    
    return df


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    date_column: str = 'ds'
) -> tuple:
    """
    Split time series data into train and test sets.
    
    Args:
        df: DataFrame to split
        test_size: Fraction of data for test set
        date_column: Name of date column
    
    Returns:
        Tuple of (train_df, test_df)
    """
    df = ensure_datetime(df, date_column)
    df_sorted = df.sort_values(date_column)
    
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test")
    
    return train_df, test_df


def create_directory(path: Union[str, Path]) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get project root directory.
    
    Returns:
        Path to project root
    """
    # Assuming this file is in scripts/common/
    return Path(__file__).parent.parent.parent


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        DataFrame
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} records from {file_path}")
    
    return df


def save_predictions(
    predictions: pd.DataFrame,
    output_path: Union[str, Path],
    model_type: str
):
    """
    Save predictions to CSV file.
    
    Args:
        predictions: DataFrame with predictions
        output_path: Path to save file
        model_type: Type of model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    predictions.to_csv(output_path, index=False)
    logger.info(f"Saved {model_type} predictions to {output_path}")


if __name__ == "__main__":
    # Test utilities
    setup_logging("INFO")
    
    print("✅ Utilities module loaded successfully")
