#!/usr/bin/env python3
"""
Feature Engineering Module
==========================

Creates features for time-series forecasting models. This module transforms
cleaned sales data into a format suitable for Prophet and other forecasting
algorithms.

Features Generated:
- Time-based features (day of week, month, quarter, year)
- Rolling averages (7-day, 14-day, 30-day)
- Lag features (previous day, week, two weeks)
- Prophet-compatible format (ds, y columns)

The features are designed to capture:
- Seasonal patterns (weekly, monthly, yearly)
- Trend information (rolling averages)
- Autocorrelation (lag features)

Usage:
------
    from scripts.feature_engineering import engineer_features

    metadata = engineer_features(
        input_path="data/processed/cleaned_data.csv",
        output_path="data/processed/features.csv",
        config=config
    )
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

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


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features extracted from the date column.

    Features added:
    - day_of_week: 0 (Monday) to 6 (Sunday)
    - day_of_month: 1 to 31
    - week_of_year: 1 to 52
    - month: 1 to 12
    - quarter: 1 to 4
    - year: YYYY
    - is_weekend: True if Saturday or Sunday
    - is_month_start: True if first day of month
    - is_month_end: True if last day of month

    Args:
        df: DataFrame with 'date' column

    Returns:
        DataFrame with additional time features
    """
    df = df.copy()

    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])

    # Extract time components
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year

    # Boolean features
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    logger.info("Added time-based features: day_of_week, month, quarter, etc.")

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_column: str,
    windows: List[int],
    group_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Add rolling average features for the target column.

    Creates rolling mean features for specified window sizes. When group_column
    is provided, calculates rolling averages within each group (e.g., per product).

    Args:
        df: DataFrame with target column
        target_column: Column to compute rolling averages on
        windows: List of window sizes (e.g., [7, 14, 30])
        group_column: Optional column to group by before computing rolling stats

    Returns:
        DataFrame with additional rolling features
    """
    df = df.copy()
    df = df.sort_values("date")

    for window in windows:
        feature_name = f"{target_column}_rolling_{window}d"

        if group_column and group_column in df.columns:
            # Calculate rolling average within each group
            df[feature_name] = df.groupby(group_column)[target_column].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        else:
            df[feature_name] = df[target_column].rolling(
                window=window, min_periods=1
            ).mean()

        # Round for cleaner output
        df[feature_name] = df[feature_name].round(2)

    logger.info(f"Added rolling features with windows: {windows}")

    return df


def add_lag_features(
    df: pd.DataFrame,
    target_column: str,
    lags: List[int],
    group_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Add lagged features for capturing autocorrelation.

    Creates lag features for specified lag periods. Useful for capturing
    how previous values influence current predictions.

    Args:
        df: DataFrame with target column
        target_column: Column to create lags from
        lags: List of lag periods (e.g., [1, 7, 14])
        group_column: Optional column to group by before computing lags

    Returns:
        DataFrame with additional lag features
    """
    df = df.copy()
    df = df.sort_values("date")

    for lag in lags:
        feature_name = f"{target_column}_lag_{lag}d"

        if group_column and group_column in df.columns:
            # Calculate lag within each group
            df[feature_name] = df.groupby(group_column)[target_column].shift(lag)
        else:
            df[feature_name] = df[target_column].shift(lag)

    logger.info(f"Added lag features with periods: {lags}")

    return df


def create_prophet_format(
    df: pd.DataFrame,
    date_column: str = "date",
    target_column: str = "sales_quantity"
) -> pd.DataFrame:
    """
    Create Prophet-compatible format with ds (datestamp) and y (target) columns.

    Prophet requires specific column names:
    - ds: Datetime column (datestamp)
    - y: Target variable to forecast

    Args:
        df: DataFrame with date and target columns
        date_column: Name of the date column
        target_column: Name of the target column

    Returns:
        DataFrame with ds and y columns (plus any additional features)
    """
    df = df.copy()

    # Rename columns for Prophet
    df["ds"] = pd.to_datetime(df[date_column])
    df["y"] = df[target_column]

    logger.info(f"Created Prophet format: ds <- {date_column}, y <- {target_column}")

    return df


def engineer_features(
    input_path: str,
    output_path: str,
    config: Optional[Dict[str, Any]] = None,
    product_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point: engineer features from cleaned data.

    Orchestrates the complete feature engineering workflow:
    1. Load cleaned data
    2. Filter to specific product (if specified)
    3. Aggregate to daily level (if multiple products)
    4. Add time-based features
    5. Add rolling average features
    6. Add lag features
    7. Create Prophet-compatible format
    8. Save engineered features

    Args:
        input_path: Path to cleaned data CSV
        output_path: Path to save feature-engineered CSV
        config: Optional configuration dictionary
        product_id: Optional product ID to filter to (default: aggregate all)

    Returns:
        Metadata dictionary containing feature engineering details
    """
    if config is None:
        config = load_config()

    feature_config = config.get("features", {})

    logger.info(f"Starting feature engineering: {input_path}")

    # Load cleaned data
    df = pd.read_csv(input_path, parse_dates=["date"])
    original_rows = len(df)
    logger.info(f"Loaded {len(df)} records")

    # Filter to specific product if specified
    if product_id and "product_id" in df.columns:
        df = df[df["product_id"] == product_id].copy()
        logger.info(f"Filtered to product {product_id}: {len(df)} records")

    # Aggregate to daily level (sum sales across products if not filtered)
    if "product_id" in df.columns and product_id is None:
        df_daily = df.groupby("date").agg({
            "sales_quantity": "sum",
            "revenue": "sum"
        }).reset_index()
        logger.info(f"Aggregated to daily level: {len(df_daily)} days")
    else:
        df_daily = df.copy()

    # Ensure sorted by date
    df_daily = df_daily.sort_values("date").reset_index(drop=True)

    # Track features added
    features_added = []

    # Add time-based features
    if feature_config.get("extract_time_features", True):
        df_daily = add_time_features(df_daily)
        features_added.extend([
            "day_of_week", "day_of_month", "week_of_year",
            "month", "quarter", "year",
            "is_weekend", "is_month_start", "is_month_end"
        ])

    # Add rolling features
    rolling_windows = feature_config.get("rolling_windows", [7, 14, 30])
    df_daily = add_rolling_features(
        df_daily,
        target_column="sales_quantity",
        windows=rolling_windows
    )
    features_added.extend([f"sales_quantity_rolling_{w}d" for w in rolling_windows])

    # Add lag features
    lag_days = feature_config.get("lag_days", [1, 7, 14])
    df_daily = add_lag_features(
        df_daily,
        target_column="sales_quantity",
        lags=lag_days
    )
    features_added.extend([f"sales_quantity_lag_{l}d" for l in lag_days])

    # Create Prophet format
    df_daily = create_prophet_format(df_daily)

    # Fill NaN values created by rolling/lag operations with forward fill then backward fill
    # This handles the initial rows where rolling/lag can't be computed
    for col in features_added:
        if col in df_daily.columns:
            df_daily[col] = df_daily[col].ffill().bfill()

    # Save engineered features
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_daily.to_csv(output_path, index=False)

    # Create metadata
    metadata = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "original_records": original_rows,
        "engineered_records": len(df_daily),
        "product_filter": product_id,
        "features_added": features_added,
        "num_features": len(features_added),
        "rolling_windows": rolling_windows,
        "lag_days": lag_days,
        "date_range": {
            "start": df_daily["ds"].min().strftime("%Y-%m-%d"),
            "end": df_daily["ds"].max().strftime("%Y-%m-%d")
        },
        "processed_at": datetime.now().isoformat()
    }

    logger.info(
        f"Feature engineering complete. "
        f"Added {len(features_added)} features, "
        f"saved {len(df_daily)} records to {output_path}"
    )

    return metadata


def main() -> None:
    """
    Main entry point for standalone feature engineering.
    """
    try:
        config = load_config()
        project_root = Path(__file__).parent.parent

        processed_path = config.get("paths", {}).get("processed_data", "data/processed")

        input_path = project_root / processed_path / "cleaned_data.csv"
        output_path = project_root / processed_path / "features.csv"

        metadata = engineer_features(str(input_path), str(output_path), config)

        logger.info("Feature engineering completed successfully")
        logger.info(f"Features added: {metadata['features_added']}")

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


if __name__ == "__main__":
    main()
