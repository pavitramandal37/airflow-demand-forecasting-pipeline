#!/usr/bin/env python3
"""
Sample Data Generation Module
=============================

Generates synthetic retail sales data for testing and demonstrating the
demand forecasting pipeline. The data simulates realistic patterns including:

- Long-term growth trend
- Weekly seasonality (weekday vs weekend patterns)
- Annual seasonality (holiday peaks)
- Random noise
- Outliers (promotional spikes, supply disruptions)
- Missing values (data collection gaps)

The generated data is suitable for training Prophet models and validating
the data quality checks implemented in the pipeline.

Usage:
------
    # Run directly
    python -m scripts.generate_sample_data

    # Or import and use programmatically
    from scripts.generate_sample_data import generate_sample_data
    df = generate_sample_data(config)
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

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
        config_path: Path to config file. Defaults to config/pipeline_config.yaml

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    if config_path is None:
        # Determine project root (parent of scripts directory)
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "pipeline_config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def generate_sample_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate synthetic sales data with realistic patterns.

    The data generation process:
    1. Create date range for specified number of days
    2. For each product, generate base sales with trend
    3. Apply weekly seasonality (lower on weekends)
    4. Apply annual seasonality (peaks around holidays)
    5. Add random noise
    6. Inject outliers (both positive and negative)
    7. Introduce missing values

    Args:
        config: Configuration dictionary with sample_data parameters

    Returns:
        DataFrame with columns: date, product_id, sales_quantity, revenue
    """
    sample_config = config.get("sample_data", {})

    # Extract configuration parameters with defaults
    num_days = sample_config.get("num_days", 365)
    product_ids = sample_config.get("product_ids", ["PROD_001"])
    base_sales = sample_config.get("base_sales", 100)
    trend_growth = sample_config.get("trend_growth", 1.15)
    seasonality_amplitude = sample_config.get("seasonality_amplitude", 0.3)
    noise_level = sample_config.get("noise_level", 0.15)
    outlier_rate = sample_config.get("outlier_rate", 0.07)
    missing_rate = sample_config.get("missing_rate", 0.025)
    random_seed = sample_config.get("random_seed", 42)

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    logger.info(f"Generating {num_days} days of data for {len(product_ids)} products")

    # Generate date range ending today
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=num_days - 1)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    all_records = []

    for product_id in product_ids:
        logger.debug(f"Generating data for {product_id}")

        # Generate base sales with linear trend
        # Trend starts at base_sales and grows to base_sales * trend_growth
        day_indices = np.arange(num_days)
        trend = base_sales * (1 + (trend_growth - 1) * day_indices / num_days)

        # Weekly seasonality: lower on weekends (Sat=5, Sun=6)
        # Weekdays: multiplier ~1.0-1.1, Weekends: multiplier ~0.7-0.8
        day_of_week = np.array([d.weekday() for d in dates])
        weekly_seasonality = np.where(
            day_of_week >= 5,  # Weekend
            0.75 + 0.05 * np.random.random(num_days),  # 0.75-0.80 multiplier
            1.0 + 0.1 * np.sin(2 * np.pi * day_of_week / 5)  # Slight weekday variation
        )

        # Annual seasonality: peaks around Dec (holiday season) and dips in Feb
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        # Peak around day 350 (mid-December), trough around day 45 (mid-February)
        annual_seasonality = 1 + seasonality_amplitude * np.sin(
            2 * np.pi * (day_of_year - 45) / 365
        )

        # Combine trend and seasonality
        sales = trend * weekly_seasonality * annual_seasonality

        # Add random noise (normally distributed)
        noise = np.random.normal(0, noise_level * base_sales, num_days)
        sales = sales + noise

        # Inject outliers
        num_outliers = int(num_days * outlier_rate)
        outlier_indices = np.random.choice(num_days, num_outliers, replace=False)

        for idx in outlier_indices:
            # 70% positive outliers (promotions), 30% negative (disruptions)
            if np.random.random() < 0.7:
                # Positive outlier: 2x to 3x normal sales
                sales[idx] *= np.random.uniform(2.0, 3.0)
            else:
                # Negative outlier: 0.2x to 0.5x normal sales
                sales[idx] *= np.random.uniform(0.2, 0.5)

        # Ensure non-negative sales (floor at 0)
        sales = np.maximum(sales, 0)

        # Round to integers (can't sell fractional units)
        sales_quantity = np.round(sales).astype(int)

        # Calculate revenue (base price with slight variation per product)
        base_price = 10.0 * (1 + 0.2 * (hash(product_id) % 10) / 10)
        # Add some price variation over time (inflation, promotions)
        price_variation = 1 + 0.1 * np.sin(2 * np.pi * day_indices / 90)
        revenue = sales_quantity * base_price * price_variation
        revenue = np.round(revenue, 2)

        # Create records for this product
        for i, date in enumerate(dates):
            all_records.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_id": product_id,
                "sales_quantity": sales_quantity[i],
                "revenue": revenue[i]
            })

    # Convert to DataFrame
    df = pd.DataFrame(all_records)

    # Introduce missing values by removing random rows
    num_missing = int(len(df) * missing_rate)
    if num_missing > 0:
        missing_indices = np.random.choice(len(df), num_missing, replace=False)
        df = df.drop(df.index[missing_indices]).reset_index(drop=True)
        logger.info(f"Removed {num_missing} rows to simulate missing data")

    logger.info(f"Generated {len(df)} total records")

    return df


def save_sample_data(
    df: pd.DataFrame,
    output_path: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Save generated data to CSV file.

    Args:
        df: DataFrame containing generated sales data
        output_path: Full path to output CSV file
        config: Configuration dictionary

    Returns:
        Metadata dictionary with generation statistics
    """
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)

    # Calculate metadata
    metadata = {
        "file_path": str(output_path),
        "num_records": len(df),
        "num_products": df["product_id"].nunique(),
        "date_range": {
            "start": df["date"].min(),
            "end": df["date"].max()
        },
        "columns": list(df.columns),
        "generated_at": datetime.now().isoformat(),
        "config_used": config.get("sample_data", {})
    }

    logger.info(f"Saved sample data to {output_path}")
    logger.info(f"Records: {metadata['num_records']}, Products: {metadata['num_products']}")

    return metadata


def main() -> None:
    """
    Main entry point for sample data generation.

    Loads configuration, generates data, and saves to configured output path.
    """
    try:
        # Load configuration
        config = load_config()

        # Generate sample data
        df = generate_sample_data(config)

        # Determine output path
        project_root = Path(__file__).parent.parent
        raw_data_path = config.get("paths", {}).get("raw_data", "data/raw")
        output_path = project_root / raw_data_path / "sample_sales_data.csv"

        # Save data
        metadata = save_sample_data(df, str(output_path), config)

        logger.info("Sample data generation completed successfully")
        logger.info(f"Output file: {metadata['file_path']}")

    except Exception as e:
        logger.error(f"Sample data generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
