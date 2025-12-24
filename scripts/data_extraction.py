#!/usr/bin/env python3
"""
Data Extraction Module
======================

Handles reading and initial validation of raw data files. This module is the
entry point for data ingestion in the forecasting pipeline.

Responsibilities:
- Validate file existence and accessibility
- Read CSV data with appropriate dtypes
- Perform basic schema validation
- Return extraction metadata for downstream tasks

The module is designed to fail fast on data access issues, preventing
downstream tasks from running with incomplete or missing data.

Usage:
------
    from scripts.data_extraction import extract_data

    metadata = extract_data(
        file_path="data/raw/sample_sales_data.csv",
        config=config
    )
"""

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Expected schema for sales data
EXPECTED_COLUMNS = ["date", "product_id", "sales_quantity", "revenue"]


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load pipeline configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to config/pipeline_config.yaml

    Returns:
        Dictionary containing configuration parameters
    """
    if config_path is None:
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "pipeline_config.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_file_exists(file_path: str) -> None:
    """
    Validate that the specified file exists and is accessible.

    Args:
        file_path: Path to the file to validate

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file is not readable
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}. "
            "Ensure sample data has been generated or provide a valid data path."
        )

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if not os.access(path, os.R_OK):
        raise PermissionError(f"File is not readable: {file_path}")

    logger.info(f"File validation passed: {file_path}")


def validate_schema(df: pd.DataFrame, expected_columns: List[str]) -> None:
    """
    Validate that DataFrame contains all expected columns.

    Args:
        df: DataFrame to validate
        expected_columns: List of required column names

    Raises:
        ValueError: If required columns are missing
    """
    missing_columns = set(expected_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Found columns: {list(df.columns)}"
        )

    logger.info(f"Schema validation passed. Columns: {list(df.columns)}")


def compute_data_hash(df: pd.DataFrame) -> str:
    """
    Compute a hash of the DataFrame for data versioning.

    Uses MD5 hash of the DataFrame's string representation.
    This hash can be used to detect data changes and version models.

    Args:
        df: DataFrame to hash

    Returns:
        Hexadecimal hash string (first 8 characters for brevity)
    """
    # Convert to string and hash
    data_str = df.to_csv(index=False)
    full_hash = hashlib.md5(data_str.encode()).hexdigest()

    # Return first 8 characters for brevity
    return full_hash[:8]


def extract_data(
    file_path: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract and validate raw sales data from CSV file.

    This function performs:
    1. File existence validation
    2. CSV reading with appropriate data types
    3. Schema validation (required columns)
    4. Basic statistics computation

    Args:
        file_path: Path to the raw data CSV file
        config: Optional configuration dictionary

    Returns:
        Metadata dictionary containing:
        - file_path: Path to the extracted data
        - num_records: Number of records in the file
        - num_products: Number of unique products
        - date_range: Start and end dates in the data
        - data_hash: Hash for versioning
        - extracted_at: Timestamp of extraction

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If schema validation fails
    """
    if config is None:
        config = load_config()

    logger.info(f"Starting data extraction from {file_path}")

    # Step 1: Validate file exists
    validate_file_exists(file_path)

    # Step 2: Read CSV with appropriate dtypes
    # Parse dates explicitly for proper handling
    df = pd.read_csv(
        file_path,
        dtype={
            "product_id": str,
            "sales_quantity": int,
            "revenue": float
        },
        parse_dates=["date"]
    )

    logger.info(f"Read {len(df)} records from {file_path}")

    # Step 3: Validate schema
    validate_schema(df, EXPECTED_COLUMNS)

    # Step 4: Compute statistics and metadata
    data_hash = compute_data_hash(df)

    # Sort dates for consistent range calculation
    df_sorted = df.sort_values("date")

    metadata = {
        "file_path": str(file_path),
        "num_records": len(df),
        "num_products": df["product_id"].nunique(),
        "products": sorted(df["product_id"].unique().tolist()),
        "date_range": {
            "start": df_sorted["date"].min().strftime("%Y-%m-%d"),
            "end": df_sorted["date"].max().strftime("%Y-%m-%d"),
            "num_days": (df_sorted["date"].max() - df_sorted["date"].min()).days + 1
        },
        "data_hash": data_hash,
        "extracted_at": datetime.now().isoformat(),
        "statistics": {
            "total_sales": int(df["sales_quantity"].sum()),
            "total_revenue": round(float(df["revenue"].sum()), 2),
            "avg_daily_sales": round(float(df["sales_quantity"].mean()), 2),
            "avg_revenue": round(float(df["revenue"].mean()), 2)
        }
    }

    logger.info(
        f"Extraction complete. Records: {metadata['num_records']}, "
        f"Products: {metadata['num_products']}, "
        f"Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}"
    )

    return metadata


def main() -> None:
    """
    Main entry point for standalone data extraction.
    """
    try:
        config = load_config()
        project_root = Path(__file__).parent.parent
        raw_data_path = config.get("paths", {}).get("raw_data", "data/raw")
        file_path = project_root / raw_data_path / "sample_sales_data.csv"

        metadata = extract_data(str(file_path), config)

        logger.info("Data extraction completed successfully")
        for key, value in metadata.items():
            logger.info(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Data extraction failed: {e}")
        raise


if __name__ == "__main__":
    main()
