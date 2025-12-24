#!/usr/bin/env python3
"""
Data Transformation Module
==========================

Handles data quality validation and cleaning operations. This module ensures
data meets business requirements before being used for model training.

Key Features:
- Comprehensive data quality validation with configurable thresholds
- Outlier detection and handling using IQR method
- Missing value imputation strategies
- Date continuity verification

The validation rules are designed to catch common data issues:
- Null values beyond acceptable thresholds
- Negative sales (business logic violation)
- Date gaps indicating data collection failures
- Insufficient data for reliable forecasting

Usage:
------
    from scripts.data_transformation import validate_and_clean_data

    metadata = validate_and_clean_data(
        input_path="data/raw/sample_sales_data.csv",
        output_path="data/processed/cleaned_data.csv",
        config=config
    )
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


def validate_data_quality(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Perform comprehensive data quality validation.

    Validates:
    1. Null rate is within acceptable threshold
    2. No negative sales values (unless explicitly allowed)
    3. Date continuity (no gaps exceeding threshold)
    4. Minimum record count for meaningful analysis

    Args:
        df: DataFrame to validate
        config: Configuration with data_quality thresholds

    Returns:
        Dictionary containing validation results and metrics

    Raises:
        ValueError: If any validation check fails
    """
    dq_config = config.get("data_quality", {})

    max_null_rate = dq_config.get("max_null_rate", 0.05)
    min_records = dq_config.get("min_records", 90)
    max_date_gap = dq_config.get("max_date_gap_days", 7)
    allow_negative = dq_config.get("allow_negative_sales", False)

    validation_results = {
        "checks_performed": [],
        "checks_passed": [],
        "checks_failed": [],
        "metrics": {}
    }

    logger.info("Starting data quality validation")

    # Check 1: Minimum record count
    check_name = "minimum_records"
    validation_results["checks_performed"].append(check_name)

    if len(df) < min_records:
        validation_results["checks_failed"].append(check_name)
        raise ValueError(
            f"Insufficient records: {len(df)} < {min_records} minimum required. "
            "Need more historical data for reliable forecasting."
        )
    validation_results["checks_passed"].append(check_name)
    validation_results["metrics"]["record_count"] = len(df)
    logger.info(f"✓ Record count check passed: {len(df)} >= {min_records}")

    # Check 2: Null rate threshold
    check_name = "null_rate"
    validation_results["checks_performed"].append(check_name)

    # Calculate null rate per column and overall
    null_counts = df.isnull().sum()
    total_cells = len(df) * len(df.columns)
    total_nulls = null_counts.sum()
    overall_null_rate = total_nulls / total_cells if total_cells > 0 else 0

    # Per-column null rates
    column_null_rates = (null_counts / len(df)).to_dict()
    validation_results["metrics"]["null_rates"] = column_null_rates
    validation_results["metrics"]["overall_null_rate"] = round(overall_null_rate, 4)

    if overall_null_rate > max_null_rate:
        validation_results["checks_failed"].append(check_name)
        raise ValueError(
            f"Null rate {overall_null_rate:.2%} exceeds threshold {max_null_rate:.2%}. "
            f"Column null rates: {column_null_rates}"
        )
    validation_results["checks_passed"].append(check_name)
    logger.info(f"✓ Null rate check passed: {overall_null_rate:.2%} <= {max_null_rate:.2%}")

    # Check 3: Negative sales values
    check_name = "no_negative_sales"
    validation_results["checks_performed"].append(check_name)

    if "sales_quantity" in df.columns:
        negative_count = (df["sales_quantity"] < 0).sum()
        validation_results["metrics"]["negative_sales_count"] = int(negative_count)

        if not allow_negative and negative_count > 0:
            validation_results["checks_failed"].append(check_name)
            raise ValueError(
                f"Found {negative_count} negative sales values. "
                "Negative sales are not allowed per business rules. "
                "Set allow_negative_sales: true in config to override."
            )
        validation_results["checks_passed"].append(check_name)
        logger.info(f"✓ Negative sales check passed: {negative_count} negative values")

    # Check 4: Date continuity
    check_name = "date_continuity"
    validation_results["checks_performed"].append(check_name)

    if "date" in df.columns:
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        # Get unique dates sorted
        unique_dates = df["date"].drop_duplicates().sort_values()

        if len(unique_dates) > 1:
            # Calculate gaps between consecutive dates
            date_gaps = unique_dates.diff().dropna()
            max_gap_found = date_gaps.max().days

            validation_results["metrics"]["max_date_gap_days"] = max_gap_found
            validation_results["metrics"]["num_unique_dates"] = len(unique_dates)

            if max_gap_found > max_date_gap:
                # Find where the gap occurred
                gap_idx = date_gaps.idxmax()
                gap_start = unique_dates[unique_dates.index.get_loc(gap_idx) - 1]
                gap_end = unique_dates.loc[gap_idx]

                validation_results["checks_failed"].append(check_name)
                raise ValueError(
                    f"Date gap of {max_gap_found} days exceeds threshold of {max_date_gap} days. "
                    f"Gap found between {gap_start.strftime('%Y-%m-%d')} and "
                    f"{gap_end.strftime('%Y-%m-%d')}. "
                    "This may indicate data collection issues."
                )

            validation_results["checks_passed"].append(check_name)
            logger.info(f"✓ Date continuity check passed: max gap {max_gap_found} <= {max_date_gap} days")

    # All checks passed
    validation_results["validation_passed"] = True
    validation_results["validated_at"] = datetime.now().isoformat()

    logger.info(
        f"Data quality validation complete. "
        f"Passed: {len(validation_results['checks_passed'])}/{len(validation_results['checks_performed'])} checks"
    )

    return validation_results


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Detect outliers using the Interquartile Range (IQR) method.

    Outliers are values below Q1 - multiplier*IQR or above Q3 + multiplier*IQR.

    Args:
        df: DataFrame containing the data
        column: Column name to check for outliers
        multiplier: IQR multiplier (1.5 = mild outliers, 3.0 = extreme outliers)

    Returns:
        Tuple of (boolean mask of outliers, statistics dictionary)
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)

    stats = {
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "outlier_count": int(outlier_mask.sum()),
        "outlier_rate": round(outlier_mask.mean(), 4)
    }

    return outlier_mask, stats


def clean_data(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean data by handling outliers, missing values, and ensuring consistency.

    Cleaning operations:
    1. Handle outliers using IQR capping (winsorization)
    2. Fill missing values with appropriate strategies
    3. Sort by date for time-series consistency
    4. Remove any remaining nulls (dropna as last resort)

    Args:
        df: DataFrame to clean
        config: Configuration with cleaning parameters

    Returns:
        Tuple of (cleaned DataFrame, cleaning metadata)
    """
    dq_config = config.get("data_quality", {})
    iqr_multiplier = dq_config.get("outlier_iqr_multiplier", 1.5)

    cleaning_metadata = {
        "original_rows": len(df),
        "operations": [],
        "outlier_handling": {},
        "null_handling": {}
    }

    logger.info(f"Starting data cleaning. Original rows: {len(df)}")

    # Make a copy to avoid modifying original
    df_cleaned = df.copy()

    # Ensure date is datetime
    if "date" in df_cleaned.columns:
        df_cleaned["date"] = pd.to_datetime(df_cleaned["date"])

    # Step 1: Handle outliers in sales_quantity using IQR capping
    if "sales_quantity" in df_cleaned.columns:
        outlier_mask, outlier_stats = detect_outliers_iqr(
            df_cleaned, "sales_quantity", iqr_multiplier
        )
        cleaning_metadata["outlier_handling"]["sales_quantity"] = outlier_stats

        if outlier_stats["outlier_count"] > 0:
            # Cap outliers at bounds (winsorization)
            df_cleaned.loc[
                df_cleaned["sales_quantity"] < outlier_stats["lower_bound"],
                "sales_quantity"
            ] = int(max(0, outlier_stats["lower_bound"]))

            df_cleaned.loc[
                df_cleaned["sales_quantity"] > outlier_stats["upper_bound"],
                "sales_quantity"
            ] = int(outlier_stats["upper_bound"])

            cleaning_metadata["operations"].append({
                "type": "outlier_capping",
                "column": "sales_quantity",
                "count": outlier_stats["outlier_count"]
            })
            logger.info(
                f"Capped {outlier_stats['outlier_count']} outliers in sales_quantity "
                f"to [{outlier_stats['lower_bound']}, {outlier_stats['upper_bound']}]"
            )

    # Step 2: Handle outliers in revenue
    if "revenue" in df_cleaned.columns:
        outlier_mask, outlier_stats = detect_outliers_iqr(
            df_cleaned, "revenue", iqr_multiplier
        )
        cleaning_metadata["outlier_handling"]["revenue"] = outlier_stats

        if outlier_stats["outlier_count"] > 0:
            df_cleaned.loc[
                df_cleaned["revenue"] < outlier_stats["lower_bound"],
                "revenue"
            ] = max(0, outlier_stats["lower_bound"])

            df_cleaned.loc[
                df_cleaned["revenue"] > outlier_stats["upper_bound"],
                "revenue"
            ] = outlier_stats["upper_bound"]

            cleaning_metadata["operations"].append({
                "type": "outlier_capping",
                "column": "revenue",
                "count": outlier_stats["outlier_count"]
            })
            logger.info(
                f"Capped {outlier_stats['outlier_count']} outliers in revenue"
            )

    # Step 3: Handle missing values
    null_before = df_cleaned.isnull().sum().to_dict()
    cleaning_metadata["null_handling"]["before"] = null_before

    # For numeric columns, use median imputation
    for col in ["sales_quantity", "revenue"]:
        if col in df_cleaned.columns and df_cleaned[col].isnull().any():
            median_val = df_cleaned[col].median()
            filled_count = df_cleaned[col].isnull().sum()
            df_cleaned[col] = df_cleaned[col].fillna(median_val)

            cleaning_metadata["operations"].append({
                "type": "null_imputation",
                "column": col,
                "strategy": "median",
                "fill_value": round(median_val, 2),
                "count": int(filled_count)
            })
            logger.info(f"Filled {filled_count} nulls in {col} with median {median_val:.2f}")

    # For date column, drop nulls (can't impute dates meaningfully)
    if "date" in df_cleaned.columns and df_cleaned["date"].isnull().any():
        null_dates = df_cleaned["date"].isnull().sum()
        df_cleaned = df_cleaned.dropna(subset=["date"])
        cleaning_metadata["operations"].append({
            "type": "null_drop",
            "column": "date",
            "count": int(null_dates)
        })
        logger.info(f"Dropped {null_dates} rows with null dates")

    null_after = df_cleaned.isnull().sum().to_dict()
    cleaning_metadata["null_handling"]["after"] = null_after

    # Step 4: Sort by date and product for consistency
    if "date" in df_cleaned.columns:
        sort_columns = ["date"]
        if "product_id" in df_cleaned.columns:
            sort_columns.append("product_id")
        df_cleaned = df_cleaned.sort_values(sort_columns).reset_index(drop=True)
        cleaning_metadata["operations"].append({
            "type": "sort",
            "columns": sort_columns
        })
        logger.info(f"Sorted data by {sort_columns}")

    # Final statistics
    cleaning_metadata["final_rows"] = len(df_cleaned)
    cleaning_metadata["rows_removed"] = cleaning_metadata["original_rows"] - len(df_cleaned)
    cleaning_metadata["cleaned_at"] = datetime.now().isoformat()

    logger.info(
        f"Data cleaning complete. "
        f"Final rows: {len(df_cleaned)} "
        f"(removed {cleaning_metadata['rows_removed']})"
    )

    return df_cleaned, cleaning_metadata


def validate_and_clean_data(
    input_path: str,
    output_path: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main entry point: validate raw data quality and perform cleaning.

    This function orchestrates the full data transformation workflow:
    1. Load raw data
    2. Run quality validation (fails fast if thresholds exceeded)
    3. Clean data (outliers, nulls, sorting)
    4. Save cleaned data
    5. Return metadata for XCom

    Args:
        input_path: Path to raw data CSV
        output_path: Path to save cleaned data CSV
        config: Optional configuration dictionary

    Returns:
        Metadata dictionary containing validation and cleaning results

    Raises:
        ValueError: If data quality validation fails
    """
    if config is None:
        config = load_config()

    logger.info(f"Starting data transformation: {input_path} -> {output_path}")

    # Load raw data
    df = pd.read_csv(input_path, parse_dates=["date"])
    logger.info(f"Loaded {len(df)} records from {input_path}")

    # Run validation (will raise if checks fail)
    validation_results = validate_data_quality(df, config)

    # Clean data
    df_cleaned, cleaning_metadata = clean_data(df, config)

    # Save cleaned data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")

    # Combine metadata
    metadata = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "validation": validation_results,
        "cleaning": cleaning_metadata,
        "processed_at": datetime.now().isoformat()
    }

    return metadata


def main() -> None:
    """
    Main entry point for standalone data transformation.
    """
    try:
        config = load_config()
        project_root = Path(__file__).parent.parent

        raw_path = config.get("paths", {}).get("raw_data", "data/raw")
        processed_path = config.get("paths", {}).get("processed_data", "data/processed")

        input_path = project_root / raw_path / "sample_sales_data.csv"
        output_path = project_root / processed_path / "cleaned_data.csv"

        metadata = validate_and_clean_data(str(input_path), str(output_path), config)

        logger.info("Data transformation completed successfully")

    except Exception as e:
        logger.error(f"Data transformation failed: {e}")
        raise


if __name__ == "__main__":
    main()
