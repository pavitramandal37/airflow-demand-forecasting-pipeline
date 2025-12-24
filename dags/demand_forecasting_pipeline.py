"""
Demand Forecasting Pipeline DAG
===============================

Production-grade Apache Airflow DAG for demand forecasting using Prophet.

This DAG orchestrates the complete forecasting workflow:
1. Data Extraction - Load and validate raw sales data
2. Data Validation & Cleaning - Quality checks and data transformation
3. Feature Engineering - Create time-series features
4. Model Training - Train Prophet model with versioning
5. Prediction Generation - Generate forecasts with confidence intervals
6. Execution Summary - Aggregate results and generate report

Design Principles:
- Modular tasks with clear responsibilities
- XCom for lightweight metadata passing (no DataFrames)
- Configuration-driven behavior
- Comprehensive error handling and logging
- Idempotent operations for safe retries

Author: Data Engineering Team
Version: 1.0.0
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from airflow import DAG
from airflow.operators.python import PythonOperator
import yaml

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import pipeline scripts
from scripts.data_extraction import extract_data
from scripts.data_transformation import validate_and_clean_data
from scripts.feature_engineering import engineer_features
from scripts.model_training import train_and_save_model
from scripts.prediction_generator import generate_predictions

# Configure logging
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load pipeline configuration from YAML file."""
    config_path = PROJECT_ROOT / "config" / "pipeline_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Load configuration
CONFIG = load_config()
AIRFLOW_CONFIG = CONFIG.get("airflow", {})


# =============================================================================
# Task Functions
# =============================================================================
# Each task function follows the pattern:
# 1. Load configuration
# 2. Determine file paths
# 3. Execute core logic (from scripts module)
# 4. Return metadata for XCom


def task_extract_data(**context) -> Dict[str, Any]:
    """
    Task: Extract and validate raw data.

    Reads raw sales data from CSV and performs initial validation:
    - File existence check
    - Schema validation
    - Basic statistics computation

    Returns:
        Metadata dictionary pushed to XCom
    """
    logger.info("Starting data extraction task")

    config = load_config()
    raw_data_path = config.get("paths", {}).get("raw_data", "data/raw")
    file_path = PROJECT_ROOT / raw_data_path / "sample_sales_data.csv"

    metadata = extract_data(str(file_path), config)

    logger.info(f"Extraction complete: {metadata['num_records']} records")
    return metadata


def task_validate_and_clean(**context) -> Dict[str, Any]:
    """
    Task: Validate data quality and perform cleaning.

    Performs comprehensive data quality validation:
    - Null rate threshold check
    - No negative sales values
    - Date continuity (no gaps > 7 days)
    - Minimum record count

    Then cleans the data:
    - Outlier handling using IQR capping
    - Missing value imputation
    - Sorting by date

    Raises:
        ValueError: If data quality validation fails

    Returns:
        Metadata dictionary with validation and cleaning results
    """
    logger.info("Starting data validation and cleaning task")

    config = load_config()
    raw_data_path = config.get("paths", {}).get("raw_data", "data/raw")
    processed_path = config.get("paths", {}).get("processed_data", "data/processed")

    input_path = PROJECT_ROOT / raw_data_path / "sample_sales_data.csv"
    output_path = PROJECT_ROOT / processed_path / "cleaned_data.csv"

    metadata = validate_and_clean_data(str(input_path), str(output_path), config)

    # Log validation summary
    validation = metadata.get("validation", {})
    logger.info(
        f"Validation: {len(validation.get('checks_passed', []))} checks passed, "
        f"null rate: {validation.get('metrics', {}).get('overall_null_rate', 'N/A')}"
    )

    return metadata


def task_feature_engineering(**context) -> Dict[str, Any]:
    """
    Task: Engineer features for time-series forecasting.

    Creates features including:
    - Time-based features (day of week, month, etc.)
    - Rolling averages (7, 14, 30 day windows)
    - Lag features (1, 7, 14 day lags)
    - Prophet-compatible format (ds, y columns)

    Returns:
        Metadata dictionary with feature engineering details
    """
    logger.info("Starting feature engineering task")

    config = load_config()
    processed_path = config.get("paths", {}).get("processed_data", "data/processed")

    input_path = PROJECT_ROOT / processed_path / "cleaned_data.csv"
    output_path = PROJECT_ROOT / processed_path / "features.csv"

    metadata = engineer_features(str(input_path), str(output_path), config)

    logger.info(f"Feature engineering complete: {metadata['num_features']} features added")
    return metadata


def task_train_model(**context) -> Dict[str, Any]:
    """
    Task: Train Prophet model with versioning.

    Trains a Prophet forecasting model with:
    - Versioned model naming: prophet_model_vYYYYMMDD_<datahash>.pkl
    - Metadata JSON saved alongside model
    - Training duration tracking

    The model version includes:
    - Date-based identifier
    - Data hash for reproducibility
    - Hyperparameter configuration

    Returns:
        Metadata dictionary with model training details
    """
    logger.info("Starting model training task")

    config = load_config()
    processed_path = config.get("paths", {}).get("processed_data", "data/processed")
    models_path = config.get("paths", {}).get("models", "models/saved_models")

    data_path = PROJECT_ROOT / processed_path / "features.csv"
    models_dir = PROJECT_ROOT / models_path

    metadata = train_and_save_model(str(data_path), str(models_dir), config)

    logger.info(
        f"Model training complete: {metadata['version']}, "
        f"duration: {metadata['training_duration_sec']}s"
    )
    return metadata


def task_generate_forecast(**context) -> Dict[str, Any]:
    """
    Task: Generate forecasts using trained model.

    Loads the latest trained model and generates:
    - Point forecasts (yhat)
    - Confidence intervals (yhat_lower, yhat_upper)
    - Trend and seasonality components

    Returns:
        Metadata dictionary with forecast details
    """
    logger.info("Starting forecast generation task")

    config = load_config()
    models_path = config.get("paths", {}).get("models", "models/saved_models")
    predictions_path = config.get("paths", {}).get("predictions", "data/predictions")

    models_dir = PROJECT_ROOT / models_path
    output_path = PROJECT_ROOT / predictions_path / "forecast.csv"

    metadata = generate_predictions(str(models_dir), str(output_path), config)

    logger.info(
        f"Forecast generation complete: {metadata['total_predictions']} predictions, "
        f"horizon: {metadata['horizon_days']} days"
    )
    return metadata


def task_notify_summary(**context) -> Dict[str, Any]:
    """
    Task: Generate execution summary report.

    Aggregates metadata from all previous tasks via XCom and generates
    a comprehensive execution report including:
    - Pipeline execution status
    - Data quality metrics
    - Feature engineering summary
    - Model training details
    - Forecast statistics

    This task can be extended to send notifications via:
    - Slack webhook
    - Email
    - External monitoring systems

    Returns:
        Complete execution summary dictionary
    """
    logger.info("Generating execution summary")

    ti = context["ti"]

    # Pull metadata from all upstream tasks
    extraction_meta = ti.xcom_pull(task_ids="extract_data")
    validation_meta = ti.xcom_pull(task_ids="validate_and_clean")
    features_meta = ti.xcom_pull(task_ids="feature_engineering")
    training_meta = ti.xcom_pull(task_ids="train_model")
    forecast_meta = ti.xcom_pull(task_ids="generate_forecast")

    # Build execution summary
    summary = {
        "pipeline_name": CONFIG.get("pipeline", {}).get("name", "demand_forecasting_pipeline"),
        "pipeline_version": CONFIG.get("pipeline", {}).get("version", "1.0.0"),
        "execution_date": str(context["execution_date"]),
        "run_id": context["run_id"],
        "completed_at": datetime.now().isoformat(),
        "status": "SUCCESS",
        "stages": {
            "extraction": {
                "records_extracted": extraction_meta.get("num_records") if extraction_meta else None,
                "products": extraction_meta.get("num_products") if extraction_meta else None,
                "date_range": extraction_meta.get("date_range") if extraction_meta else None
            },
            "validation": {
                "checks_passed": len(validation_meta.get("validation", {}).get("checks_passed", [])) if validation_meta else None,
                "null_rate": validation_meta.get("validation", {}).get("metrics", {}).get("overall_null_rate") if validation_meta else None,
                "rows_cleaned": validation_meta.get("cleaning", {}).get("rows_removed") if validation_meta else None
            },
            "feature_engineering": {
                "features_added": features_meta.get("num_features") if features_meta else None,
                "records_processed": features_meta.get("engineered_records") if features_meta else None
            },
            "model_training": {
                "model_version": training_meta.get("version") if training_meta else None,
                "training_duration_sec": training_meta.get("training_duration_sec") if training_meta else None,
                "training_records": training_meta.get("training_records") if training_meta else None
            },
            "forecasting": {
                "horizon_days": forecast_meta.get("horizon_days") if forecast_meta else None,
                "total_predictions": forecast_meta.get("total_predictions") if forecast_meta else None,
                "forecast_statistics": forecast_meta.get("forecast_statistics") if forecast_meta else None
            }
        }
    }

    # Generate human-readable report
    report_lines = [
        "=" * 70,
        "DEMAND FORECASTING PIPELINE - EXECUTION SUMMARY",
        "=" * 70,
        f"Pipeline: {summary['pipeline_name']} v{summary['pipeline_version']}",
        f"Execution Date: {summary['execution_date']}",
        f"Run ID: {summary['run_id']}",
        f"Completed At: {summary['completed_at']}",
        f"Status: {summary['status']}",
        "",
        "-" * 70,
        "STAGE RESULTS",
        "-" * 70,
        "",
        "1. DATA EXTRACTION",
        f"   - Records extracted: {summary['stages']['extraction']['records_extracted']}",
        f"   - Products: {summary['stages']['extraction']['products']}",
        "",
        "2. DATA VALIDATION & CLEANING",
        f"   - Quality checks passed: {summary['stages']['validation']['checks_passed']}",
        f"   - Null rate: {summary['stages']['validation']['null_rate']}",
        f"   - Rows cleaned/removed: {summary['stages']['validation']['rows_cleaned']}",
        "",
        "3. FEATURE ENGINEERING",
        f"   - Features added: {summary['stages']['feature_engineering']['features_added']}",
        f"   - Records processed: {summary['stages']['feature_engineering']['records_processed']}",
        "",
        "4. MODEL TRAINING",
        f"   - Model version: {summary['stages']['model_training']['model_version']}",
        f"   - Training duration: {summary['stages']['model_training']['training_duration_sec']}s",
        f"   - Training records: {summary['stages']['model_training']['training_records']}",
        "",
        "5. FORECAST GENERATION",
        f"   - Horizon: {summary['stages']['forecasting']['horizon_days']} days",
        f"   - Total predictions: {summary['stages']['forecasting']['total_predictions']}",
    ]

    # Add forecast statistics if available
    if summary['stages']['forecasting']['forecast_statistics']:
        stats = summary['stages']['forecasting']['forecast_statistics']
        report_lines.extend([
            f"   - Mean forecast: {stats.get('mean_forecast')}",
            f"   - Total forecast: {stats.get('total_forecast')}",
            f"   - Confidence width: {stats.get('avg_confidence_width')}",
        ])

    report_lines.extend([
        "",
        "=" * 70,
        "PIPELINE EXECUTION COMPLETED SUCCESSFULLY",
        "=" * 70
    ])

    # Log the full report
    report = "\n".join(report_lines)
    logger.info("\n" + report)

    # Store report in summary
    summary["report"] = report

    # Future: Add notification integrations here
    # - send_slack_notification(summary)
    # - send_email_notification(summary)
    # - publish_to_monitoring(summary)

    return summary


# =============================================================================
# DAG Definition
# =============================================================================

# Default arguments for all tasks
default_args = {
    "owner": AIRFLOW_CONFIG.get("default_args", {}).get("owner", "data-team"),
    "depends_on_past": False,
    "email_on_failure": AIRFLOW_CONFIG.get("default_args", {}).get("email_on_failure", False),
    "email_on_retry": AIRFLOW_CONFIG.get("default_args", {}).get("email_on_retry", False),
    "retries": AIRFLOW_CONFIG.get("default_args", {}).get("retries", 2),
    "retry_delay": timedelta(
        seconds=AIRFLOW_CONFIG.get("default_args", {}).get("retry_delay_seconds", 300)
    ),
}

# Parse start date from config or use default
start_date_str = AIRFLOW_CONFIG.get("start_date", "2024-01-01")
start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

# Create DAG
with DAG(
    dag_id=AIRFLOW_CONFIG.get("dag_id", "demand_forecasting_pipeline"),
    default_args=default_args,
    description="End-to-end demand forecasting pipeline using Prophet",
    schedule_interval=AIRFLOW_CONFIG.get("schedule_interval", "@daily"),
    start_date=start_date,
    catchup=AIRFLOW_CONFIG.get("catchup", False),
    max_active_runs=AIRFLOW_CONFIG.get("max_active_runs", 1),
    tags=AIRFLOW_CONFIG.get("tags", ["forecasting", "demand", "ml"]),
) as dag:

    # Task 1: Extract Data
    extract_data_task = PythonOperator(
        task_id="extract_data",
        python_callable=task_extract_data,
        doc_md="""
        ### Extract Data
        Loads raw sales data from CSV and performs initial validation.

        **Inputs:** `data/raw/sample_sales_data.csv`
        **Outputs:** Extraction metadata (pushed to XCom)
        """,
    )

    # Task 2: Validate and Clean
    validate_clean_task = PythonOperator(
        task_id="validate_and_clean",
        python_callable=task_validate_and_clean,
        doc_md="""
        ### Validate and Clean Data
        Performs data quality validation and cleaning operations.

        **Validation Checks:**
        - Null rate threshold
        - No negative sales
        - Date continuity
        - Minimum record count

        **Cleaning Operations:**
        - Outlier capping (IQR method)
        - Null imputation
        - Date sorting

        **Inputs:** `data/raw/sample_sales_data.csv`
        **Outputs:** `data/processed/cleaned_data.csv`
        """,
    )

    # Task 3: Feature Engineering
    feature_engineering_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=task_feature_engineering,
        doc_md="""
        ### Feature Engineering
        Creates time-series features for Prophet model.

        **Features Created:**
        - Time-based: day_of_week, month, quarter, etc.
        - Rolling averages: 7, 14, 30 day windows
        - Lag features: 1, 7, 14 day lags
        - Prophet format: ds, y columns

        **Inputs:** `data/processed/cleaned_data.csv`
        **Outputs:** `data/processed/features.csv`
        """,
    )

    # Task 4: Train Model
    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
        doc_md="""
        ### Train Model
        Trains Prophet forecasting model with versioning.

        **Model Features:**
        - Versioned naming: `prophet_model_vYYYYMMDD_<hash>.pkl`
        - Metadata JSON saved alongside model
        - Configurable hyperparameters

        **Inputs:** `data/processed/features.csv`
        **Outputs:** `models/saved_models/prophet_model_v*.pkl`
        """,
    )

    # Task 5: Generate Forecast
    generate_forecast_task = PythonOperator(
        task_id="generate_forecast",
        python_callable=task_generate_forecast,
        doc_md="""
        ### Generate Forecast
        Generates demand forecasts using trained model.

        **Outputs Include:**
        - Point forecasts (yhat)
        - Confidence intervals
        - Trend/seasonality components

        **Inputs:** Latest model from `models/saved_models/`
        **Outputs:** `data/predictions/forecast.csv`
        """,
    )

    # Task 6: Execution Summary
    notify_summary_task = PythonOperator(
        task_id="notify_summary",
        python_callable=task_notify_summary,
        doc_md="""
        ### Execution Summary
        Aggregates metadata and generates execution report.

        **Report Includes:**
        - Pipeline status
        - Data quality metrics
        - Model training details
        - Forecast statistics

        Can be extended for Slack/email notifications.
        """,
    )

    # Define task dependencies
    # Linear pipeline: extract -> validate -> features -> train -> forecast -> summary
    (
        extract_data_task
        >> validate_clean_task
        >> feature_engineering_task
        >> train_model_task
        >> generate_forecast_task
        >> notify_summary_task
    )
