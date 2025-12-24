"""
Airflow Demand Forecasting Pipeline - Scripts Package
======================================================

This package contains modular, reusable Python scripts for the demand
forecasting pipeline. Each module is designed to be:

1. **Independent**: Can run standalone without Airflow for testing/debugging
2. **Testable**: Pure functions with clear inputs/outputs
3. **Configurable**: All parameters loaded from YAML configuration
4. **Observable**: Comprehensive logging for production monitoring

Modules:
--------
- generate_sample_data: Creates synthetic sales data for testing
- data_extraction: Reads and validates raw data files
- data_transformation: Data quality validation and cleaning
- feature_engineering: Time-series feature creation
- model_training: Prophet model training with versioning
- prediction_generator: Forecast generation using trained models

Usage:
------
Each module can be imported and used independently:

    from scripts.data_extraction import extract_data
    from scripts.data_transformation import validate_and_clean_data
    from scripts.model_training import train_and_save_model

Or run directly via command line for testing:

    python -m scripts.generate_sample_data
    python -m scripts.data_extraction
"""

__version__ = "1.0.0"
__author__ = "Data Engineering Team"

# Expose key functions at package level for convenience
from scripts.data_extraction import extract_data
from scripts.data_transformation import validate_data_quality, clean_data
from scripts.feature_engineering import engineer_features
from scripts.model_training import train_and_save_model
from scripts.prediction_generator import generate_predictions
