# Airflow Demand Forecasting Pipeline

A production-grade Apache Airflow pipeline for retail demand forecasting using Meta's Prophet library. This project demonstrates MLOps best practices, data quality validation, and modular Python design patterns.

## Overview

This pipeline processes historical sales data to generate demand forecasts, implementing a complete ML workflow:

```
Raw Data → Validation → Feature Engineering → Model Training → Forecasting
```

### Key Features

- **Modular Architecture**: Standalone Python scripts that work independently of Airflow
- **Data Quality Gates**: Configurable validation thresholds that fail fast on bad data
- **Model Versioning**: Automatic versioning with data lineage tracking
- **Production-Ready**: Comprehensive logging, error handling, and observability
- **Configuration-Driven**: YAML-based configuration for all pipeline parameters

## Architecture

<img width="1093" height="727" alt="image" src="https://github.com/user-attachments/assets/ce522751-bf38-48c3-891e-11e7889442a7" />

## Project Structure

```
airflow-demand-forecasting/
├── dags/
│   └── demand_forecasting_pipeline.py   # Airflow DAG definition
├── data/
│   ├── raw/                             # Raw input data
│   ├── processed/                       # Cleaned & engineered data
│   └── predictions/                     # Generated forecasts
├── scripts/
│   ├── __init__.py
│   ├── generate_sample_data.py          # Synthetic data generator
│   ├── data_extraction.py               # Data loading module
│   ├── data_transformation.py           # Validation & cleaning
│   ├── feature_engineering.py           # Feature creation
│   ├── model_training.py                # Prophet training
│   └── prediction_generator.py          # Forecast generation
├── models/
│   └── saved_models/                    # Versioned model storage
├── config/
│   └── pipeline_config.yaml             # Pipeline configuration
├── logs/                                # Runtime logs
├── tests/
│   ├── __init__.py
│   └── test_data_validation.py          # Validation tests
├── docs/
│   └── architecture.md                  # Detailed architecture docs
├── README.md
├── requirements.txt
├── .gitignore
└── setup.sh                             # Environment setup script
```

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd airflow-demand-forecasting
   ```

2. **Run setup script**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   This will:
   - Create a Python virtual environment
   - Install dependencies
   - Initialize Airflow database
   - Generate sample data

3. **Start Airflow services**

   In terminal 1 (webserver):
   ```bash
   source venv/bin/activate
   export AIRFLOW_HOME=$(pwd)
   airflow webserver --port 8080
   ```

   In terminal 2 (scheduler):
   ```bash
   source venv/bin/activate
   export AIRFLOW_HOME=$(pwd)
   airflow scheduler
   ```

4. **Access Airflow UI**

   Open http://localhost:8080
   - Username: `admin`
   - Password: `admin`

5. **Trigger the pipeline**

   Enable and trigger the `demand_forecasting_pipeline` DAG from the UI.

### Running Scripts Standalone

Each script can be run independently for testing and debugging:

```bash
# Generate sample data
python -m scripts.generate_sample_data

# Run data extraction
python -m scripts.data_extraction

# Run validation and cleaning
python -m scripts.data_transformation

# Run feature engineering
python -m scripts.feature_engineering

# Train model
python -m scripts.model_training

# Generate predictions
python -m scripts.prediction_generator
```

## Configuration

All pipeline parameters are centralized in `config/pipeline_config.yaml`:

### Data Quality Thresholds

```yaml
data_quality:
  max_null_rate: 0.05          # Max 5% null values allowed
  min_records: 90              # Minimum 90 days of data required
  max_date_gap_days: 7         # No gaps > 7 days allowed
  allow_negative_sales: false  # Reject negative sales values
  outlier_iqr_multiplier: 1.5  # IQR multiplier for outlier detection
```

### Feature Engineering

```yaml
features:
  rolling_windows: [7, 14, 30]  # Rolling average windows (days)
  lag_days: [1, 7, 14]          # Lag feature periods
  extract_time_features: true   # Extract day_of_week, month, etc.
```

### Model Parameters

```yaml
model:
  type: "prophet"
  horizon_days: 30              # Forecast 30 days ahead
  prophet:
    changepoint_prior_scale: 0.05
    seasonality_prior_scale: 10.0
    seasonality_mode: "multiplicative"
    weekly_seasonality: true
    yearly_seasonality: true
    interval_width: 0.95        # 95% confidence intervals
```

### Airflow Settings

```yaml
airflow:
  dag_id: "demand_forecasting_pipeline"
  schedule_interval: "@daily"
  catchup: false
  default_args:
    retries: 2
    retry_delay_seconds: 300
```

## Data Quality Validation

The pipeline implements strict data quality checks that **fail the pipeline** if thresholds are exceeded:

| Check | Threshold | Behavior |
|-------|-----------|----------|
| Null Rate | ≤ 5% | Pipeline fails if exceeded |
| Negative Sales | 0 allowed | Pipeline fails if found |
| Date Gaps | ≤ 7 days | Pipeline fails if larger gap found |
| Minimum Records | ≥ 90 | Pipeline fails if insufficient data |

This fail-fast approach prevents bad data from propagating through the pipeline.

## Model Versioning

Models are saved with automatic versioning:

```
prophet_model_v20240115_a1b2c3d4.pkl
               ^^^^^^^^ ^^^^^^^^
               │        └── Data hash (first 8 chars)
               └── Training date (YYYYMMDD)
```

Each model includes a metadata JSON file with:
- Training timestamp
- Data hash for reproducibility
- Hyperparameters used
- Training duration
- Forecast horizon

## XCom Usage

The pipeline uses Airflow XCom for passing **metadata only** (not DataFrames):

```python
# ✅ Good: Pass metadata
return {
    "num_records": 365,
    "file_path": "data/processed/cleaned_data.csv",
    "data_hash": "a1b2c3d4"
}

# ❌ Bad: Never pass DataFrames via XCom
# return df  # Don't do this!
```

Actual data stays on disk; only lightweight metadata is passed between tasks.

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=scripts --cov-report=html

# Run specific test
pytest tests/test_data_validation.py -v
```

## Design Decisions

### Why Modular Scripts?

1. **Testability**: Each module can be tested independently
2. **Reusability**: Scripts work outside Airflow for debugging
3. **Maintainability**: Clear separation of concerns
4. **Flexibility**: Easy to swap out components

### Why Prophet?

1. **Robust to missing data**: Handles gaps automatically
2. **Strong seasonality handling**: Captures weekly/yearly patterns
3. **Interpretable**: Components can be visualized
4. **Production-proven**: Used at scale by Meta

### Why YAML Configuration?

1. **No code changes** for parameter tuning
2. **Environment-specific** configs possible
3. **Readable** by non-developers
4. **Version-controllable** separately

## Future Enhancements

- [ ] Multi-product forecasting with separate models
- [ ] Model performance tracking (MAE, MAPE metrics)
- [ ] Automated retraining based on forecast accuracy
- [ ] Slack/email notifications on pipeline completion
- [ ] Data drift detection between training runs
- [ ] A/B testing for model versions
- [ ] Integration with feature stores
- [ ] GPU acceleration for larger datasets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [Apache Airflow](https://airflow.apache.org/) - Workflow orchestration
- [Meta Prophet](https://facebook.github.io/prophet/) - Time-series forecasting
- [Pandas](https://pandas.pydata.org/) - Data manipulation
