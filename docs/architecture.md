# Architecture Documentation

## System Overview

The Demand Forecasting Pipeline is designed as a modular, maintainable system following MLOps best practices. This document details the technical architecture, data flow, and design decisions.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                    │
│                                                                              │
│    ┌──────────────────┐                                                     │
│    │ Raw Sales Data   │ CSV files, future: database/API integration         │
│    │ (sample_sales_   │                                                     │
│    │  data.csv)       │                                                     │
│    └────────┬─────────┘                                                     │
│             │                                                                │
└─────────────│───────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION LAYER                                    │
│                                                                              │
│    ┌──────────────────┐    ┌──────────────────┐                            │
│    │  File Validation │───▶│ Schema Validation│                            │
│    │  (exists, size)  │    │ (columns, types) │                            │
│    └──────────────────┘    └────────┬─────────┘                            │
│                                      │                                       │
│    Output: Extraction metadata ◀─────┘                                      │
│                                                                              │
└─────────────────────────────────────│───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VALIDATION LAYER                                   │
│                                                                              │
│    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│    │ Null Rate Check  │    │ Negative Value   │    │ Date Continuity  │   │
│    │ (≤ 5% threshold) │    │ Check            │    │ Check (≤ 7 days) │   │
│    └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘   │
│             │                        │                       │              │
│             └────────────────────────┴───────────────────────┘              │
│                                      │                                       │
│                                      ▼                                       │
│    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│    │ Outlier Handling │    │ Null Imputation  │    │ Data Sorting     │   │
│    │ (IQR Capping)    │    │ (Median fill)    │    │ (by date)        │   │
│    └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘   │
│             │                        │                       │              │
│             └────────────────────────┴───────────────────────┘              │
│                                      │                                       │
│    Output: cleaned_data.csv + validation metrics                            │
│                                                                              │
└─────────────────────────────────────│───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FEATURE ENGINEERING LAYER                             │
│                                                                              │
│    ┌──────────────────┐                                                     │
│    │ Time Features    │ day_of_week, month, quarter, is_weekend, etc.      │
│    └────────┬─────────┘                                                     │
│             │                                                                │
│    ┌────────▼─────────┐                                                     │
│    │ Rolling Features │ 7-day, 14-day, 30-day moving averages              │
│    └────────┬─────────┘                                                     │
│             │                                                                │
│    ┌────────▼─────────┐                                                     │
│    │ Lag Features     │ 1-day, 7-day, 14-day lagged values                 │
│    └────────┬─────────┘                                                     │
│             │                                                                │
│    ┌────────▼─────────┐                                                     │
│    │ Prophet Format   │ Rename to ds (date), y (target)                    │
│    └────────┬─────────┘                                                     │
│             │                                                                │
│    Output: features.csv + feature metadata                                  │
│                                                                              │
└─────────────────────────────────────│───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MODEL TRAINING LAYER                              │
│                                                                              │
│    ┌──────────────────┐                                                     │
│    │ Data Hashing     │ Compute MD5 hash of training data                  │
│    └────────┬─────────┘                                                     │
│             │                                                                │
│    ┌────────▼─────────┐                                                     │
│    │ Version Creation │ prophet_model_vYYYYMMDD_<hash>.pkl                 │
│    └────────┬─────────┘                                                     │
│             │                                                                │
│    ┌────────▼─────────┐                                                     │
│    │ Prophet Training │ Configure hyperparameters from config              │
│    │                  │ Fit model on ds, y columns                         │
│    └────────┬─────────┘                                                     │
│             │                                                                │
│    ┌────────▼─────────┐    ┌──────────────────┐                            │
│    │ Model Saving     │───▶│ Metadata Saving  │                            │
│    │ (pickle)         │    │ (JSON)           │                            │
│    └──────────────────┘    └──────────────────┘                            │
│                                                                              │
│    Output: prophet_model_v*.pkl + *_metadata.json                           │
│                                                                              │
└─────────────────────────────────────│───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PREDICTION LAYER                                   │
│                                                                              │
│    ┌──────────────────┐                                                     │
│    │ Latest Model     │ Find most recent model by modification time        │
│    │ Detection        │                                                     │
│    └────────┬─────────┘                                                     │
│             │                                                                │
│    ┌────────▼─────────┐                                                     │
│    │ Future DataFrame │ Generate dates for forecast horizon                │
│    │ Generation       │                                                     │
│    └────────┬─────────┘                                                     │
│             │                                                                │
│    ┌────────▼─────────┐                                                     │
│    │ Prediction       │ yhat (point), yhat_lower/upper (intervals)         │
│    │ Generation       │ trend, seasonality components                       │
│    └────────┬─────────┘                                                     │
│             │                                                                │
│    Output: forecast.csv + forecast metadata                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Input Data Format

```csv
date,product_id,sales_quantity,revenue
2024-01-01,PROD_001,125,1375.00
2024-01-01,PROD_002,89,979.00
2024-01-02,PROD_001,130,1430.00
...
```

**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| date | date (YYYY-MM-DD) | Transaction date |
| product_id | string | Product identifier |
| sales_quantity | integer | Units sold |
| revenue | float | Total revenue |

### 2. Intermediate Data: Cleaned Data

After validation and cleaning:
- Outliers capped using IQR method
- Missing values imputed with median
- Sorted by date
- No null values remaining

### 3. Intermediate Data: Features

After feature engineering:
```csv
date,sales_quantity,revenue,ds,y,day_of_week,month,...,sales_quantity_rolling_7d,sales_quantity_lag_1d
```

**Added Features:**
- Time features: day_of_week, day_of_month, week_of_year, month, quarter, year, is_weekend, is_month_start, is_month_end
- Rolling features: 7d, 14d, 30d moving averages
- Lag features: 1d, 7d, 14d lags
- Prophet format: ds (date), y (target)

### 4. Model Artifacts

**Model File:** `prophet_model_v20240115_a1b2c3d4.pkl`

**Metadata File:** `prophet_model_v20240115_a1b2c3d4_metadata.json`
```json
{
  "version": "prophet_model_v20240115_a1b2c3d4",
  "created_at": "2024-01-15T10:30:00",
  "data_hash": "a1b2c3d4",
  "model_type": "prophet",
  "training_duration_sec": 12.5,
  "horizon_days": 30,
  "training_records": 365,
  "date_range": {
    "start": "2023-01-15",
    "end": "2024-01-14"
  },
  "prophet_params": {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "seasonality_mode": "multiplicative"
  }
}
```

### 5. Output Data: Forecasts

```csv
ds,yhat,yhat_lower,yhat_upper,trend,weekly,yearly
2024-01-15,145.23,120.50,170.00,140.00,1.05,0.98
2024-01-16,142.10,118.00,166.20,140.50,1.02,0.99
...
```

## DAG Dependencies

```
extract_data
     │
     ▼
validate_and_clean
     │
     ▼
feature_engineering
     │
     ▼
train_model
     │
     ▼
generate_forecast
     │
     ▼
notify_summary
```

**Dependency Rationale:**
- Each task depends on the output of the previous task
- No parallel execution possible due to sequential data dependencies
- Fail-fast design: if validation fails, no downstream tasks run

## Error Handling Strategy

### 1. Validation Failures

```python
# Example: Null rate exceeds threshold
if overall_null_rate > max_null_rate:
    raise ValueError(
        f"Null rate {overall_null_rate:.2%} exceeds threshold {max_null_rate:.2%}. "
        f"Column null rates: {column_null_rates}"
    )
```

**Behavior:** Pipeline fails immediately with descriptive error message.

### 2. File Not Found

```python
if not path.exists():
    raise FileNotFoundError(
        f"Data file not found: {file_path}. "
        "Ensure sample data has been generated."
    )
```

**Behavior:** Clear message indicating missing prerequisite.

### 3. Retry Strategy

Configured in Airflow DAG:
```python
default_args = {
    "retries": 2,
    "retry_delay": timedelta(seconds=300)
}
```

**Behavior:** Tasks retry twice with 5-minute delay on transient failures.

### 4. Logging

All modules use Python's logging module with consistent format:
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

**Log Levels:**
- `INFO`: Normal operation progress
- `WARNING`: Non-fatal issues (e.g., missing optional metadata)
- `ERROR`: Failures requiring attention

## Monitoring Approach

### 1. Task-Level Monitoring

Each task returns metadata to XCom for visibility:
```python
metadata = {
    "num_records": 365,
    "validation_passed": True,
    "processing_time_sec": 2.5,
    ...
}
return metadata
```

### 2. Execution Summary

The `notify_summary` task aggregates all metadata:
```
=====================================================================
DEMAND FORECASTING PIPELINE - EXECUTION SUMMARY
=====================================================================
Pipeline: demand_forecasting_pipeline v1.0.0
Execution Date: 2024-01-15
Status: SUCCESS

STAGE RESULTS
---------------------------------------------------------------------
1. DATA EXTRACTION
   - Records extracted: 1095
   - Products: 3

2. DATA VALIDATION & CLEANING
   - Quality checks passed: 4
   - Null rate: 0.023
   - Rows cleaned/removed: 5

3. FEATURE ENGINEERING
   - Features added: 18
   - Records processed: 365

4. MODEL TRAINING
   - Model version: prophet_model_v20240115_a1b2c3d4
   - Training duration: 12.5s

5. FORECAST GENERATION
   - Horizon: 30 days
   - Mean forecast: 145.23
=====================================================================
```

### 3. Future Monitoring Extensions

The architecture supports adding:
- **Slack notifications**: POST to webhook on completion/failure
- **Email alerts**: SMTP integration for stakeholder updates
- **Metrics export**: Prometheus/StatsD for dashboarding
- **Data quality dashboards**: Visualization of validation metrics over time

## Configuration Management

### Configuration Hierarchy

```
config/pipeline_config.yaml
├── pipeline          # Pipeline metadata
├── paths             # File/directory paths
├── data_quality      # Validation thresholds
├── features          # Feature engineering params
├── model             # Training hyperparameters
├── airflow           # DAG configuration
├── logging           # Log settings
└── sample_data       # Data generation params
```

### Environment-Specific Overrides

The YAML structure supports environment-specific configs:
```bash
# Development
export CONFIG_PATH=config/pipeline_config.yaml

# Production (future)
export CONFIG_PATH=config/pipeline_config.prod.yaml
```

## Scalability Considerations

### Current Limitations

1. **Single-node execution**: LocalExecutor for development
2. **File-based storage**: CSV files for data, pickle for models
3. **Single product aggregation**: Aggregates all products into one forecast

### Future Scaling Paths

1. **Kubernetes Executor**: For distributed task execution
2. **Cloud Storage**: S3/GCS for data and model storage
3. **Per-product models**: Separate models for each product_id
4. **Distributed training**: Ray/Spark for large datasets
5. **Model registry**: MLflow/Vertex AI for model management

## Security Considerations

### Current Implementation

1. **No credentials in code**: Configuration uses relative paths
2. **Read-only operations**: Scripts don't modify source data
3. **Isolated execution**: Each task runs in its own process

### Production Recommendations

1. **Secrets management**: Use Airflow Connections/Variables
2. **IAM roles**: Cloud provider identity management
3. **Encryption**: At-rest encryption for sensitive data
4. **Audit logging**: Track all data access

## Testing Strategy

### Unit Tests

Located in `tests/`:
```
tests/
├── test_data_validation.py    # Validation logic tests
└── (future)
    ├── test_feature_engineering.py
    ├── test_model_training.py
    └── test_integration.py
```

### Test Categories

1. **Validation Tests**: Verify threshold checks work correctly
2. **Feature Tests**: Verify feature creation logic
3. **Model Tests**: Verify model training produces valid artifacts
4. **Integration Tests**: End-to-end pipeline execution

### Running Tests

```bash
# All tests with coverage
pytest tests/ --cov=scripts --cov-report=html

# Specific test file
pytest tests/test_data_validation.py -v
```

## Deployment Checklist

1. [ ] Configure `pipeline_config.yaml` for target environment
2. [ ] Set up Airflow with appropriate executor (Celery/Kubernetes)
3. [ ] Configure cloud storage backend (if applicable)
4. [ ] Set up monitoring/alerting integrations
5. [ ] Run integration tests on sample data
6. [ ] Enable DAG and verify scheduled execution
7. [ ] Set up data ingestion from production sources
