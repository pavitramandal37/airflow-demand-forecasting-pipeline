# Multi-Model Demand Forecasting Pipeline - v2.0.0

## 🎯 Overview

Production-ready multi-model forecasting system combining **Prophet**, **SARIMA**, and **DeepAR** models with intelligent ensemble strategies.

### Key Features

- ✅ **4 Models**: Prophet, SARIMA (per-product), DeepAR (with external features), Ensemble
- ✅ **Auto-Optimization**: Automatic weight optimization for ensemble
- ✅ **Versioning**: Complete model versioning with reproducibility (seed + data hash)
- ✅ **Modular Architecture**: Clean separation of concerns
- ✅ **Production-Ready**: Airflow DAGs, data validation, comprehensive testing
- ✅ **Scalable**: Per-product SARIMA, parallel training support

---

## 📁 Project Structure

```
airflow-demand-forecasting/
│
├── config/                          # Configuration files
│   ├── base_config.yaml            # Shared settings
│   ├── prophet_config.yaml         # Prophet hyperparameters
│   ├── sarima_config.yaml          # SARIMA with per-product support
│   ├── deepar_config.yaml          # DeepAR with external features
│   └── ensemble_config.yaml        # Ensemble strategies
│
├── scripts/                         # Model implementations
│   ├── common/                     # Shared utilities
│   │   ├── config_loader.py       # Hierarchical config loading
│   │   ├── metrics.py             # MAPE, RMSE, MAE, SMAPE
│   │   ├── model_versioning.py   # Version management
│   │   ├── data_validator.py     # Data quality checks
│   │   └── utils.py               # Helper functions
│   │
│   ├── prophet/                    # Prophet model
│   │   ├── model_training.py
│   │   └── prediction.py
│   │
│   ├── sarima/                     # SARIMA model (per-product)
│   │   └── model_training.py
│   │
│   ├── deepar/                     # DeepAR model
│   │   └── model_training.py     # With external features
│   │
│   └── ensemble/                   # Ensemble combiner
│       └── model_combiner.py      # Weighted average + optimization
│
├── dags/                           # Airflow DAGs
│   ├── dag_prophet.py
│   ├── dag_sarima.py
│   ├── dag_deepar.py
│   └── dag_ensemble.py            # Combines all models
│
├── models/                         # Trained models
│   ├── prophet/
│   ├── sarima/                    # Per-product structure
│   │   ├── PROD_001/
│   │   ├── PROD_002/
│   │   └── PROD_003/
│   ├── deepar/
│   └── ensemble/
│
├── data/                           # Data storage
│   ├── raw/
│   ├── processed/
│   └── predictions/
│       ├── prophet/
│       ├── sarima/
│       ├── deepar/
│       └── ensemble/
│
├── tests/                          # Unit tests
│   ├── test_common/
│   ├── test_prophet/
│   ├── test_sarima/
│   ├── test_deepar/
│   └── test_ensemble/
│
└── requirements/                   # Dependencies
    ├── base.txt                   # Shared
    ├── prophet.txt
    ├── sarima.txt
    ├── deepar.txt
    └── ensemble.txt
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Install base requirements
pip install -r requirements/base.txt

# Install model-specific requirements
pip install -r requirements/prophet.txt
pip install -r requirements/sarima.txt
pip install -r requirements/deepar.txt  # Optional: requires gluonts + mxnet
pip install -r requirements/ensemble.txt
```

### 2. Configuration

All models are configured via YAML files in `config/`:

```yaml
# config/base_config.yaml
pipeline:
  name: "multi_model_demand_forecasting"
  version: "2.0.0"

data_quality:
  max_null_rate: 0.05
  min_records: 90

evaluation:
  metrics: ["mape", "rmse", "mae", "smape"]
```

### 3. Train Models

#### Prophet
```bash
python scripts/prophet/model_training.py --data data/processed/cleaned_data.csv
```

#### SARIMA (Per-Product)
```bash
python scripts/sarima/model_training.py --data data/processed/cleaned_data.csv
```

#### DeepAR (With External Features)
```bash
python scripts/deepar/model_training.py --data data/processed/cleaned_data.csv
```

### 4. Generate Ensemble Forecast

```bash
python scripts/ensemble/model_combiner.py \
  --prophet data/predictions/prophet/forecast_20241226.csv \
  --sarima data/predictions/sarima/forecast_20241226.csv \
  --deepar data/predictions/deepar/forecast_20241226.csv
```

---

## 🎯 Ensemble Strategies

### Strategy 1: Weighted Average (Recommended)

**Auto-Optimization** - Weights optimized based on validation performance:

```python
# Inverse error weighting
w_prophet = 1 / mape_prophet
w_sarima = 1 / mape_sarima
w_deepar = 1 / mape_deepar

# Normalize to sum to 1
ensemble_forecast = (
    w_prophet * prophet_forecast +
    w_sarima * sarima_forecast +
    w_deepar * deepar_forecast
)
```

**Configuration:**
```yaml
# config/ensemble_config.yaml
ensemble:
  strategy: "weighted_average"
  weighted_average:
    weight_strategy: "auto"  # or "manual", "equal"
    optimization:
      metric: "mape"
      method: "inverse_error"
```

### Strategy 2: Best Model Selection (Fallback)

Select best performing model based on evaluation window:

```yaml
ensemble:
  best_model_selection:
    enabled: true
    evaluation_window: 0.2  # First 20% of predictions
    selection_metric: "mape"
```

---

## 📊 Model Comparison

| Model | MAPE (Expected) | Strengths | Use Case |
|-------|----------------|-----------|----------|
| **Prophet** | 12-18% | Fast, robust to missing data | Baseline, quick forecasts |
| **SARIMA** | 10-15% | Good for stationary patterns | Per-product forecasting |
| **DeepAR** | 8-12% | Handles complex patterns, external features | Advanced forecasting |
| **Ensemble** | **7-10%** | **Best of all models** | **Production forecasts** |

**Expected Improvement:** 20-30% over best single model

---

## 🔧 Configuration Guide

### Prophet Hyperparameters

```yaml
# config/prophet_config.yaml
model:
  prophet:
    changepoint_prior_scale: 0.05  # Trend flexibility
    seasonality_prior_scale: 10.0  # Seasonality flexibility
    seasonality_mode: "multiplicative"
    weekly_seasonality: true
    yearly_seasonality: true
```

### SARIMA Per-Product

```yaml
# config/sarima_config.yaml
model:
  per_product: true
  sarima:
    auto_arima:
      enabled: true
      seasonal: true
      m: 7  # Weekly seasonality
      max_p: 5
      max_q: 5
```

### DeepAR External Features

```yaml
# config/deepar_config.yaml
model:
  deepar:
    external_features:
      enabled: true
      features:
        - name: "revenue"
          type: "dynamic_real"
      preprocessing:
        lag_external: true
        rolling_external: true
```

---

## 🧪 Testing

Run all tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_common/ -v
pytest tests/test_ensemble/ -v

# Run with coverage
pytest tests/ --cov=scripts --cov-report=html
```

---

## 📈 Airflow DAG Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Preparation                         │
│                  (Shared across all models)                  │
└────────────┬────────────┬────────────┬──────────────────────┘
             │            │            │
             ▼            ▼            ▼
    ┌────────────┐ ┌────────────┐ ┌────────────┐
    │  Prophet   │ │   SARIMA   │ │   DeepAR   │
    │    DAG     │ │    DAG     │ │    DAG     │
    │            │ │            │ │            │
    │ Validate   │ │ Validate   │ │ Validate   │
    │ Train      │ │ Train      │ │ Train      │
    │ Predict    │ │ Predict    │ │ Predict    │
    └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
          │              │              │
          └──────────────┴──────────────┘
                         │
                         ▼
                  ┌────────────┐
                  │  Ensemble  │
                  │    DAG     │
                  │            │
                  │ Wait All   │
                  │ Combine    │
                  │ Predict    │
                  └────────────┘
```

---

## 🔑 Key Design Decisions

### 1. Ensemble Strategy
**Decision:** Weighted average with auto-optimization (NOT best 20% selection)

**Rationale:**
- More stable than single model selection
- Leverages diversity of all models
- Industry standard approach
- Better generalization

### 2. Model Versioning
**Decision:** Use BOTH versioning AND seed

```python
version = "prophet_v20241226_a1b2c3d4_seed42"
```

**Rationale:**
- **Versioning**: Production tracking, rollback capability
- **Seed**: Reproducibility, debugging
- Both needed for complete MLOps

### 3. SARIMA Per-Product
**Decision:** Train separate SARIMA model for each product

**Rationale:**
- Each product has unique patterns
- Better accuracy than single model
- Easy to add/remove products
- Parallel training support

### 4. DeepAR External Features
**Decision:** Use revenue as external feature

**Rationale:**
- Revenue correlates with sales quantity
- DeepAR designed for external features
- Improves forecast accuracy
- Demonstrates advanced capability

---

## 📝 Model Versioning

Every model is versioned with:

```json
{
  "model_version": "prophet_v20241226_a1b2c3d4_seed42",
  "model_type": "prophet",
  "seed": 42,
  "data_hash": "a1b2c3d4e5f6...",
  "hyperparameters": {...},
  "validation_metrics": {
    "mape": 2.5,
    "rmse": 10.2,
    "mae": 8.1,
    "smape": 2.3
  },
  "training_time_seconds": 45.3,
  "created_at": "2024-12-26T10:30:00"
}
```

---

## 🎓 Next Steps

### Phase 1: Validate Setup ✅
- [x] Folder structure created
- [x] Config files created
- [x] Common utilities implemented

### Phase 2: Train Individual Models
- [ ] Train Prophet model
- [ ] Train SARIMA models (per-product)
- [ ] Train DeepAR model (optional - requires gluonts)

### Phase 3: Create Ensemble
- [ ] Generate predictions from all models
- [ ] Optimize ensemble weights
- [ ] Compare ensemble vs individual models

### Phase 4: Deploy to Airflow
- [ ] Test DAGs locally
- [ ] Deploy to Airflow
- [ ] Schedule daily runs

### Phase 5: Monitor & Optimize
- [ ] Track model performance
- [ ] Retrain models periodically
- [ ] Optimize hyperparameters

---

## 🐛 Troubleshooting

### Issue: DeepAR not working
**Solution:** DeepAR requires gluonts and mxnet. Install with:
```bash
pip install gluonts mxnet
```

### Issue: SARIMA training slow
**Solution:** Enable parallel training in config:
```yaml
training:
  parallel_training:
    enabled: true
    n_jobs: -1
```

### Issue: Ensemble weights not optimal
**Solution:** Increase validation data size or use cross-validation:
```yaml
weighted_average:
  optimization:
    validation:
      method: "time_series_split"
      n_splits: 5
```

---

## 📚 Resources

- **Prophet**: [Facebook Prophet Documentation](https://facebook.github.io/prophet/)
- **SARIMA**: [Statsmodels SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
- **DeepAR**: [GluonTS DeepAR](https://ts.gluon.ai/stable/api/gluonts/gluonts.model.deepar.html)
- **Ensemble Methods**: [Kaggle Time Series Ensembles](https://www.kaggle.com/competitions)

---

## 👥 Contributors

- Data Engineering Team
- Version: 2.0.0
- Last Updated: 2024-12-26

---

## 📄 License

Internal use only - Data Engineering Team

---

**🚀 You're ready to build a production-grade forecasting system!**

For questions or issues, contact the Data Engineering Team.
