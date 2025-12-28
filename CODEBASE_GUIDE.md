# 📘 Multi-Model Demand Forecasting - Codebase Guide

This document provides a comprehensive overview of the `airflow-demand-forecasting` project codebase. It explains the project structure, the purpose of each module, validation flows, and how to extend the system.

---

## 🏗️ Project Architecture

The project follows a **modular, configuration-driven architecture** designed for scalability and maintainability.

### **Directory Structure**

```
airflow-demand-forecasting/
│
├── config/                  # ⚙️ Configuration (YAML)
│   ├── base_config.yaml     # Global settings (paths, logging, validation)
│   ├── prophet_config.yaml  # Prophet implementation details
│   ├── sarima_config.yaml   # SARIMA specific settings
│   ├── deepar_config.yaml   # DeepAR architecture & training params
│   └── ensemble_config.yaml # Ensemble strategy settings
│
├── scripts/                 # 🐍 Python Implementation
│   ├── common/              # Shared utilities (Crucial!)
│   │   ├── config_loader.py # Loads & merges YAML configs
│   │   ├── metrics.py       # Performance calculations (MAPE, RMSE, etc.)
│   │   ├── model_versioning.py # Handles saving models with hash+seed
│   │   ├── data_validator.py # Checks data quality before training
│   │   └── utils.py         # General helpers (logging, dates)
│   │
│   ├── prophet/             # Facebook Prophet Module
│   │   ├── model_training.py
│   │   └── prediction.py
│   │
│   ├── sarima/              # SARIMA Module
│   │   └── model_training.py # (Prediction logic handled here or implicitly)
│   │
│   ├── deepar/              # AWS/GluonTS DeepAR Module
│   │   ├── model_training.py
│   │   ├── prediction.py
│   │   ├── external_features.py # Preprocessing for extra regressors
│   │   └── data_formatting.py   # Converters for GluonTS format
│   │
│   └── ensemble/            # Ensemble Module
│       └── model_combiner.py # Logic to combine forecasts
│
├── dags/                    # 🌪️ Airflow DAGs (Orchestration)
│   ├── dag_prophet.py       # DAG for Prophet pipeline
│   ├── dag_sarima.py        # DAG for SARIMA pipeline
│   ├── dag_deepar.py        # DAG for DeepAR pipeline
│   └── dag_ensemble.py      # DAG that waits for others & combines results
│
├── tests/                   # 🧪 Unit Tests
│   ├── test_common/
│   ├── test_prophet/
│   ├── test_ensemble/
│   └── ...
│
├── models/                  # 💾 Model Artifacts (Saved Models)
│   ├── prophet/
│   ├── sarima/ (organized by Product ID)
│   ├── deepar/
│   └── ensemble/
│
├── data/                    # 📊 Data Store
│   ├── raw/                 # Input CSVs
│   ├── processed/           # Cleaned/Transformed data
│   └── predictions/         # Output forecasts
│
└── archive/                 # 📦 Archived legacy files
```

---

## 🔑 Key Modules Explained

### **1. Configuration (`config/`)**
The system is controlled via YAML files. **Do not hardcode parameters in Python scripts.**
*   **`base_config.yaml`**: The source of truth for file paths and validation rules.
*   **Model Configs**: Each model has its own config file defining hyperparameters (e.g., `changepoint_prior_scale` for Prophet, `context_length` for DeepAR).

### **2. Common Utilities (`scripts/common/`)**
This is the backbone of the project.
*   **`config_loader.py`**: Reads YAMLs and enables hierarchical overrides.
*   **`model_versioning.py`**: Ensures reproducibility. Every saved model includes:
    *   **Timestamp**: When it was trained.
    *   **Data Hash**: Unique signature of the training data.
    *   **Seed**: Random seed used for initialization.
*   **`data_validator.py`**: Runs sanity checks (null values, schema validation) before any training job starts.

### **3. Model Implementations (`scripts/<model>/`)**
Each model is isolated in its own package.
*   **Prophet**: Standard implementation wrapping Facebook's library.
*   **SARIMA**: Handles **Per-Product** modeling (looping through unique IDs), as SARIMA is univariate.
*   **DeepAR**: Advanced implementation using PyTorch/GluonTS. Supports external text/numerical features via `external_features.py`.

### **4. Ensemble (`scripts/ensemble/`)**
This module reads predictions from individual models and combines them.
*   **`model_combiner.py`**: Implements the weighted average logic. It can optimized weights based on historical performance (Inverse Variance Weighting or OLS).

### **5. Orchestration (`dags/`)**
Airflow DAGs manage the workflow.
*   **Decoupled Execution**: Prophet, SARIMA, and DeepAR run in parallel DAGs.
*   **Dependencies**: The `dag_ensemble.py` uses `ExternalTaskSensor` to wait for the completion of the individual model DAGs before running.

---

## 🛠️ How to Extend

### **Adding a New Model (e.g., XGBoost)**
1.  **Create Config**: Add `config/xgboost_config.yaml`.
2.  **Create Module**: Create `scripts/xgboost/` with `model_training.py` and `prediction.py`.
3.  **Implement Logic**: Use `common.model_versioning` to save artifacts.
4.  **Create DAG**: Add `dags/dag_xgboost.py`.
5.  **Update Ensemble**: Modify `scripts/ensemble/model_combiner.py` to include the new model's output in the weighting logic.
6.  **Add Dependencies**: Update `requirements/` (or create `requirements/xgboost.txt`).

---

## ✅ Best Practices

1.  **Always use `load_config()`**: Never manually modify paths in code.
2.  **Validate Data First**: Ensure your DAG calls `validate_data()` before training.
3.  **Version Everything**: Use the provided `create_model_version()` function.
4.  **Isolate Dependencies**: Keep `requirements/*.txt` clean to avoid conflicts between libraries (e.g., Statsmodels vs PyTorch).

---

## 🚀 Execution Flow

1.  **Installation**: Run `install_dependencies.sh` (WSL) or install from `requirements/`.
2.  **Data Prep**: Place raw data in `data/raw/` (or configure pipeline to fetch it).
3.  **Training**: Trigger `dag_prophet`, `dag_sarima`, `dag_deepar` in Airflow.
4.  **Ensembling**: Once training DAGs complete, `dag_ensemble` triggers automatically (or manually) to generate the final forecast.
