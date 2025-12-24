# WSL Testing - Quick Command Reference

## 🚀 Quick Start (Automated)

```bash
# Open WSL
wsl

# Navigate to project
cd "/mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting"

# Run automated test
chmod +x wsl_quick_test.sh
./wsl_quick_test.sh
```

## 📋 Manual Testing Commands

### Initial Setup
```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh

# Activate environment
source venv/bin/activate

# Set Airflow home
export AIRFLOW_HOME=$(pwd)
```

### Test Individual Components
```bash
# Generate sample data
python -m scripts.generate_sample_data

# Extract data
python -m scripts.data_extraction

# Transform data
python -m scripts.data_transformation

# Engineer features
python -m scripts.feature_engineering

# Train model
python -m scripts.model_training

# Generate predictions
python -m scripts.prediction_generator
```

### Start Airflow Services

**Terminal 1 - Webserver:**
```bash
cd "/mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting"
source venv/bin/activate
export AIRFLOW_HOME=$(pwd)
airflow webserver --port 8080
```

**Terminal 2 - Scheduler:**
```bash
cd "/mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting"
source venv/bin/activate
export AIRFLOW_HOME=$(pwd)
airflow scheduler
```

### Airflow CLI Commands
```bash
# List all DAGs
airflow dags list

# Show DAG structure
airflow dags show demand_forecasting_pipeline

# Trigger DAG manually
airflow dags trigger demand_forecasting_pipeline

# Test specific task
airflow tasks test demand_forecasting_pipeline extract_data 2024-01-01

# View task logs
airflow tasks logs demand_forecasting_pipeline extract_data 2024-01-01
```

### Check Results
```bash
# View processed data
ls -lh data/processed/
cat data/processed/cleaned_data.csv | head -20

# View trained models
ls -lh models/saved_models/

# View predictions
ls -lh data/predictions/
cat data/predictions/forecast_*.csv | head -20

# View logs
ls -R logs/
```

### Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=scripts --cov-report=html

# Run specific test
pytest tests/test_data_validation.py -v
```

## 🛠️ Troubleshooting Commands

### Check Python Version
```bash
python3 --version
which python3
```

### Check Virtual Environment
```bash
ls -la venv/
source venv/bin/activate
which python
pip list
```

### Reset Airflow Database
```bash
# Stop all Airflow processes
pkill -f airflow

# Remove database
rm -f airflow.db airflow.db-shm airflow.db-wal

# Reinitialize
airflow db init

# Recreate admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### Check Disk Space
```bash
df -h
du -sh data/ models/ logs/
```

### View Configuration
```bash
# View pipeline config
cat config/pipeline_config.yaml

# View Airflow config
cat airflow.cfg | grep -A 5 "dags_folder"
```

## 🌐 Access Points

- **Airflow UI**: http://localhost:8080
- **Username**: admin
- **Password**: admin

## 📊 Expected File Structure After Testing

```
data/
├── raw/
│   └── sales_data_YYYYMMDD.csv
├── processed/
│   └── cleaned_data.csv
└── predictions/
    └── forecast_YYYYMMDD_HASH.csv

models/
└── saved_models/
    ├── prophet_model_vYYYYMMDD_HASH.pkl
    └── prophet_model_vYYYYMMDD_HASH_metadata.json

logs/
└── dag_id=demand_forecasting_pipeline/
    └── [task logs]
```

## 🔄 Common Workflows

### Full Pipeline Test
```bash
./wsl_quick_test.sh
```

### Manual Step-by-Step Test
```bash
source venv/bin/activate
export AIRFLOW_HOME=$(pwd)
python -m scripts.generate_sample_data
python -m scripts.data_extraction
python -m scripts.data_transformation
python -m scripts.feature_engineering
python -m scripts.model_training
python -m scripts.prediction_generator
```

### Clean and Restart
```bash
# Clean generated files
rm -rf data/raw/* data/processed/* data/predictions/* models/saved_models/*

# Keep .gitkeep files
touch data/raw/.gitkeep data/processed/.gitkeep data/predictions/.gitkeep models/saved_models/.gitkeep

# Regenerate sample data
python -m scripts.generate_sample_data
```

## 💡 Pro Tips

1. **Use tab completion**: Type partial command and press Tab
2. **Use history**: Press ↑ to cycle through previous commands
3. **Use aliases**: Add to `~/.bashrc`:
   ```bash
   alias af-activate='source venv/bin/activate && export AIRFLOW_HOME=$(pwd)'
   alias af-web='airflow webserver --port 8080'
   alias af-sched='airflow scheduler'
   ```
4. **Monitor logs in real-time**:
   ```bash
   tail -f logs/dag_id=*/run_id=*/task_id=*/attempt=*.log
   ```

## 🆘 Getting Help

```bash
# Airflow help
airflow --help
airflow dags --help
airflow tasks --help

# Python module help
python -m scripts.data_extraction --help

# Check script documentation
cat scripts/data_extraction.py | head -50
```
