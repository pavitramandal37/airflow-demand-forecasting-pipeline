# WSL Testing Checklist

Use this checklist to ensure your WSL environment is properly configured for testing the Airflow Demand Forecasting Pipeline.

## ✅ Pre-Testing Checklist

### System Requirements
- [ ] WSL 2 is installed and running
  ```bash
  wsl --list --verbose
  # Should show VERSION 2
  ```

- [ ] Ubuntu or compatible Linux distribution is installed
  ```bash
  wsl cat /etc/os-release
  ```

- [ ] Python 3.10+ is available
  ```bash
  wsl python3 --version
  # Should show Python 3.10.x or higher
  ```

- [ ] pip is installed
  ```bash
  wsl python3 -m pip --version
  ```

- [ ] Git is installed (optional, for version control)
  ```bash
  wsl git --version
  ```

### Project Access
- [ ] Can access project directory from WSL
  ```bash
  wsl ls "/mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting"
  ```

- [ ] Setup script exists and is readable
  ```bash
  wsl cat "/mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting/setup.sh" | head -5
  ```

- [ ] Requirements file exists
  ```bash
  wsl cat "/mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting/requirements.txt"
  ```

## ✅ Setup Checklist

### Environment Setup
- [ ] Setup script is executable
  ```bash
  chmod +x setup.sh
  ```

- [ ] Virtual environment created successfully
  ```bash
  ls -la venv/
  ```

- [ ] Virtual environment can be activated
  ```bash
  source venv/bin/activate
  which python
  # Should point to venv/bin/python
  ```

- [ ] All dependencies installed without errors
  ```bash
  pip list | grep -E "airflow|prophet|pandas|numpy"
  ```

- [ ] Airflow database initialized
  ```bash
  ls -lh airflow.db
  # Should exist and be > 0 bytes
  ```

- [ ] Admin user created
  ```bash
  airflow users list | grep admin
  ```

### Directory Structure
- [ ] Data directories created
  ```bash
  ls -la data/raw data/processed data/predictions
  ```

- [ ] Model directory created
  ```bash
  ls -la models/saved_models
  ```

- [ ] Logs directory created
  ```bash
  ls -la logs
  ```

## ✅ Component Testing Checklist

### Data Generation
- [ ] Sample data script runs without errors
  ```bash
  python -m scripts.generate_sample_data
  ```

- [ ] Sample data file created
  ```bash
  ls -lh data/raw/sales_data_*.csv
  ```

- [ ] Sample data has expected structure
  ```bash
  head -5 data/raw/sales_data_*.csv
  # Should show: date,sales columns
  ```

### Data Extraction
- [ ] Extraction script runs without errors
  ```bash
  python -m scripts.data_extraction
  ```

- [ ] Script outputs record count
  ```bash
  python -m scripts.data_extraction 2>&1 | grep -i "records"
  ```

### Data Transformation
- [ ] Transformation script runs without errors
  ```bash
  python -m scripts.data_transformation
  ```

- [ ] Cleaned data file created
  ```bash
  ls -lh data/processed/cleaned_data.csv
  ```

- [ ] Data passes validation checks
  ```bash
  python -m scripts.data_transformation 2>&1 | grep -i "validation"
  ```

### Feature Engineering
- [ ] Feature engineering script runs without errors
  ```bash
  python -m scripts.feature_engineering
  ```

- [ ] Engineered features file created
  ```bash
  ls -lh data/processed/*features*.csv
  ```

### Model Training
- [ ] Training script runs without errors
  ```bash
  python -m scripts.model_training
  ```

- [ ] Model file created
  ```bash
  ls -lh models/saved_models/*.pkl
  ```

- [ ] Model metadata file created
  ```bash
  ls -lh models/saved_models/*_metadata.json
  ```

- [ ] Metadata contains expected fields
  ```bash
  cat models/saved_models/*_metadata.json | grep -E "training_date|data_hash|horizon_days"
  ```

### Prediction Generation
- [ ] Prediction script runs without errors
  ```bash
  python -m scripts.prediction_generator
  ```

- [ ] Forecast file created
  ```bash
  ls -lh data/predictions/forecast_*.csv
  ```

- [ ] Forecast has expected columns
  ```bash
  head -2 data/predictions/forecast_*.csv
  # Should show: ds, yhat, yhat_lower, yhat_upper
  ```

## ✅ Airflow Services Checklist

### Webserver
- [ ] Webserver starts without errors
  ```bash
  # In separate terminal
  airflow webserver --port 8080
  # Should show "Listening at: http://0.0.0.0:8080"
  ```

- [ ] Can access UI from Windows browser
  - Open: http://localhost:8080
  - Should see Airflow login page

- [ ] Can login with admin credentials
  - Username: admin
  - Password: admin
  - Should see DAGs list

### Scheduler
- [ ] Scheduler starts without errors
  ```bash
  # In separate terminal
  airflow scheduler
  # Should show "Starting the scheduler"
  ```

- [ ] Scheduler detects DAG
  ```bash
  airflow dags list | grep demand_forecasting_pipeline
  ```

### DAG Execution
- [ ] DAG appears in UI
  - Should see "demand_forecasting_pipeline" in DAGs list

- [ ] DAG can be unpaused
  - Toggle switch should turn blue/green

- [ ] DAG can be triggered manually
  - Click play button
  - Should create new DAG run

- [ ] All tasks complete successfully
  - All task boxes should be green
  - No red (failed) or orange (retrying) boxes

- [ ] Task logs are accessible
  - Click on task box
  - Click "Log" button
  - Should see detailed execution logs

## ✅ Output Verification Checklist

### Data Files
- [ ] Raw data exists and is valid CSV
  ```bash
  file data/raw/sales_data_*.csv
  head -10 data/raw/sales_data_*.csv
  ```

- [ ] Processed data exists and is valid CSV
  ```bash
  file data/processed/cleaned_data.csv
  head -10 data/processed/cleaned_data.csv
  ```

- [ ] Predictions exist and are valid CSV
  ```bash
  file data/predictions/forecast_*.csv
  head -10 data/predictions/forecast_*.csv
  ```

### Model Files
- [ ] Model pickle file exists
  ```bash
  file models/saved_models/*.pkl
  ```

- [ ] Model metadata is valid JSON
  ```bash
  cat models/saved_models/*_metadata.json | python3 -m json.tool
  ```

### Logs
- [ ] Application logs exist
  ```bash
  ls -R logs/
  ```

- [ ] No critical errors in logs
  ```bash
  grep -r "ERROR" logs/ | grep -v "No such file"
  ```

## ✅ Testing Checklist

### Unit Tests
- [ ] pytest is installed
  ```bash
  pytest --version
  ```

- [ ] Tests can be discovered
  ```bash
  pytest --collect-only tests/
  ```

- [ ] All tests pass
  ```bash
  pytest tests/ -v
  ```

- [ ] Coverage report generated
  ```bash
  pytest tests/ --cov=scripts --cov-report=term
  ```

## ✅ Performance Checklist

### Resource Usage
- [ ] Disk space is sufficient
  ```bash
  df -h /mnt/d
  # Should have at least 1GB free
  ```

- [ ] Memory usage is reasonable
  ```bash
  free -h
  ```

- [ ] No zombie processes
  ```bash
  ps aux | grep -E "airflow|python" | grep -v grep
  ```

### Execution Time
- [ ] Data generation completes in < 30 seconds
- [ ] Data transformation completes in < 1 minute
- [ ] Model training completes in < 5 minutes
- [ ] Full pipeline completes in < 10 minutes

## ✅ Cleanup Checklist

### Stop Services
- [ ] Webserver stopped gracefully
  ```bash
  # Ctrl+C in webserver terminal
  # Should show "Shutting down"
  ```

- [ ] Scheduler stopped gracefully
  ```bash
  # Ctrl+C in scheduler terminal
  # Should show "Shutting down"
  ```

- [ ] No orphaned processes
  ```bash
  ps aux | grep airflow | grep -v grep
  # Should return nothing
  ```

### Optional: Clean Generated Files
- [ ] Remove generated data (if needed)
  ```bash
  rm -rf data/raw/* data/processed/* data/predictions/*
  ```

- [ ] Remove trained models (if needed)
  ```bash
  rm -rf models/saved_models/*
  ```

- [ ] Keep .gitkeep files
  ```bash
  touch data/raw/.gitkeep data/processed/.gitkeep data/predictions/.gitkeep models/saved_models/.gitkeep
  ```

## 🎯 Success Criteria

Your testing is successful if:

✅ All items in "Component Testing Checklist" are checked  
✅ All items in "Airflow Services Checklist" are checked  
✅ All items in "Output Verification Checklist" are checked  
✅ DAG runs successfully from start to finish  
✅ Predictions are generated and look reasonable  
✅ No errors in logs  
✅ Tests pass with good coverage  

## 🚨 Common Issues and Solutions

| Issue | Check | Solution |
|-------|-------|----------|
| Setup script fails | Python version | Use `python3` instead of `python3.11` |
| Import errors | Virtual environment | Ensure `source venv/bin/activate` |
| Airflow not found | AIRFLOW_HOME | Run `export AIRFLOW_HOME=$(pwd)` |
| Port 8080 in use | Port conflict | Use `--port 8081` |
| Database locked | Stale processes | Run `pkill -f airflow` |
| Prophet install fails | System deps | Install build-essential |
| Permission denied | File permissions | Run `chmod +x setup.sh` |
| Out of memory | WSL config | Increase WSL memory limit |

## 📝 Notes

- Save this checklist and mark items as you complete them
- If any item fails, refer to WSL_TESTING_GUIDE.md for detailed troubleshooting
- Document any issues you encounter for future reference
- Consider creating a test report with your findings

---

**Testing Date**: _____________  
**Tester**: _____________  
**WSL Version**: _____________  
**Python Version**: _____________  
**Overall Result**: ⬜ Pass ⬜ Fail ⬜ Partial
