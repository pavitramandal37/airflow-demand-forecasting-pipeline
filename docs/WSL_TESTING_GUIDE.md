# WSL Testing Guide - Airflow Demand Forecasting Pipeline

This guide provides step-by-step instructions for testing the Airflow Demand Forecasting project in Windows Subsystem for Linux (WSL).

## Prerequisites

Before starting, ensure you have:
- WSL 2 installed on Windows
- Ubuntu (or your preferred Linux distribution) installed in WSL
- Python 3.10+ available in WSL
- Git installed in WSL

## Step 1: Access WSL

Open PowerShell or Windows Terminal and launch WSL:

```bash
wsl
```

## Step 2: Navigate to Project Directory

The Windows D: drive is accessible in WSL at `/mnt/d/`. Navigate to your project:

```bash
cd "/mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting"
```

## Step 3: Verify Project Structure

Check that all files are accessible:

```bash
ls -la
```

You should see:
- `dags/` - Airflow DAG definitions
- `scripts/` - Python modules
- `config/` - Configuration files
- `setup.sh` - Setup script
- `requirements.txt` - Python dependencies

## Step 4: Install System Dependencies (if needed)

Install required system packages:

```bash
# Update package list
sudo apt update

# Install Python 3.11 and pip
sudo apt install -y python3.11 python3.11-venv python3-pip

# Install build essentials (needed for some Python packages)
sudo apt install -y build-essential python3.11-dev

# Install additional dependencies for Prophet
sudo apt install -y libssl-dev libffi-dev
```

## Step 5: Make Setup Script Executable

```bash
chmod +x setup.sh
```

## Step 6: Run Setup Script

Execute the setup script to initialize the environment:

```bash
./setup.sh
```

This will:
- Create necessary directories
- Set up Python virtual environment
- Install all dependencies
- Initialize Airflow database
- Create admin user (username: `admin`, password: `admin`)
- Generate sample data

**Note**: If you encounter any errors, see the Troubleshooting section below.

## Step 7: Activate Virtual Environment

```bash
source venv/bin/activate
```

Your prompt should now show `(venv)` prefix.

## Step 8: Set Airflow Home

```bash
export AIRFLOW_HOME=$(pwd)
```

## Step 9: Test Individual Scripts (Optional but Recommended)

Before running the full Airflow pipeline, test each script independently:

### Test 1: Generate Sample Data
```bash
python -m scripts.generate_sample_data
```
**Expected**: Creates CSV file in `data/raw/` with synthetic sales data.

### Test 2: Data Extraction
```bash
python -m scripts.data_extraction
```
**Expected**: Loads data and prints record count.

### Test 3: Data Transformation
```bash
python -m scripts.data_transformation
```
**Expected**: Validates and cleans data, saves to `data/processed/`.

### Test 4: Feature Engineering
```bash
python -m scripts.feature_engineering
```
**Expected**: Creates features (rolling averages, lags), saves engineered data.

### Test 5: Model Training
```bash
python -m scripts.model_training
```
**Expected**: Trains Prophet model, saves to `models/saved_models/`.

### Test 6: Prediction Generation
```bash
python -m scripts.prediction_generator
```
**Expected**: Generates forecasts, saves to `data/predictions/`.

## Step 10: Start Airflow Services

You'll need **two separate WSL terminals** for this.

### Terminal 1: Start Airflow Webserver

```bash
# Navigate to project
cd "/mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting"

# Activate environment
source venv/bin/activate

# Set Airflow home
export AIRFLOW_HOME=$(pwd)

# Start webserver
airflow webserver --port 8080
```

Wait until you see: `Listening at: http://0.0.0.0:8080`

### Terminal 2: Start Airflow Scheduler

```bash
# Navigate to project
cd "/mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting"

# Activate environment
source venv/bin/activate

# Set Airflow home
export AIRFLOW_HOME=$(pwd)

# Start scheduler
airflow scheduler
```

## Step 11: Access Airflow Web UI

1. Open your Windows browser
2. Navigate to: `http://localhost:8080`
3. Login with:
   - **Username**: `admin`
   - **Password**: `admin`

## Step 12: Run the Pipeline

1. In the Airflow UI, find the DAG: `demand_forecasting_pipeline`
2. Toggle the DAG to **ON** (unpause it)
3. Click the **Play** button to trigger a manual run
4. Monitor the task execution in the Graph or Grid view

## Step 13: Verify Results

After successful execution, check the outputs:

```bash
# Check processed data
ls -lh data/processed/

# Check trained models
ls -lh models/saved_models/

# Check predictions
ls -lh data/predictions/

# View a sample prediction
head -20 data/predictions/forecast_*.csv
```

## Step 14: View Logs

Check logs for any issues:

```bash
# View all logs
ls -R logs/

# View specific task log (example)
cat logs/dag_id=demand_forecasting_pipeline/run_id=*/task_id=*/attempt=*.log
```

## Running Tests

Execute the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=scripts --cov-report=html

# View coverage report
# The HTML report will be in htmlcov/index.html
```

## Stopping Airflow

To stop the services:

1. In each terminal, press `Ctrl+C`
2. Wait for graceful shutdown
3. Deactivate virtual environment: `deactivate`

## Troubleshooting

### Issue: "python3.11: command not found"

**Solution**: Install Python 3.11 or modify `setup.sh` to use available Python version:

```bash
# Check available Python versions
python3 --version

# Edit setup.sh line 63 to use your version
# Change: python3.11 -m venv venv
# To: python3 -m venv venv
```

### Issue: "Permission denied" when running setup.sh

**Solution**: Make the script executable:
```bash
chmod +x setup.sh
```

### Issue: Prophet installation fails

**Solution**: Install system dependencies:
```bash
sudo apt install -y build-essential python3-dev libssl-dev libffi-dev
pip install --upgrade pip setuptools wheel
pip install pystan==2.19.1.1
pip install prophet
```

### Issue: Port 8080 already in use

**Solution**: Use a different port:
```bash
airflow webserver --port 8081
```

### Issue: "No module named 'scripts'"

**Solution**: Ensure you're running from the project root and PYTHONPATH is set:
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Issue: Database locked errors

**Solution**: Stop all Airflow processes and reset:
```bash
# Kill all Airflow processes
pkill -f airflow

# Remove lock files
rm -f airflow.db-shm airflow.db-wal

# Reinitialize
airflow db reset
```

### Issue: WSL file permissions

**Solution**: If you encounter permission issues with Windows files:
```bash
# Copy project to WSL home directory for better performance
cp -r "/mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting" ~/
cd ~/airflow-demand-forecasting
```

## Performance Tips

1. **Use WSL 2**: Ensure you're using WSL 2 for better performance
   ```bash
   wsl --list --verbose
   ```

2. **Work in WSL filesystem**: For better I/O performance, consider copying the project to `~/projects/` instead of accessing via `/mnt/d/`

3. **Increase WSL memory**: Edit `.wslconfig` in Windows user directory:
   ```ini
   [wsl2]
   memory=4GB
   processors=2
   ```

## Quick Reference Commands

```bash
# Activate environment
source venv/bin/activate

# Set Airflow home
export AIRFLOW_HOME=$(pwd)

# List DAGs
airflow dags list

# Test specific task
airflow tasks test demand_forecasting_pipeline extract_data 2024-01-01

# Trigger DAG manually
airflow dags trigger demand_forecasting_pipeline

# View DAG structure
airflow dags show demand_forecasting_pipeline

# Clear task instances
airflow tasks clear demand_forecasting_pipeline

# Check Airflow version
airflow version
```

## Next Steps

After successful testing:

1. ✅ Review the generated forecasts in `data/predictions/`
2. ✅ Check model metadata in `models/saved_models/`
3. ✅ Experiment with configuration in `config/pipeline_config.yaml`
4. ✅ Add custom validation rules
5. ✅ Implement additional features
6. ✅ Set up production deployment

## Additional Resources

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [WSL Documentation](https://docs.microsoft.com/en-us/windows/wsl/)
- Project Architecture: See `docs/architecture.md`

---

**Happy Testing! 🚀**
