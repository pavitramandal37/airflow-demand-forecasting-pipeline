#!/bin/bash
# =============================================================================
# WSL Quick Start Script - Airflow Demand Forecasting Pipeline
# =============================================================================
# This script provides a streamlined way to test the pipeline in WSL
# =============================================================================

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Airflow Pipeline - WSL Quick Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Run setup if needed
if [ ! -d "venv" ]; then
    echo -e "${GREEN}[1/6]${NC} Running initial setup..."
    chmod +x setup.sh
    ./setup.sh
else
    echo -e "${YELLOW}[1/6]${NC} Virtual environment exists, skipping setup..."
fi

# Step 2: Activate environment
echo -e "${GREEN}[2/6]${NC} Activating virtual environment..."
source venv/bin/activate

# Step 3: Set Airflow home
echo -e "${GREEN}[3/6]${NC} Setting Airflow home..."
export AIRFLOW_HOME=$(pwd)

# Step 4: Test individual scripts
echo -e "${GREEN}[4/6]${NC} Testing individual pipeline components..."
echo ""

echo "  → Testing data generation..."
python -m scripts.generate_sample_data
echo "    ✓ Sample data generated"
echo ""

echo "  → Testing data extraction..."
python -m scripts.data_extraction
echo "    ✓ Data extraction successful"
echo ""

echo "  → Testing data transformation..."
python -m scripts.data_transformation
echo "    ✓ Data transformation successful"
echo ""

echo "  → Testing feature engineering..."
python -m scripts.feature_engineering
echo "    ✓ Feature engineering successful"
echo ""

echo "  → Testing model training..."
python -m scripts.model_training
echo "    ✓ Model training successful"
echo ""

echo "  → Testing prediction generation..."
python -m scripts.prediction_generator
echo "    ✓ Predictions generated"
echo ""

# Step 5: Show results
echo -e "${GREEN}[5/6]${NC} Checking generated outputs..."
echo ""
echo "  📁 Processed Data:"
ls -lh data/processed/ | tail -n +2
echo ""
echo "  🤖 Trained Models:"
ls -lh models/saved_models/ | tail -n +2
echo ""
echo "  📊 Predictions:"
ls -lh data/predictions/ | tail -n +2
echo ""

# Step 6: Display next steps
echo -e "${GREEN}[6/6]${NC} All tests completed successfully! ✨"
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Next Steps${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "To start Airflow and run the full DAG:"
echo ""
echo "  Terminal 1 (Webserver):"
echo "    source venv/bin/activate"
echo "    export AIRFLOW_HOME=\$(pwd)"
echo "    airflow webserver --port 8080"
echo ""
echo "  Terminal 2 (Scheduler):"
echo "    source venv/bin/activate"
echo "    export AIRFLOW_HOME=\$(pwd)"
echo "    airflow scheduler"
echo ""
echo "  Then visit: http://localhost:8080"
echo "  Login: admin / admin"
echo ""
echo -e "${BLUE}========================================${NC}"
