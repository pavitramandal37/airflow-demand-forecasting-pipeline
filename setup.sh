#!/bin/bash
# =============================================================================
# Airflow Demand Forecasting Pipeline - Setup Script
# =============================================================================
# This script initializes the development environment for the forecasting
# pipeline. It creates necessary directories, sets up Python dependencies,
# and configures Airflow for local development.
# =============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# -----------------------------------------------------------------------------
# Directory Setup
# -----------------------------------------------------------------------------

log_info "Creating project directories..."

# Create data directories with .gitkeep files to preserve structure
mkdir -p data/raw data/processed data/predictions
touch data/raw/.gitkeep data/processed/.gitkeep data/predictions/.gitkeep

# Create model storage directory
mkdir -p models/saved_models
touch models/saved_models/.gitkeep

# Create logs directory
mkdir -p logs
touch logs/.gitkeep

log_info "Directory structure created successfully."

# -----------------------------------------------------------------------------
# Python Environment Setup
# -----------------------------------------------------------------------------

log_info "Setting up Python environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv venv
else
    log_warn "Virtual environment already exists, skipping creation."
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
log_info "Installing Python dependencies..."
pip install -r requirements.txt

log_info "Python dependencies installed successfully."

# -----------------------------------------------------------------------------
# Airflow Configuration
# -----------------------------------------------------------------------------

log_info "Configuring Airflow..."

# Set Airflow home to current directory
export AIRFLOW_HOME=$(pwd)

# Initialize Airflow database (SQLite for local development)
log_info "Initializing Airflow database..."
airflow db init

# Create default admin user for web UI access
log_info "Creating Airflow admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin 2>/dev/null || log_warn "Admin user may already exist."

log_info "Airflow configured successfully."

# -----------------------------------------------------------------------------
# Generate Sample Data
# -----------------------------------------------------------------------------

log_info "Generating sample data for testing..."
python scripts/generate_sample_data.py

log_info "Sample data generated successfully."

# -----------------------------------------------------------------------------
# Completion
# -----------------------------------------------------------------------------

echo ""
echo "============================================================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "============================================================================="
echo ""
echo "To start the Airflow services:"
echo ""
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Set AIRFLOW_HOME:"
echo "     export AIRFLOW_HOME=\$(pwd)"
echo ""
echo "  3. Start the webserver (in one terminal):"
echo "     airflow webserver --port 8080"
echo ""
echo "  4. Start the scheduler (in another terminal):"
echo "     airflow scheduler"
echo ""
echo "  5. Access the UI at http://localhost:8080"
echo "     Username: admin"
echo "     Password: admin"
echo ""
echo "============================================================================="
