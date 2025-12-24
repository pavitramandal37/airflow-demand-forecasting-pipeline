#!/bin/bash
# Fix Prophet stan_backend error by reinstalling dependencies

echo "Fixing Prophet dependencies..."

# Navigate to project directory
cd /mnt/d/My\ Apps/Airflow\ Demand\ Forecast\ Project/airflow-demand-forecasting

# Activate virtual environment
source venv/bin/activate

# Uninstall prophet and cmdstanpy
echo "Uninstalling existing Prophet and cmdstanpy..."
pip uninstall -y prophet cmdstanpy

# Install cmdstanpy first
echo "Installing cmdstanpy 1.2.0..."
pip install cmdstanpy==1.2.0

# Install prophet
echo "Installing Prophet 1.1.5..."
pip install prophet==1.1.5

echo "Dependencies fixed! You can now retry the Airflow task."
