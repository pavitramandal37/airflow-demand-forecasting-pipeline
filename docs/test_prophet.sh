#!/bin/bash
# Test Prophet initialization

cd /mnt/d/My\ Apps/Airflow\ Demand\ Forecast\ Project/airflow-demand-forecasting
source venv/bin/activate
python -c "from prophet import Prophet; m = Prophet(); print('Prophet initialized successfully!')"
