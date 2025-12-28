"""
SARIMA forecasting DAG.

Runs SARIMA model training and prediction.
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import sys
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

from sarima.model_training import train_sarima_models
# Assuming prediction module will be similar to Prophet's if not yet implemented
# from sarima.prediction import generate_sarima_forecast
from common.config_loader import load_config
from common.data_validator import validate_data
from common.utils import load_data

# Load configuration
config = load_config('sarima')
airflow_config = config.get('airflow', {})

# Default args
default_args = {
    'owner': airflow_config.get('default_args', {}).get('owner', 'data-team'),
    'depends_on_past': False,
    'email_on_failure': airflow_config.get('default_args', {}).get('email_on_failure', False),
    'email_on_retry': False,
    'retries': airflow_config.get('default_args', {}).get('retries', 2),
    'retry_delay': timedelta(seconds=airflow_config.get('default_args', {}).get('retry_delay_seconds', 300)),
}

# Create DAG
dag = DAG(
    'sarima_forecasting',
    default_args=default_args,
    description='SARIMA model training and forecasting',
    schedule_interval=airflow_config.get('schedule_interval', '@daily'),
    start_date=days_ago(1),
    catchup=airflow_config.get('catchup', False),
    max_active_runs=airflow_config.get('max_active_runs', 1),
    tags=['forecasting', 'sarima', 'multi-model'],
)


def validate_data_task(**context):
    """Validate input data quality."""
    data_path = Path(config['paths']['processed_data']) / 'cleaned_data.csv'
    df = load_data(data_path)
    
    is_valid, report = validate_data(df, config)
    
    if not is_valid:
        raise ValueError(f"Data validation failed: {report['errors']}")
    
    context['ti'].xcom_push(key='validation_report', value=report)
    return "Data validation passed"


def train_model_task(**context):
    """Train SARIMA models."""
    data_path = Path(config['paths']['processed_data']) / 'cleaned_data.csv'
    
    results = train_sarima_models(
        data_path=data_path,
        config=config,
        save_model_flag=True
    )
    
    # context['ti'].xcom_push(key='model_version', value=results['model_version'])
    # context['ti'].xcom_push(key='validation_mape', value=results['validation_metrics']['mape'])
    
    return f"Models trained"


# Define tasks
validate_data_op = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data_task,
    provide_context=True,
    dag=dag,
)

train_model_op = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    provide_context=True,
    dag=dag,
)

# Define dependencies
validate_data_op >> train_model_op
