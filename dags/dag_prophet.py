"""
Prophet forecasting DAG.

Runs Prophet model training and prediction.
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import sys
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

from prophet.model_training import train_prophet_model
from prophet.prediction import generate_prophet_forecast
from common.config_loader import load_config
from common.data_validator import validate_data
from common.utils import load_data

# Load configuration
config = load_config('prophet')
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
    'prophet_forecasting',
    default_args=default_args,
    description='Prophet model training and forecasting',
    schedule_interval=airflow_config.get('schedule_interval', '@daily'),
    start_date=days_ago(1),
    catchup=airflow_config.get('catchup', False),
    max_active_runs=airflow_config.get('max_active_runs', 1),
    tags=['forecasting', 'prophet', 'multi-model'],
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
    """Train Prophet model."""
    data_path = Path(config['paths']['processed_data']) / 'cleaned_data.csv'
    
    results = train_prophet_model(
        data_path=data_path,
        config=config,
        save_model_flag=True
    )
    
    context['ti'].xcom_push(key='model_version', value=results['model_version'])
    context['ti'].xcom_push(key='validation_mape', value=results['validation_metrics']['mape'])
    
    return f"Model trained: {results['model_version']}"


def generate_forecast_task(**context):
    """Generate forecast using trained model."""
    predictions = generate_prophet_forecast(
        model_path=None,  # Uses latest model
        periods=config['model'].get('horizon_days', 30),
        save_output=True
    )
    
    context['ti'].xcom_push(key='forecast_count', value=len(predictions))
    
    return f"Generated {len(predictions)} predictions"


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

generate_forecast_op = PythonOperator(
    task_id='generate_forecast',
    python_callable=generate_forecast_task,
    provide_context=True,
    dag=dag,
)

# Define dependencies
validate_data_op >> train_model_op >> generate_forecast_op
