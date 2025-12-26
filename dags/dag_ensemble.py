"""
Ensemble forecasting DAG.

Combines predictions from Prophet, SARIMA, and DeepAR.
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago

import sys
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

from ensemble.model_combiner import create_ensemble_forecast
from common.config_loader import load_config

# Load configuration
config = load_config('ensemble')
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
    'ensemble_forecasting',
    default_args=default_args,
    description='Ensemble model combining Prophet, SARIMA, and DeepAR',
    schedule_interval=airflow_config.get('schedule_interval', '@daily'),
    start_date=days_ago(1),
    catchup=airflow_config.get('catchup', False),
    max_active_runs=airflow_config.get('max_active_runs', 1),
    tags=['forecasting', 'ensemble', 'multi-model'],
)


def combine_predictions_task(**context):
    """Combine predictions from all models."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Paths to individual model predictions
    prophet_path = Path(config['paths']['predictions']) / 'prophet' / f'forecast_{timestamp}.csv'
    sarima_path = Path(config['paths']['predictions']) / 'sarima' / f'forecast_{timestamp}.csv'
    deepar_path = Path(config['paths']['predictions']) / 'deepar' / f'forecast_{timestamp}.csv'
    
    # Create ensemble forecast
    ensemble_forecast = create_ensemble_forecast(
        prophet_predictions_path=prophet_path,
        sarima_predictions_path=sarima_path,
        deepar_predictions_path=deepar_path if deepar_path.exists() else None,
        config=config
    )
    
    # Save ensemble forecast
    output_dir = Path(config['ensemble']['paths']['predictions_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'forecast_{timestamp}.csv'
    
    ensemble_forecast.to_csv(output_path, index=False)
    
    context['ti'].xcom_push(key='ensemble_forecast_path', value=str(output_path))
    context['ti'].xcom_push(key='forecast_count', value=len(ensemble_forecast))
    
    return f"Generated {len(ensemble_forecast)} ensemble predictions"


# Wait for Prophet DAG
wait_for_prophet = ExternalTaskSensor(
    task_id='wait_for_prophet',
    external_dag_id='prophet_forecasting',
    external_task_id='generate_forecast',
    timeout=3600,
    allowed_states=['success'],
    failed_states=['failed', 'skipped'],
    mode='poke',
    poke_interval=60,
    dag=dag,
)

# Wait for SARIMA DAG
wait_for_sarima = ExternalTaskSensor(
    task_id='wait_for_sarima',
    external_dag_id='sarima_forecasting',
    external_task_id='generate_forecast',
    timeout=3600,
    allowed_states=['success'],
    failed_states=['failed', 'skipped'],
    mode='poke',
    poke_interval=60,
    dag=dag,
)

# Wait for DeepAR DAG (optional)
wait_for_deepar = ExternalTaskSensor(
    task_id='wait_for_deepar',
    external_dag_id='deepar_forecasting',
    external_task_id='generate_forecast',
    timeout=3600,
    allowed_states=['success'],
    failed_states=['failed', 'skipped'],
    mode='poke',
    poke_interval=60,
    dag=dag,
)

# Combine predictions
combine_predictions_op = PythonOperator(
    task_id='combine_predictions',
    python_callable=combine_predictions_task,
    provide_context=True,
    dag=dag,
)

# Define dependencies
[wait_for_prophet, wait_for_sarima, wait_for_deepar] >> combine_predictions_op
