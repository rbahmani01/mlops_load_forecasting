from __future__ import annotations

from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

from energy_forecasting.pipline.training_pipeline import TrainPipeline



def run_training_pipeline() -> None:
    """
    it runs ingestion + transformation + training + evaluation.
    """
    pipeline = TrainPipeline()
    pipeline.run_pipeline()

default_args = {
    "owner": "airflow",              
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="energy_mlops_training",         
    description="energy training",
    default_args=default_args,
    # schedule="0 2 * * *",                  # daily at 02:00
    schedule="*/30 * * * *",   # every 30 minutes (for testing)

    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["energy", "training", "mlops"],
) as dag:


    run_training = PythonOperator(
        task_id="run_training_pipeline",
        python_callable=run_training_pipeline,
    )

    run_training
