
from __future__ import annotations
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from energy_forecasting.pipline.batch_prediction_pipeline import BatchPredictionPipeline


def run_batch_prediction_pipeline():
    pipeline = BatchPredictionPipeline()
    pipeline.run_pipeline()


default_args = {
    "owner": "Owner",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="energy_mlops_batch_prediction",
    description="Batch prediction for energy MLOps project",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    # schedule_interval="0 * * * *",  # every hour in real life; 
    schedule_interval="*/10 * * * *",  # every 10 minutes for testing
    catchup=False,
    tags=["batch_prediction", "energy"],
) as dag:

    run_batch_prediction = PythonOperator(
        task_id="run_batch_prediction_pipeline",
        python_callable=run_batch_prediction_pipeline,
    )
