from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import os   

from energy_forecasting.constants import ARTIFACTS_DIR, DATA_DIR, MODELS_DIR, ROOT_DIR


@dataclass
class TrainingPipelineConfig:
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def artifacts_root(self) -> Path:
        return ARTIFACTS_DIR / self.timestamp


@dataclass
class DataIngestionConfig:
    training_pipeline_config: TrainingPipelineConfig

    @property
    def raw_data_dir(self) -> Path:
        # still useful: we can keep a CSV snapshot of DB data in artifacts
        return self.training_pipeline_config.artifacts_root / "data_ingestion"

    @property
    def raw_data_path(self) -> Path:
        return self.raw_data_dir / "raw_energy_data.csv"



@dataclass
class DataTransformationConfig:
    training_pipeline_config: TrainingPipelineConfig
    test_hours: int


    # optional explicit feature list from config.yaml
    # e.g. data_transformation.feature_cols: ["is_holiday", "temperature", ...]
    feature_cols: Optional[List[str]] = None

    @property
    def transformed_dir(self) -> Path:
        return self.training_pipeline_config.artifacts_root / "data_transformation"

    @property
    def train_array_path(self) -> Path:
        return self.transformed_dir / "train.csv"

    @property
    def test_array_path(self) -> Path:
        return self.transformed_dir / "test.csv"



@dataclass
class ModelTrainerConfig:
    training_pipeline_config: TrainingPipelineConfig

    # generic
    model_type: str = "mlforecast"
    model_file_name: str = "model.pkl"
    metrics_file_name: str = "metrics.json"

    # --- MLForecast-specific config (all config-driven) ---
    # columns
    mlf_unique_id_col: str = "unique_id"
    mlf_time_col: str = "ds"
    mlf_target_col: str = "y"

    # time settings
    mlf_freq: str = "h"
    mlf_h: int = 24
    mlf_n_windows: int = 4

    # features
    mlf_dynamic_features: Optional[List[str]] = None
    mlf_static_features: Optional[List[str]] = None

    # lags / differences / transforms / calendar features
    mlf_lags: Optional[List[int]] = None
    mlf_differences: Optional[List[int]] = None
    mlf_lag_transforms: Optional[Dict[str, Any]] = None
    mlf_date_features: Optional[List[str]] = None

    # models + hyperparameters (config-driven)
    mlf_models: Optional[Dict[str, Any]] = None

    @property
    def model_dir(self) -> Path:
        return self.training_pipeline_config.artifacts_root / "model_trainer"

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_file_name

    @property
    def metrics_path(self) -> Path:
        return self.model_dir / self.metrics_file_name


@dataclass
class ModelEvaluationConfig:
    """
    Configuration for model evaluation and 'best model' registry.
    """
    training_pipeline_config: TrainingPipelineConfig

    @property
    def best_model_dir(self) -> Path:
        # Central location where the best (production) model is stored
        return MODELS_DIR / "best_model"

    @property
    def best_model_path(self) -> Path:
        return self.best_model_dir / "model.pkl"

    @property
    def best_metrics_path(self) -> Path:
        return self.best_model_dir / "metrics.json"

@dataclass
class BatchPredictionConfig:
    """
    Configuration for batch prediction:

    - n_future_hours: horizon to predict (in "hours" units matching freq)
    - pred_table: optional DB table name to store predictions
    - exog_source_type: how to obtain FUTURE exogenous variables
        "none" | "file" | "api"  (you can add more types later)
    - exog_source_params: config for that source (depends on type)
        e.g. for 'file':
            {"file_path": "data/weather_forecast.csv",
             "time_col": "ds"}
    """
    training_pipeline_config: TrainingPipelineConfig

    n_future_hours: int = 24

    # Optional: DB table for storing predictions
    pred_table: Optional[str] = None

    # How to get FUTURE exogenous features
    exog_source_type: str = "none"  # db", ...
    exog_source_params: Optional[Dict[str, Any]] = None

    @property
    def predictions_dir(self) -> Path:
        return self.training_pipeline_config.artifacts_root / "batch_prediction"

    @property
    def predictions_path(self) -> Path:
        return self.predictions_dir / "predictions.csv"

    @property
    def best_model_path(self) -> Path:
        """
        Stable location of the best model, updated by ModelEvaluation.
        """
        return MODELS_DIR / "best_model" / "model.pkl"

    @property
    def best_metrics_path(self) -> Path:
        """
        Stable location of the best model's metrics.json
        (copied by ModelEvaluation).
        """
        return MODELS_DIR / "best_model" / "metrics.json"



@dataclass
class DatabaseConfig:
    host: str
    port: int
    db_name: str
    user: str
    password: str
    table: str
    hours_history: int 


@dataclass
class MLflowConfig:
    """
    Configuration for MLflow tracking.
    """
    experiment_name: str = "energy_mlops"

    # If MLFLOW_TRACKING_URI is set (e.g. in docker-compose), use that.
    # Otherwise fall back to local file-based ./mlruns.
    tracking_uri: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_TRACKING_URI",
            f"file://{(ROOT_DIR / 'mlruns').resolve()}",
        )
    )

    # optional: name for the "production" model in the registry
    registered_model_name: str = "energy_load_forecaster"