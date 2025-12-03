from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import os
from dotenv import load_dotenv

import yaml

from energy_forecasting.constants import ROOT_DIR
from energy_forecasting.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    BatchPredictionConfig,
    DatabaseConfig,
    MLflowConfig,   

)
from energy_forecasting.logger import logger


CONFIG_DIR = ROOT_DIR / "config"
CONFIG_FILE_PATH = CONFIG_DIR / "config.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        logger.warning("Config file %s not found. Using defaults.", path)
        return {}
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a top-level mapping")
    return data


class ConfigManager:
    """
    Loads config/config.yaml and builds config dataclasses.

    - If a section/field is missing in YAML, the dataclass default is used.
    - YAML values override defaults where present.
    """

    def __init__(self, config_file: Optional[Path] = None) -> None:
        self._config_file = config_file or CONFIG_FILE_PATH
        self._raw_config = _load_yaml(self._config_file)
        load_dotenv()

        # One TrainingPipelineConfig per pipeline run
        self.training_pipeline_config = TrainingPipelineConfig()
        logger.info(
            "ConfigManager initialized with artifacts_root=%s",
            self.training_pipeline_config.artifacts_root,
        )

        db_section = self._section("database")

        self._database_config = DatabaseConfig(
            host=os.getenv("DB_HOST", db_section.get("host", "localhost")),
            port=int(os.getenv("DB_PORT", db_section.get("port", 5432))),
            db_name=os.getenv("DB_NAME", db_section.get("db_name", "energy")),
            user=os.getenv("DB_USER", db_section.get("user", "energy_user")),
            # if password is not in YAML, fall back to env or a default
            password=os.getenv(
                "DB_PASSWORD",
                db_section.get("password"),
            ),
            table=os.getenv("DB_TABLE", db_section.get("table", "df_mlf")),
            hours_history=int(
                os.getenv(
                    "DB_HOURS_HISTORY",
                    db_section.get("hours_history", 8760),
                )
            ),
        )


        self._mlflow_config = MLflowConfig()

    # Helpers to get section from YAML
    def _section(self, name: str) -> Dict[str, Any]:
        section = self._raw_config.get(name, {})
        if section is None:
            section = {}
        if not isinstance(section, dict):
            raise ValueError(f"Config section '{name}' must be a mapping")
        return section

    # ---- Specific configs ----

    def get_database_config(self) -> DatabaseConfig:
        return self._database_config
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        # DB-only: no n_samples, no source
        return DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)



    def get_data_transformation_config(self) -> DataTransformationConfig:
        section = self._section("data_transformation")

        # read from YAML or fall back to 24
        test_hours = int(section.get("test_hours", 24))

        cfg = DataTransformationConfig(
            training_pipeline_config=self.training_pipeline_config,
            test_hours=test_hours,
        )

        if "feature_cols" in section and section["feature_cols"] is not None:
            cfg.feature_cols = list(section["feature_cols"])

        return cfg



    def get_model_trainer_config(self) -> ModelTrainerConfig:
        section = self._section("model_trainer")
        cfg = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)

        # general type (optional)
        if "model_type" in section:
            cfg.model_type = str(section["model_type"]).lower()

        # === MLForecast sub-section ===
        mlf = section.get("mlforecast", {}) or {}

        # column names
        if "unique_id_col" in mlf:
            cfg.mlf_unique_id_col = str(mlf["unique_id_col"])
        if "time_col" in mlf:
            cfg.mlf_time_col = str(mlf["time_col"])
        if "target_col" in mlf:
            cfg.mlf_target_col = str(mlf["target_col"])

        # features
        if "dynamic_features" in mlf and mlf["dynamic_features"] is not None:
            cfg.mlf_dynamic_features = list(mlf["dynamic_features"])
        if "static_features" in mlf and mlf["static_features"] is not None:
            cfg.mlf_static_features = list(mlf["static_features"])

        # time setup
        if "freq" in mlf:
            cfg.mlf_freq = str(mlf["freq"])
        if "h" in mlf:
            cfg.mlf_h = int(mlf["h"])
        if "n_windows" in mlf:
            cfg.mlf_n_windows = int(mlf["n_windows"])

        # lags / differences
        if "lags" in mlf and mlf["lags"] is not None:
            cfg.mlf_lags = list(mlf["lags"])
        if "differences" in mlf and mlf["differences"] is not None:
            cfg.mlf_differences = list(mlf["differences"])

        # lag transforms (dict of lag -> list[transform_spec])
        if "lag_transforms" in mlf and mlf["lag_transforms"] is not None:
            cfg.mlf_lag_transforms = dict(mlf["lag_transforms"])

        # calendar / date features
        if "date_features" in mlf and mlf["date_features"] is not None:
            cfg.mlf_date_features = list(mlf["date_features"])

        # models + hyperparameter lists
        if "models" in mlf and mlf["models"] is not None:
            cfg.mlf_models = dict(mlf["models"])

        return cfg




    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        # all paths are derived from TrainingPipelineConfig / MODELS_DIR
        _ = self._section("model_evaluation")  # placeholder if you want thresholds later
        cfg = ModelEvaluationConfig(training_pipeline_config=self.training_pipeline_config)
        return cfg

    def get_batch_prediction_config(
        self,
        n_future_hours_override: Optional[int] = None,
    ) -> BatchPredictionConfig:
        section = self._section("batch_prediction")

        cfg = BatchPredictionConfig(
            training_pipeline_config=self.training_pipeline_config
        )

        # --- horizon ---
        if n_future_hours_override is not None:
            cfg.n_future_hours = int(n_future_hours_override)
        elif "n_future_hours" in section:
            cfg.n_future_hours = int(section["n_future_hours"])

        # --- exogenous source (generic, config-driven) ---
        exog_cfg = section.get("exog_source") or {}
        if isinstance(exog_cfg, dict) and exog_cfg:
            # accept both 'kind' and 'type'
            kind = exog_cfg.get("kind", exog_cfg.get("type", "none"))
            cfg.exog_source_type = str(kind).lower()

            params = exog_cfg.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError("batch_prediction.exog_source.params must be a mapping")
            cfg.exog_source_params = params

        return cfg





    def get_mlflow_config(self) -> MLflowConfig:
        return self._mlflow_config

