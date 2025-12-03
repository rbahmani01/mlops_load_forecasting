from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from energy_forecasting.pipline.training_pipeline import TrainPipeline
from energy_forecasting.components import data_ingestion as di_module
from energy_forecasting.entity.artifact_entity import DataIngestionArtifact


def _make_synthetic_history(n_hours: int = 500) -> pd.DataFrame:
    """
    Build a small but realistic hourly history, compatible with config.yaml:

    - ds          : timestamp
    - unique_id   : single meter id
    - y           : target load
    - is_holiday  : dynamic feature (in config.feature_cols)
    - temperature : dynamic feature (in config.feature_cols)
    """
    idx = pd.date_range("2024-01-01", periods=n_hours, freq="H")

    df = pd.DataFrame(
        {
            "ds": idx,
            "unique_id": ["meter_1"] * len(idx),
            "y": range(len(idx)),            # simple increasing series
            "is_holiday": [0] * len(idx),    # no holidays
            "temperature": [20.0] * len(idx) # constant temp
        }
    )
    return df


def test_train_pipeline_runs_offline_with_synthetic_ingestion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    End-to-end-ish smoke test for TrainPipeline that:
      - avoids Postgres and Kaggle completely
      - feeds a synthetic CSV into the pipeline via a monkeypatched DataIngestion
      - lets DataTransformation, ModelTrainer, and ModelEvaluation run normally

    The main assertion: pipeline finishes without raising an exception.
    """

    # ------------------------------------------------------------------
    # 1) Build synthetic "DB" history and write a CSV snapshot
    # ------------------------------------------------------------------
    df = _make_synthetic_history(n_hours=500)
    raw_csv_path = tmp_path / "raw_energy_data.csv"
    df.to_csv(raw_csv_path, index=False)

    # ------------------------------------------------------------------
    # 2) Monkeypatch DataIngestion.initiate_data_ingestion
    #    so it returns our synthetic CSV instead of hitting Postgres
    # ------------------------------------------------------------------
    def _fake_initiate_data_ingestion(self: Any) -> DataIngestionArtifact:
        # 'self' is a DataIngestion instance, but we ignore DB config completely.
        return DataIngestionArtifact(
            raw_data_path=raw_csv_path,
            n_rows=len(df),
        )

    # Patch the method on the class, so any new DataIngestion instance uses it.
    monkeypatch.setattr(
        di_module.DataIngestion,
        "initiate_data_ingestion",
        _fake_initiate_data_ingestion,
    )

    # ------------------------------------------------------------------
    # 3) Make sure we don't accidentally talk to a remote MLflow
    #    (fallback to local file-based mlruns in the repo)
    # ------------------------------------------------------------------
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    # Optionally also neutralize DB_* envs (not strictly needed here,
    # because we never call the real ingestion logic):
    for var in ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_TABLE"]:
        monkeypatch.delenv(var, raising=False)

    # ------------------------------------------------------------------
    # 4) Run the full TrainPipeline
    # ------------------------------------------------------------------
    pipeline = TrainPipeline()

    # If anything in ingestion / transformation / trainer / evaluation is broken,
    # this will raise and the test will fail.
    pipeline.run_pipeline()

    # We don't assert specific artifact paths here, because TrainPipelineConfig
    # uses a timestamp in the path, but as a smoke test it's enough that
    # the pipeline completes successfully.
