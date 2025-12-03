from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

# ----------------------------------------------------------------------
# 0) Provide a fake psycopg2 module BEFORE importing project code
#    so that `from psycopg2.extras import execute_batch` works.
# ----------------------------------------------------------------------
if "psycopg2" not in sys.modules:
    fake_psycopg2 = types.ModuleType("psycopg2")
    fake_extras = types.ModuleType("psycopg2.extras")

    def _fake_execute_batch(*args, **kwargs):
        # In the real world this would send batched inserts to Postgres.
        # For this offline smoke test we just do nothing.
        return None

    fake_extras.execute_batch = _fake_execute_batch
    fake_psycopg2.extras = fake_extras

    sys.modules["psycopg2"] = fake_psycopg2
    sys.modules["psycopg2.extras"] = fake_extras

# it's safe to import the pipeline that depends on psycopg2.extras.execute_batch
from energy_forecasting.pipline import batch_prediction_pipeline as bp_module
from energy_forecasting.entity.artifact_entity import BatchPredictionArtifact


def test_batch_prediction_pipeline_runs_offline_with_fake_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Offline smoke test for BatchPredictionPipeline:

    - monkeypatches BatchPrediction so it does NOT touch DB / ML models
    - writes a tiny predictions CSV into tmp_path
    - runs BatchPredictionPipeline.run_pipeline()
    - asserts that an artifact is returned and the CSV exists
    """

    class FakeBatchPrediction:
        def __init__(self, config) -> None:
            self.config = config

        def initiate_batch_prediction(self) -> BatchPredictionArtifact:
            n_rows = 8
            df = pd.DataFrame(
                {
                    "unique_id": ["meter_1"] * n_rows,
                    "ds": pd.date_range("2024-01-01", periods=n_rows, freq="H"),
                    "yhat": [1.23] * n_rows,
                }
            )

            preds_path = tmp_path / "fake_batch_predictions.csv"
            df.to_csv(preds_path, index=False)

            return BatchPredictionArtifact(
                predictions_path=preds_path,
                n_predictions=len(df),
            )

    # Use FakeBatchPrediction inside the real pipeline module
    monkeypatch.setattr(bp_module, "BatchPrediction", FakeBatchPrediction)

    BatchPredictionPipeline = bp_module.BatchPredictionPipeline
    pipeline = BatchPredictionPipeline()

    artifact = pipeline.run_pipeline()

    assert isinstance(artifact, BatchPredictionArtifact)
    assert artifact.n_predictions == 8
    assert artifact.predictions_path.is_file()

    preds = pd.read_csv(artifact.predictions_path)
    assert len(preds) == 8
    assert {"unique_id", "ds", "yhat"}.issubset(preds.columns)
