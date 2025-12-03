from __future__ import annotations

import sys
import types
import inspect

import pytest

# ----------------------------------------------------------------------
# Fake psycopg2 (with extras.execute_batch) so importing the pipeline works
# ----------------------------------------------------------------------
if "psycopg2" not in sys.modules:
    fake_psycopg2 = types.ModuleType("psycopg2")
    fake_extras = types.ModuleType("psycopg2.extras")

    def _fake_execute_batch(*args, **kwargs):
        return None

    fake_extras.execute_batch = _fake_execute_batch
    fake_psycopg2.extras = fake_extras

    sys.modules["psycopg2"] = fake_psycopg2
    sys.modules["psycopg2.extras"] = fake_extras

from energy_forecasting.pipline.training_pipeline import TrainPipeline
from energy_forecasting.pipline.batch_prediction_pipeline import BatchPredictionPipeline


def test_train_pipeline_has_run_pipeline_method() -> None:
    pipeline = TrainPipeline()
    assert hasattr(pipeline, "run_pipeline")

    method = getattr(pipeline, "run_pipeline")
    assert callable(method)

    sig = inspect.signature(method)
    assert all(
        p.default is not inspect._empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        for p in sig.parameters.values()
    )


def test_batch_prediction_pipeline_has_run_pipeline_method() -> None:
    pipeline = BatchPredictionPipeline()
    assert hasattr(pipeline, "run_pipeline")

    method = getattr(pipeline, "run_pipeline")
    assert callable(method)

    sig = inspect.signature(method)
    assert all(
        p.default is not inspect._empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        for p in sig.parameters.values()
    )
