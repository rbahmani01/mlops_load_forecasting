from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionArtifact:
    raw_data_path: Path
    n_rows: int


@dataclass
class DataTransformationArtifact:
    train_array_path: Path
    test_array_path: Path
    n_train_rows: int
    n_test_rows: int


@dataclass
class ModelTrainerArtifact:
    model_path: Path
    metrics_path: Path
    rmse: float
    mae: float
    test_data_path: Path



@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    best_model_path: Path | None
    best_metrics_path: Path | None
    current_model_path: Path
    current_metrics_path: Path

@dataclass
class BatchPredictionArtifact:
    predictions_path: Path
    n_predictions: int


