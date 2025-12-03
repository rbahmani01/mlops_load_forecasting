from pathlib import Path

import pandas as pd

from energy_forecasting.components.data_transformation import DataTransformation
from energy_forecasting.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from energy_forecasting.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataTransformationConfig,
)


def _make_synthetic_history(n_hours: int = 48) -> pd.DataFrame:
    """Small synthetic hourly time series with some extra features."""
    idx = pd.date_range("2024-01-01", periods=n_hours, freq="H")
    df = pd.DataFrame(
        {
            "ds": idx,
            "unique_id": ["meter_1"] * len(idx),
            "y": range(len(idx)),
            "is_holiday": [0] * len(idx),
            "temperature": [20.0] * len(idx),
            "extra_feature": range(len(idx)),
        }
    )
    return df


def test_data_transformation_time_split_and_feature_cols(tmp_path: Path) -> None:
    """DataTransformation should:
    - split the last N hours into test set
    - keep only the requested feature_cols
    """
    df = _make_synthetic_history(n_hours=48)
    raw_path = tmp_path / "raw_energy_data.csv"
    df.to_csv(raw_path, index=False)

    # Pipeline + ingestion config (only needed for paths)
    tp_cfg = TrainingPipelineConfig()
    _ = DataIngestionConfig(training_pipeline_config=tp_cfg)

    # Ingestion artifact points to our synthetic CSV
    ingestion_artifact = DataIngestionArtifact(
        raw_data_path=raw_path,
        n_rows=len(df),
    )

    # Use last 24 hours as test set
    dt_cfg = DataTransformationConfig(
        training_pipeline_config=tp_cfg,
        test_hours=24,
    )
    dt_cfg.feature_cols = ["is_holiday", "temperature"]

    transformer = DataTransformation(config=dt_cfg)
    artifact: DataTransformationArtifact = transformer.initiate_data_transformation(
        ingestion_artifact
    )

    # Train/test CSVs must exist
    assert artifact.train_array_path.is_file()
    assert artifact.test_array_path.is_file()

    train_df = pd.read_csv(artifact.train_array_path)
    test_df = pd.read_csv(artifact.test_array_path)

    # Total rows conserved
    assert len(train_df) + len(test_df) == len(df)

    # Exactly 24 rows in test
    assert len(test_df) == 24

    # Test timestamps should be strictly after train timestamps
    train_max = pd.to_datetime(train_df["ds"]).max()
    test_min = pd.to_datetime(test_df["ds"]).min()
    assert test_min > train_max

    # Columns: time/id/target + explicitly requested features
    expected_cols = {"ds", "unique_id", "y", "is_holiday", "temperature"}
    assert set(train_df.columns) == expected_cols
    assert set(test_df.columns) == expected_cols
    assert "extra_feature" not in train_df.columns
