from pathlib import Path

import pytest

from energy_forecasting.components.model_trainer import ModelTrainer
from energy_forecasting.entity.artifact_entity import DataTransformationArtifact
from energy_forecasting.entity.config_entity import (
    TrainingPipelineConfig,
    ModelTrainerConfig,
)
from energy_forecasting.exception import EnergyException


def _dummy_data_transformation_artifact(tmp_path: Path) -> DataTransformationArtifact:
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    # Minimal CSVs – we don't actually read them in these tests
    train.write_text("ds,y\n2024-01-01 00:00,1\n", encoding="utf-8")
    test.write_text("ds,y\n2024-01-01 01:00,2\n", encoding="utf-8")

    return DataTransformationArtifact(
        train_array_path=train,
        test_array_path=test,
        n_train_rows=1,
        n_test_rows=1,
    )


def test_model_trainer_build_lag_transforms_invalid_shape_raises(tmp_path: Path) -> None:
    """mlf_lag_transforms must be a mapping; anything else should raise EnergyException."""
    tp_cfg = TrainingPipelineConfig()
    mt_cfg = ModelTrainerConfig(training_pipeline_config=tp_cfg)

    # Wrong type: list instead of dict
    mt_cfg.mlf_lag_transforms = ["not-a-dict"]  # type: ignore[assignment]

    dt_artifact = _dummy_data_transformation_artifact(tmp_path)

    trainer = ModelTrainer(
        config=mt_cfg,
        data_transformation_artifact=dt_artifact,
        mlflow_config=None,
    )

    with pytest.raises(EnergyException):
        trainer._build_lag_transforms()


def test_model_trainer_build_lag_transforms_valid(tmp_path: Path) -> None:
    """Valid YAML-like config should produce lag -> list[transform] mapping."""
    tp_cfg = TrainingPipelineConfig()
    mt_cfg = ModelTrainerConfig(training_pipeline_config=tp_cfg)

    mt_cfg.mlf_lag_transforms = {
        "1": [{"type": "ExpandingMean"}],
        "24": [{"type": "RollingMean", "window_size": 48}],
    }

    dt_artifact = _dummy_data_transformation_artifact(tmp_path)

    trainer = ModelTrainer(
        config=mt_cfg,
        data_transformation_artifact=dt_artifact,
        mlflow_config=None,
    )

    lag_transforms = trainer._build_lag_transforms()

    # integer keys 1 and 24
    assert set(lag_transforms.keys()) == {1, 24}

    # each key must map to a non-empty list of transform objects
    assert len(lag_transforms[1]) >= 1
    assert len(lag_transforms[24]) >= 1

    # import inside here to avoid hard dependency at import-time of tests
    from mlforecast.lag_transforms import ExpandingMean, RollingMean

    assert isinstance(lag_transforms[1][0], ExpandingMean)
    assert isinstance(lag_transforms[24][0], RollingMean)
