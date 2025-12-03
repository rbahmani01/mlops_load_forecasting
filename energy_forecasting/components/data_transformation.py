from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from energy_forecasting.entity.config_entity import DataTransformationConfig
from energy_forecasting.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from energy_forecasting.logger import logger


class DataTransformation:
    """
    Simple transformation for df_mlf-style data.

    Assumes raw CSV has at least:
      - 'ds'        : timestamp column
      - 'y'         : target column
      - 'unique_id' : series id (not a feature)

    We do NOT create new time features here.
    We just:
      - sort by time
      - build X from existing features (numeric + encoded categoricals)
      - build y from 'y'
      - time-based train/test split (no shuffle)
      - save numpy dicts: {"X": ..., "y": ...}
    """

    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config

    def _load_raw(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        logger.info(
            "Loaded raw data from %s with %d rows and %d columns",
            path,
            len(df),
            df.shape[1],
        )
        return df

    def initiate_data_transformation(
        self, ingestion_artifact: DataIngestionArtifact
    ) -> DataTransformationArtifact:
        logger.info("Starting DataTransformation (time-based train/test CSV split).")

        df = self._load_raw(ingestion_artifact.raw_data_path)

        # ---- 1) Basic checks & sorting by time ----
        if "y" not in df.columns:
            raise ValueError(
                "Expected target column 'y' in the data. "
                f"Found columns: {list(df.columns)}"
            )

        if "ds" in df.columns:
            df["ds"] = pd.to_datetime(df["ds"])
            # sort by time, then by unique_id if present (more stable)
            sort_cols = ["ds"]
            if "unique_id" in df.columns:
                sort_cols.append("unique_id")
            df = df.sort_values(sort_cols).reset_index(drop=True)
            logger.info("Data sorted by %s.", sort_cols)

        n_rows_raw = len(df)
        logger.info("Raw data after sorting: %d rows", n_rows_raw)

        # ---- 2) Select columns to keep ----
        # Always keep time, id and target
        base_cols = []
        if "ds" in df.columns:
            base_cols.append("ds")
        if "unique_id" in df.columns:
            base_cols.append("unique_id")

        if self.config.feature_cols:
            # Use explicit feature list from config.yaml
            feature_cols = self.config.feature_cols
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f"The following feature_cols from config are missing in the data: {missing}. "
                    f"Available columns: {list(df.columns)}"
                )
            cols_to_keep = base_cols + feature_cols + ["y"]
            logger.info("Using feature_cols from config: %s", feature_cols)
        else:
            # Fallback: keep everything
            cols_to_keep = list(df.columns)
            logger.info(
                "No feature_cols specified in config; keeping all columns for CSV."
            )

        df = df[cols_to_keep].copy()
        logger.info("Columns kept for transformation: %s", cols_to_keep)

        # ---- 3) Time-based train/test split ----
        # Train: all days except the last calendar day
        # Test : the last calendar day of history
        #
        # Requires at least 2 distinct calendar days in the data.

        if "ds" not in df.columns:
            raise ValueError(
                "Expected time column 'ds' for time-based split, "
                f"found columns: {list(df.columns)}"
            )

        # Ensure sorted by time
        df = df.sort_values("ds").reset_index(drop=True)

        # Last timestamp in the dataset
        last_ts = df["ds"].max()

        hours = self.config.test_hours
        last_hours_start = last_ts - pd.Timedelta(hours=hours)

        test_mask = df["ds"] > last_hours_start
        test_df = df.loc[test_mask].reset_index(drop=True)
        train_df = df.loc[~test_mask].reset_index(drop=True)


        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError(
                "Time-based split with last-day-as-test produced empty train or test set. "
                "Need at least 2 distinct calendar days in the data."
            )

        n_train = len(train_df)
        n_test = len(test_df)

        logger.info(
            "Time-based split (last-%d-hours-as-test): n_train=%d, n_test=%d (total=%d). "
            "Train until %s, test from %s.",
            self.config.test_hours,
            n_train,
            n_test,
            n_train + n_test,
            train_df["ds"].max() if n_train > 0 else None,
            test_df["ds"].min() if n_test > 0 else None,
        )




        # ---- 4) Save CSVs ----
        self.config.transformed_dir.mkdir(parents=True, exist_ok=True)

        train_csv_path = self.config.train_array_path
        test_csv_path = self.config.test_array_path

        train_df.to_csv(train_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)

        logger.info(
            "Saved train CSV to %s and test CSV to %s",
            train_csv_path,
            test_csv_path,
        )

        # ---- 5) Build artifact ----
        artifact = DataTransformationArtifact(
            train_array_path=train_csv_path,
            test_array_path=test_csv_path,
            n_train_rows=len(train_df),
            n_test_rows=len(test_df),
        )

        logger.info(
            "DataTransformation completed: train_rows=%d, test_rows=%d",
            artifact.n_train_rows,
            artifact.n_test_rows,
        )

        return artifact