from __future__ import annotations

from typing import Optional

import pandas as pd

from energy_forecasting.entity.config_entity import (
    DataIngestionConfig,
    DatabaseConfig,
)
from energy_forecasting.entity.artifact_entity import DataIngestionArtifact
from energy_forecasting.data_access.postgres_loader import (
    load_recent_table_from_db,
)
from energy_forecasting.exception import EnergyException
from energy_forecasting.logger import logger


class DataIngestion:
    """
    Data ingestion component (DB-only).

    Always:
      - reads history from Postgres using a DatabaseConfig
      - writes a raw CSV snapshot into the artifacts directory
    """

    def __init__(
        self,
        data_ingestion_config: DataIngestionConfig,
        database_config: Optional[DatabaseConfig] = None,
    ) -> None:
        self.config = data_ingestion_config
        self.database_config = database_config

    # ------------------------------------------------------------------
    # DB-backed ingestion
    # ------------------------------------------------------------------
    def _ingest_from_db(self) -> DataIngestionArtifact:
        """
        Load recent history from Postgres and save it as raw_energy_data.csv
        under the current artifacts run directory.
        """
        if self.database_config is None:
            raise EnergyException(
                "DataIngestion requires a DatabaseConfig in DB-only mode, "
                "but no DatabaseConfig was provided."
            )

        df = load_recent_table_from_db(self.database_config)

        if df.empty:
            raise EnergyException(
                "DB ingestion returned an empty dataframe; nothing to ingest."
            )

        # Normalize columns so they align with the rest of the pipeline
        if "ts" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"ts": "timestamp"})

        if "hour" not in df.columns and "timestamp" in df.columns:
            # derive hour of day from timestamp
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

        # Ensure output directory exists
        self.config.raw_data_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(self.config.raw_data_path, index=False)

        logger.info(
            "DB ingestion: wrote %d rows to %s",
            len(df),
            self.config.raw_data_path,
        )

        return DataIngestionArtifact(
            raw_data_path=self.config.raw_data_path,
            n_rows=len(df),
        )

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logger.info("Starting DataIngestion in DB-only mode.")
        try:
            artifact = self._ingest_from_db()

            logger.info(
                "DataIngestion completed: path=%s, rows=%d",
                artifact.raw_data_path,
                artifact.n_rows,
            )
            return artifact
        except Exception as e:
            raise EnergyException(e) from e
