from __future__ import annotations
import os
import json
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

import joblib
import psycopg2
from psycopg2.extras import execute_batch

from energy_forecasting.configuration.configuration import ConfigManager
from energy_forecasting.data_access.postgres_loader import load_recent_table_from_db
from energy_forecasting.entity.config_entity import (
    BatchPredictionConfig,
    DatabaseConfig,
)
from energy_forecasting.entity.artifact_entity import BatchPredictionArtifact
from energy_forecasting.exception import EnergyException
from energy_forecasting.logger import logger



class BatchPrediction:
    """
    Generic batch prediction component for MLForecast models.

    It:
      - loads the best MLForecast model & its mlforecast_config
      - loads recent *history* from DB (df_mlf or whatever table is in DatabaseConfig)
      - obtains *future exogenous features* from db
      - predicts the next n_future_hours
      - saves predictions to:
          - artifacts/<run_id>/batch_prediction/predictions.csv
    """

    def __init__(
        self,
        config: BatchPredictionConfig,
        database_config: Optional[DatabaseConfig] = None,
    ) -> None:
        self.config = config
        self.config.predictions_dir.mkdir(parents=True, exist_ok=True)

        if database_config is None:
            cm = ConfigManager()
            database_config = cm.get_database_config()
        self.db_cfg = database_config

    # ------------------------------------------------------------------
    # Helpers: model & config
    # ------------------------------------------------------------------
    def _load_best_model_and_mlf_config(self) -> Tuple[Any, Dict[str, Any]]:
        model_path = self.config.best_model_path
        metrics_path = self.config.best_metrics_path

        if not model_path.exists():
            raise EnergyException(f"Best model not found at {model_path}")
        if not metrics_path.exists():
            raise EnergyException(f"Best metrics not found at {metrics_path}")

        logger.info("Loading best MLForecast model from %s", model_path)
        model = joblib.load(model_path)

        with metrics_path.open("r") as f:
            metrics = json.load(f)

        mlf_cfg = metrics.get("mlforecast_config")
        if not mlf_cfg:
            raise EnergyException(
                f"'mlforecast_config' not found in {metrics_path}. "
                "It is needed for id/time/feature columns & horizon."
            )

        logger.info("Loaded mlforecast_config: %s", mlf_cfg)
        return model, mlf_cfg

    # ------------------------------------------------------------------
    # Helpers: history
    # ------------------------------------------------------------------
    def _load_history_from_db(self, time_col: str) -> pd.DataFrame:
        """
        Uses the same DatabaseConfig + loader as DataIngestion
        (generic: works for any df_mlf-like table).
        """
        df = load_recent_table_from_db(self.db_cfg)

        if df.empty:
            raise EnergyException(
                "DB history loader returned an empty dataframe; "
                "cannot run batch prediction."
            )

        # Normalize timestamp column name if different
        if "ts" in df.columns and time_col not in df.columns:
            df = df.rename(columns={"ts": time_col})
        if "timestamp" in df.columns and time_col not in df.columns:
            df = df.rename(columns={"timestamp": time_col})

        logger.info(
            "History loaded from DB: %d rows, columns=%s",
            len(df),
            list(df.columns),
        )
        return df

    def _load_future_exog(
        self,
        mlf_cfg: Dict[str, Any],
        series_ids: np.ndarray,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        """
        Wrapper to load FUTURE exogenous variables.

        For now we support only 'db' (df_exog) as the source.
        """
        time_col = mlf_cfg["time_col"]
        dyn_cols: List[str] = mlf_cfg.get("dynamic_features") or []
        static_cols: List[str] = mlf_cfg.get("static_features") or []

        # If the model was trained WITHOUT exogenous features, nothing to load.
        if not dyn_cols and not static_cols:
            logger.info(
                "No dynamic/static features in mlforecast_config; "
                "skipping exogenous loading."
            )
            return None

        # Only support DB exog
        src_type = (self.config.exog_source_type or "none").lower().strip()
        params = self.config.exog_source_params or {}

        if src_type == "db":
            return self._load_future_exog_from_db(
                time_col=time_col,
                dyn_cols=dyn_cols,
                static_cols=static_cols,
                start_ts=start_ts,
                end_ts=end_ts,
                params=params,
            )

        if src_type in ("none", ""):
            raise EnergyException(
                "mlforecast_config has dynamic/static features, but "
                "BatchPredictionConfig.exog_source_type is 'none'. "
                "Configure batch_prediction.exog_source.type: db in config.yaml."
            )

        # everything else is unsupported for now
        raise EnergyException(
            f"Unsupported exog_source_type='{self.config.exog_source_type}'. "
            "Supported: 'db'."
        )

    # ---------------------- DB exog (df_exog) -------------------------
    def _load_future_exog_from_db(
        self,
        time_col: str,
        dyn_cols: List[str],
        static_cols: List[str],
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Load future exogenous variables from a DB table (e.g. df_exog).

        Expected config:

        batch_prediction:
          exog_source:
            type: db
            params:
              table: df_exog
              time_col: ds   # optional, defaults to mlf_cfg[time_col]

        df_exog schema (example from  ETL):
          id SERIAL PRIMARY KEY
          ds TIMESTAMPTZ
          is_holiday, temperature, ...

        We:
          - query rows where ds in (start_ts, end_ts]
          - convert ds from tz-aware UTC -> naive UTC
          - return [time_col] + exog columns (no id_col, global exog).
        """
        table = params.get("table") or params.get("exog_table") or "df_exog"
        time_col_db = params.get("time_col", time_col)


        logger.info(
            "Loading exogenous data from DB table '%s' "
            "for window (%s, %s] ...",
            table,
            start_ts,
            end_ts,
        )

        # make start/end tz-aware UTC for TIMESTAMPTZ comparison in Postgres
        start_ts_utc = start_ts.tz_localize("UTC")
        end_ts_utc = end_ts.tz_localize("UTC")

        conn = psycopg2.connect(
            host=os.getenv("ENERGY_DB_HOST", self.db_cfg.host),
            port=int(os.getenv("ENERGY_DB_PORT", str(self.db_cfg.port))),
            dbname=os.getenv("ENERGY_DB_NAME", self.db_cfg.db_name),
            user=os.getenv("ENERGY_DB_USER", self.db_cfg.user),
            password=os.getenv("ENERGY_DB_PASSWORD", self.db_cfg.password),
        )


        try:
            # we use pandas to read the query
            query = f"""
                SELECT *
                FROM {table}
                WHERE "{time_col_db}" > %(start)s
                  AND "{time_col_db}" <= %(end)s
                ORDER BY "{time_col_db}"
            """
            exog_raw = pd.read_sql_query(
                query,
                conn,
                params={"start": start_ts_utc, "end": end_ts_utc},
            )
        finally:
            conn.close()

        if exog_raw.empty:
            raise EnergyException(
                f"Exogenous DB table '{table}' has no rows in window "
                f"({start_ts_utc}, {end_ts_utc}]."
            )

        if time_col_db not in exog_raw.columns:
            raise EnergyException(
                f"Exogenous DB table '{table}' missing time column '{time_col_db}'. "
                f"Columns: {list(exog_raw.columns)}"
            )

        needed_cols = dyn_cols + static_cols
        missing = [c for c in needed_cols if c not in exog_raw.columns]
        if missing:
            raise EnergyException(
                f"Exogenous DB table '{table}' is missing required columns {missing} "
                f"used during training. Columns present: {list(exog_raw.columns)}"
            )

        # normalize time col name
        exog_raw = exog_raw.rename(columns={time_col_db: time_col})

        # ds is TIMESTAMPTZ → tz-aware; convert to naive UTC (to match MLForecast)
        exog_raw[time_col] = pd.to_datetime(exog_raw[time_col]).dt.tz_convert(None)

        exog_df = (
            exog_raw[[time_col] + needed_cols]
            .drop_duplicates(subset=[time_col])
            .sort_values(time_col)
        )

        logger.info(
            "Loaded %d exogenous rows from DB table '%s' for (%s, %s].",
            len(exog_df),
            table,
            start_ts,
            end_ts,
        )
        return exog_df

    # ------------------------------------------------------------------
    # Helpers: write predictions to DB
    # ------------------------------------------------------------------
    def _write_predictions_to_db(
        self,
        preds_df: pd.DataFrame,
        id_col: str,
        time_col: str,
    ) -> None:
        pred_table = self.config.pred_table
        if not pred_table:
            logger.info("No pred_table configured; skipping DB write.")
            return

        host = self.db_cfg.host
        port = self.db_cfg.port
        dbname = self.db_cfg.db_name
        user = self.db_cfg.user
        password = self.db_cfg.password

        logger.info(
            "Writing %d predictions to DB '%s' table '%s' ...",
            len(preds_df),
            dbname,
            pred_table,
        )

        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )

        try:
            with conn.cursor() as cur:
                create_sql = f"""
                    CREATE TABLE IF NOT EXISTS {pred_table} (
                        id SERIAL PRIMARY KEY,
                        "{id_col}" TEXT NOT NULL,
                        "{time_col}" TIMESTAMPTZ NOT NULL,
                        "y_pred" DOUBLE PRECISION NOT NULL,
                        "forecast_origin_ts" TIMESTAMPTZ NOT NULL,
                        "horizon_index" INTEGER NOT NULL
                    );
                """
                cur.execute(create_sql)

                cols = [id_col, time_col, "y_pred", "forecast_origin_ts", "horizon_index"]
                col_list_sql = ", ".join(f'"{c}"' for c in cols)
                placeholders = ", ".join(["%s"] * len(cols))
                insert_sql = f"INSERT INTO {pred_table} ({col_list_sql}) VALUES ({placeholders});"

                df_insert = preds_df[cols].copy()
                df_insert[time_col] = pd.to_datetime(df_insert[time_col]).dt.tz_localize(
                    "UTC"
                )
                df_insert["forecast_origin_ts"] = pd.to_datetime(
                    df_insert["forecast_origin_ts"]
                ).dt.tz_localize("UTC")

                rows = list(df_insert.itertuples(index=False, name=None))
                execute_batch(cur, insert_sql, rows, page_size=1000)

            conn.commit()
            logger.info(
                "Inserted %d prediction rows into table '%s' on DB '%s'.",
                len(preds_df),
                pred_table,
                dbname,
            )
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def initiate_batch_prediction(self) -> BatchPredictionArtifact:
        logger.info("Starting BatchPrediction (generic, config-driven).")

        # 1) load model + mlforecast_config
        model, mlf_cfg = self._load_best_model_and_mlf_config()

        id_col = mlf_cfg["unique_id_col"]
        time_col = mlf_cfg["time_col"]
        target_col = mlf_cfg["target_col"]
        dyn_cols: List[str] = mlf_cfg.get("dynamic_features") or []
        static_cols: List[str] = mlf_cfg.get("static_features") or []
        freq = str(mlf_cfg.get("freq", "H"))
        h_trained = int(mlf_cfg.get("h", 24))

        # 2) load history from DB
        history_df = self._load_history_from_db(time_col=time_col)

        # normalize time column: TIMESTAMPTZ -> naive UTC
        history_df[time_col] = pd.to_datetime(history_df[time_col], utc=True).dt.tz_convert(
            None
        )

        required_cols = [id_col, time_col, target_col] + dyn_cols + static_cols
        missing = [c for c in required_cols if c not in history_df.columns]
        if missing:
            raise EnergyException(
                f"History from table '{self.db_cfg.table}' is missing columns: {missing}. "
                f"Columns present: {list(history_df.columns)}"
            )

        history_df = history_df[required_cols].copy()
        history_df = history_df.sort_values([id_col, time_col])
        series_ids = history_df[id_col].unique()
        last_ts = history_df[time_col].max()

        logger.info(
            "History covers %d series; last timestamp in history: %s",
            len(series_ids),
            last_ts,
        )

        # 3) determine horizon H (>= trained horizon)
        n_future_hours = int(self.config.n_future_hours)
        H = max(n_future_hours, h_trained)

        # map freq to pandas offset to compute end_ts
        try:
            offset = pd.tseries.frequencies.to_offset(freq)
        except Exception:
            offset = pd.tseries.frequencies.to_offset("H")

        end_ts = last_ts + H * offset
        logger.info(
            "Batch prediction horizon: H=%d steps (freq=%s) => (%s, %s]. "
            "Requested n_future_hours=%d.",
            H,
            freq,
            last_ts,
            end_ts,
            n_future_hours,
        )

        # 4) load future exogenous from configured source (db)
        exog_df = self._load_future_exog(
            mlf_cfg=mlf_cfg,
            series_ids=series_ids,
            start_ts=last_ts,
            end_ts=end_ts,
        )

        # 5) Use MLForecast in stateless mode:
        #    - current_df = latest history (for lags)
        #    - exog_df   = exogenous features for (last_ts, end_ts]
        current_df = history_df.copy()

        if exog_df is not None:
            # Ensure X_df has the required key columns ('unique_id', 'ds')
            if id_col not in exog_df.columns:
                logger.info(
                    "Exogenous df has no id_col '%s'. "
                    "Replicating rows for each series id.",
                    id_col,
                )

                # unique times in exog_df
                time_vals = exog_df[time_col].drop_duplicates().sort_values().values

                # build full (unique_id, ds) grid
                cross = pd.MultiIndex.from_product(
                    [series_ids, time_vals],
                    names=[id_col, time_col],
                ).to_frame(index=False)

                # merge exogenous features onto the grid
                exog_df = cross.merge(
                    exog_df,
                    on=[time_col],
                    how="left",
                )

            logger.info(
                "Calling model.predict with new_df (history) and X_df (future exogenous)."
            )
            preds = model.predict(
                h=H,
                new_df=current_df,
                X_df=exog_df,
            )
        else:
            logger.info(
                "Calling model.predict with new_df (history) and NO exogenous features."
            )
            preds = model.predict(
                h=H,
                new_df=current_df,
            )



        # prediction col = any non id/time column
        pred_cols = [c for c in preds.columns if c not in (id_col, time_col)]
        if not pred_cols:
            raise EnergyException(
                f"No prediction column found in preds. Got columns: {list(preds.columns)}"
            )

        pred_col = pred_cols[0]
        preds = preds.rename(columns={pred_col: "y_pred"}).copy()
        preds[time_col] = pd.to_datetime(preds[time_col])

        # 7) assign horizon_index per series and keep first n_future_hours
        preds = preds.sort_values([id_col, time_col])
        preds["horizon_index"] = preds.groupby(id_col).cumcount() + 1

        preds = preds[preds["horizon_index"] <= n_future_hours].copy()
        if preds.empty:
            raise EnergyException(
                "Predictions dataframe is empty after selecting the first "
                f"{n_future_hours} horizon steps. Check model.predict output."
            )

        forecast_origin_ts = last_ts
        preds["forecast_origin_ts"] = forecast_origin_ts


        # 8) save to CSV artifact
        preds_out_path = self.config.predictions_path
        preds.to_csv(preds_out_path, index=False)
        logger.info("Saved batch predictions to %s (n=%d).", preds_out_path, len(preds))

        # 9) optionally write to DB
        self._write_predictions_to_db(preds_df=preds, id_col=id_col, time_col=time_col)

        artifact = BatchPredictionArtifact(
            predictions_path=preds_out_path,
            n_predictions=len(preds),
        )

        logger.info(
            "BatchPrediction completed: n_predictions=%d, forecast_origin_ts=%s",
            artifact.n_predictions,
            forecast_origin_ts,
        )

        return artifact
