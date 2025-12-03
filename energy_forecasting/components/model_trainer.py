from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from energy_forecasting.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from energy_forecasting.exception import EnergyException
from energy_forecasting.logger import logger

import mlflow
from energy_forecasting.entity.config_entity import ModelTrainerConfig, MLflowConfig



def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE explicitly to avoid sklearn version differences."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse))


class ModelTrainer:
    """
    Train time-series forecasting models with MLForecast using ONLY TRAIN data.

    - Inputs (from DataTransformationArtifact):
        train_array_path -> train.csv
        (must contain: unique_id, ds, y, dynamic features, static features)

    - Configuration (from ModelTrainerConfig, YAML-driven):
        - mlf_* fields: columns, freq, lags, differences, lag_transforms, date_features
        - mlf_models: models + params_list (list of hyperparameter dicts)

    - Procedure:
        1) Load train.csv
        2) For each model name & each param dict from config:
             - Build MLForecast with ONE model
             - Run cross_validation(df_train, n_windows, h)
             - Compute RMSE/MAE on CV
        3) Choose best candidate (lowest RMSE)
        4) Refit MLForecast with the best model on FULL train
        5) Save final MLForecast + metrics.json
        6) Log params + metrics to MLflow
    """

    def __init__(
        self,
        config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
        mlflow_config: MLflowConfig | None = None,
    ) -> None:
        self.config = config
        self.data_transformation_artifact = data_transformation_artifact
        self.mlflow_config = mlflow_config

    # ------------------------------------------------------------------
    # Helpers to build objects from config
    # ------------------------------------------------------------------
    def _build_lag_transforms(self) -> Dict[int, List[Any]]:
        """
        Build lag_transforms dict for MLForecast from config.mlf_lag_transforms.

        Expected YAML shape:

        lag_transforms:
          "1":
            - type: ExpandingMean
          "24":
            - type: RollingMean
              window_size: 48
        """
        from mlforecast.lag_transforms import ExpandingMean, RollingMean

        cfg = self.config
        lt_cfg = cfg.mlf_lag_transforms or {}
        if not isinstance(lt_cfg, dict):
            raise EnergyException("mlf_lag_transforms must be a mapping of lag -> list[...]")

        result: Dict[int, List[Any]] = {}
        for lag_str, items in lt_cfg.items():
            try:
                lag = int(lag_str)
            except Exception as e:
                raise EnergyException(f"Invalid lag key '{lag_str}' in mlf_lag_transforms") from e

            if not isinstance(items, list):
                raise EnergyException(f"mlf_lag_transforms['{lag_str}'] must be a list")

            transforms: List[Any] = []
            for item in items:
                if not isinstance(item, dict) or "type" not in item:
                    raise EnergyException(
                        f"Each entry under lag {lag_str} must be a dict with at least 'type'."
                    )
                ttype = str(item["type"]).lower()

                if ttype == "expandingmean":
                    transforms.append(ExpandingMean())
                elif ttype == "rollingmean":
                    window_size = int(item.get("window_size", 24))
                    transforms.append(RollingMean(window_size=window_size))
                else:
                    raise EnergyException(
                        f"Unsupported lag transform type '{item['type']}' for lag {lag_str}."
                    )

            if transforms:
                result[lag] = transforms

        return result

    def _iter_candidates_from_config(self) -> List[Tuple[str, str, Dict[str, Any], Any]]:
        """
        Read models + params_list from config.mlf_models and build candidates.

        YAML example:

          models:
            lgbm:
              type: lightgbm
              params_list:
                - num_leaves: 128
                  learning_rate: 0.05
                  n_estimators: 200
                  verbosity: -1
                - num_leaves: 512
                  learning_rate: 0.05
                  n_estimators: 200
                  verbosity: -1

            lin_reg:
              type: linear
              params_list:
                - fit_intercept: true
                - fit_intercept: false
        """
        import lightgbm as lgb
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

        MODEL_REGISTRY = {
            "lightgbm": lgb.LGBMRegressor,
            "linear": LinearRegression,
            "rf": RandomForestRegressor,
            "gbr": GradientBoostingRegressor,
        }

        cfg = self.config
        models_cfg = cfg.mlf_models or {}
        if not isinstance(models_cfg, dict) or not models_cfg:
            raise EnergyException("mlf_models must be a non-empty mapping in config.")

        candidates: List[Tuple[str, str, Dict[str, Any], Any]] = []

        for model_name, mcfg in models_cfg.items():
            if not isinstance(mcfg, dict):
                raise EnergyException(f"Config for model '{model_name}' must be a dict.")
            model_type = str(mcfg.get("type", "")).lower()
            if model_type not in MODEL_REGISTRY:
                raise EnergyException(
                    f"Unknown model type '{model_type}' for model '{model_name}'. "
                    f"Supported: {list(MODEL_REGISTRY.keys())}"
                )

            param_list = mcfg.get("params_list", [])
            if not isinstance(param_list, list) or not param_list:
                raise EnergyException(
                    f"Model '{model_name}' must have 'params_list' as a non-empty list."
                )

            model_cls = MODEL_REGISTRY[model_type]
            for idx, params in enumerate(param_list):
                if not isinstance(params, dict):
                    raise EnergyException(
                        f"Each item in params_list for model '{model_name}' must be a dict."
                    )
                alias = f"{model_name}_{idx}"
                est = model_cls(**params)
                candidates.append((alias, model_type, params, est))

        return candidates

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logger.info("Starting ModelTrainer (MLForecast + CV on train only).")

        try:
            # Lazy imports for MLForecast
            from mlforecast import MLForecast
            from mlforecast.target_transforms import Differences

            cfg = self.config

            # ------------------------------------------------------------------
            # 1) Load TRAIN CSV
            # ------------------------------------------------------------------
            train_path: Path = self.data_transformation_artifact.train_array_path
            logger.info("Loading training CSV for MLForecast from %s", train_path)

            if not train_path.exists():
                raise EnergyException(
                    f"Train CSV not found at {train_path}. Did DataTransformation run?"
                )

            train_df = pd.read_csv(train_path)

            id_col = cfg.mlf_unique_id_col
            time_col = cfg.mlf_time_col
            target_col = cfg.mlf_target_col
            dyn_cols = list(cfg.mlf_dynamic_features or [])
            static_cols = list(cfg.mlf_static_features or [])

            # Time column must be proper datetime64[ns]
            if time_col in train_df.columns:
                train_df[time_col] = pd.to_datetime(
                    train_df[time_col],
                    utc=True,
                    errors="coerce",
                )
                na_mask = train_df[time_col].isna()
                if na_mask.any():
                    n_bad = int(na_mask.sum())
                    logger.warning(
                        "Dropping %d rows with invalid '%s' timestamps during ModelTrainer.",
                        n_bad,
                        time_col,
                    )
                    train_df = train_df.loc[~na_mask].copy()
                # remove timezone (MLForecast wants naive timestamps)
                train_df[time_col] = train_df[time_col].dt.tz_convert(None)
            else:
                raise EnergyException(f"Time column '{time_col}' not found in training data.")

            required_cols = [id_col, time_col, target_col] + dyn_cols + static_cols
            missing_cols = [c for c in required_cols if c not in train_df.columns]
            if missing_cols:
                raise EnergyException(
                    f"MLForecast training: missing columns in train_df: {missing_cols}"
                )

            df_train = train_df[required_cols].copy()
            logger.info("Train df shape: %s", df_train.shape)
            logger.info(
                "MLForecast columns -> id_col=%s, time_col=%s, target_col=%s, dyn=%s, static=%s",
                id_col,
                time_col,
                target_col,
                dyn_cols,
                static_cols,
            )

            # ------------------------------------------------------------------
            # 2) Build config-driven objects
            # ------------------------------------------------------------------
            lags = cfg.mlf_lags or [1, 24, 168]
            diffs = cfg.mlf_differences or [24]
            date_features = list(cfg.mlf_date_features or [])
            lag_transforms = self._build_lag_transforms()
            candidates = self._iter_candidates_from_config()

            logger.info(
                "MLForecast settings: freq=%s, lags=%s, differences=%s, n_windows=%d, h=%d",
                cfg.mlf_freq,
                lags,
                diffs,
                cfg.mlf_n_windows,
                cfg.mlf_h,
            )
            logger.info("Number of candidate model+param combos: %d", len(candidates))

            # ------------------------------------------------------------------
            # 3) Cross-validation PER CANDIDATE (memory friendly)
            # ------------------------------------------------------------------
            best_alias: str | None = None
            best_model_type: str | None = None
            best_params: Dict[str, Any] | None = None
            best_rmse: float = float("inf")
            best_mae: float = float("inf")

            all_metrics: Dict[str, Dict[str, float]] = {}

            for alias, model_type, params, est in candidates:
                logger.info("=== CV for candidate alias=%s, type=%s, params=%s", alias, model_type, params)

                fcst = MLForecast(
                    models={alias: est},
                    target_transforms=[Differences(diffs)] if diffs else [],
                    freq=cfg.mlf_freq,
                    lags=lags,
                    lag_transforms=lag_transforms,
                    date_features=date_features,
                )

                cv_df = fcst.cross_validation(
                    df=df_train,
                    n_windows=cfg.mlf_n_windows,
                    h=cfg.mlf_h,
                    static_features=static_cols,
                )

                # For a single model, MLForecast returns wide format:
                # columns: [id_col, time_col, 'cutoff', target_col, alias]
                expected_cols = {id_col, time_col, "cutoff", target_col, alias}
                missing_cv = expected_cols - set(cv_df.columns)
                if missing_cv:
                    raise EnergyException(
                        f"MLForecast.cross_validation for alias='{alias}' "
                        f"returned df missing columns {missing_cv}. "
                        f"Got columns: {list(cv_df.columns)}"
                    )

                # Ground truth and predictions
                y_true = cv_df[target_col].values
                y_pred = cv_df[alias].values

                rmse_val = _rmse(y_true, y_pred)
                mae_val = float(mean_absolute_error(y_true, y_pred))


                all_metrics[alias] = {"rmse": rmse_val, "mae": mae_val}
                logger.info(
                    "CV metrics for alias=%s -> RMSE=%.4f, MAE=%.4f",
                    alias,
                    rmse_val,
                    mae_val,
                )

                # track best
                if rmse_val < best_rmse:
                    best_rmse = rmse_val
                    best_mae = mae_val
                    best_alias = alias
                    best_model_type = model_type
                    best_params = params

                # free memory from this CV result
                del cv_df

            if best_alias is None or best_params is None or best_model_type is None:
                raise EnergyException("No valid candidate found during CV.")

            logger.info(
                "Best candidate: alias=%s, type=%s, RMSE=%.4f, MAE=%.4f, params=%s",
                best_alias,
                best_model_type,
                best_rmse,
                best_mae,
                best_params,
            )

            # ------------------------------------------------------------------
            # 4) Refit final MLForecast with ONLY the best model on FULL TRAIN
            # ------------------------------------------------------------------
            # Rebuild the estimator cleanly from best_model_type/best_params
            import lightgbm as lgb
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

            MODEL_REGISTRY = {
                "lightgbm": lgb.LGBMRegressor,
                "linear": LinearRegression,
                "rf": RandomForestRegressor,
                "gbr": GradientBoostingRegressor,
            }

            best_cls = MODEL_REGISTRY[best_model_type]
            best_estimator = best_cls(**best_params)

            fcst_final = MLForecast(
                models={"best": best_estimator},
                target_transforms=[Differences(diffs)] if diffs else [],
                freq=cfg.mlf_freq,
                lags=lags,
                lag_transforms=lag_transforms,
                date_features=date_features,
            )

            fcst_final.fit(
                df_train[[id_col, time_col, target_col] + dyn_cols + static_cols],
                static_features=static_cols,
            )
            logger.info("Final MLForecast model fitted with best candidate only.")

            # ------------------------------------------------------------------
            # 5) Persist model + metrics locally
            # ------------------------------------------------------------------
            self.config.model_dir.mkdir(parents=True, exist_ok=True)
            self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)

            import joblib

            joblib.dump(fcst_final, self.config.model_path)
            logger.info("Final MLForecast object saved to %s", self.config.model_path)

            metrics = {
                "rmse": best_rmse,
                "mae": best_mae,
                "model_name": "mlforecast",
                "best_alias": best_alias,
                "best_model_type": best_model_type,
                "best_params": best_params,
                "all_candidates_cv_metrics": all_metrics,
                "mlforecast_config": {
                    "freq": cfg.mlf_freq,
                    "lags": cfg.mlf_lags,
                    "differences": cfg.mlf_differences,
                    "unique_id_col": cfg.mlf_unique_id_col,
                    "time_col": cfg.mlf_time_col,
                    "target_col": cfg.mlf_target_col,
                    "dynamic_features": cfg.mlf_dynamic_features,
                    "static_features": cfg.mlf_static_features,
                    "n_windows": cfg.mlf_n_windows,
                    "h": cfg.mlf_h,
                    "lag_transforms": cfg.mlf_lag_transforms,
                    "date_features": cfg.mlf_date_features,
                },
            }

            self.config.metrics_path.write_text(
                json.dumps(metrics, indent=2),
                encoding="utf-8",
            )
            logger.info(
                "Metrics saved to %s with content: %s",
                self.config.metrics_path,
                metrics,
            )

            # ------------------------------------------------------------------
            # 6) Log to MLflow (if configured)
            # ------------------------------------------------------------------
            if self.mlflow_config is not None:
                try:
                    logger.info(
                        "Using MLflow tracking URI: %s",
                        self.mlflow_config.tracking_uri,
                    )
                    mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
                    mlflow.set_experiment(self.mlflow_config.experiment_name)

                    with mlflow.start_run(run_name=f"train_mlforecast_{best_model_type}"):

                        # params
                        mlflow.log_param("trainer_type", "mlforecast")
                        mlflow.log_param("best_alias", best_alias)
                        mlflow.log_param("best_model_type", best_model_type)
                        for k, v in best_params.items():
                            mlflow.log_param(f"best_param_{k}", v)

                        mlflow.log_param("freq", cfg.mlf_freq)
                        mlflow.log_param("lags", str(cfg.mlf_lags))
                        mlflow.log_param("differences", str(cfg.mlf_differences))
                        mlflow.log_param("unique_id_col", cfg.mlf_unique_id_col)
                        mlflow.log_param("time_col", cfg.mlf_time_col)
                        mlflow.log_param("target_col", cfg.mlf_target_col)
                        mlflow.log_param(
                            "dynamic_features",
                            ",".join(cfg.mlf_dynamic_features or []),
                        )
                        mlflow.log_param(
                            "static_features",
                            ",".join(cfg.mlf_static_features or []),
                        )
                        mlflow.log_param("n_windows", cfg.mlf_n_windows)
                        mlflow.log_param("h", cfg.mlf_h)
                        mlflow.log_param("date_features", ",".join(cfg.mlf_date_features or []))

                        # metrics (CV on train)
                        mlflow.log_metric("rmse_cv", best_rmse)
                        mlflow.log_metric("mae_cv", best_mae)

                        # artifacts
                        mlflow.log_artifact(
                            str(self.config.metrics_path), artifact_path="metrics"
                        )
                        mlflow.log_artifact(
                            str(self.config.model_path), artifact_path="model"
                        )

                    logger.info(
                        "Logged MLForecast training run to MLflow "
                        "(experiment=%s, best_model_type=%s)",
                        self.mlflow_config.experiment_name,
                        best_model_type,
                    )
                except Exception as e:
                    logger.warning("MLflow logging failed: %s", e)

            # ------------------------------------------------------------------
            # 7) Return artifact (ModelEvaluation uses rmse / mae)
            # ------------------------------------------------------------------
            artifact = ModelTrainerArtifact(
                model_path=self.config.model_path,
                metrics_path=self.config.metrics_path,
                rmse=best_rmse,
                mae=best_mae,
                test_data_path=self.data_transformation_artifact.test_array_path,
            )

            logger.info(
                "ModelTrainer completed: model=%s, metrics=%s, rmse=%.4f, mae=%.4f",
                artifact.model_path,
                artifact.metrics_path,
                artifact.rmse,
                artifact.mae,
            )
            return artifact

        except Exception as e:
            raise EnergyException(e) from e
