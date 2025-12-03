from energy_forecasting.logger import logger
from energy_forecasting.configuration.configuration import ConfigManager
from energy_forecasting.components.data_ingestion import DataIngestion
from energy_forecasting.components.data_transformation import DataTransformation
from energy_forecasting.components.model_trainer import ModelTrainer
from energy_forecasting.components.model_evaluation import ModelEvaluation


class TrainPipeline:
    """
    Orchestrates the full training pipeline.

    Steps:
        1. DataIngestion → read history from DB, write raw CSV snapshot
        2. DataTransformation → csv train/test 
        3. ModelTrainer → trains models, saves model + metrics
        4. ModelEvaluation → compares with best model, updates registry


    All knobs are loaded from config/config.yaml via ConfigManager.
    """

    def __init__(self) -> None:
        self.config_manager = ConfigManager()
        self.pipeline_config = self.config_manager.training_pipeline_config

    def run_pipeline(self) -> None:
        logger.info("=== TrainPipeline started ===")
        logger.info("Artifacts root: %s", self.pipeline_config.artifacts_root)

        # 1. Data Ingestion (DB-only)
        di_config = self.config_manager.get_data_ingestion_config()
        db_config = self.config_manager.get_database_config()


        di_component = DataIngestion(
            data_ingestion_config=di_config,
            database_config=db_config,
        )

        logger.info(
            "DataIngestion (DB-only): host=%s, db=%s, table=%s, hours_history=%d",
            db_config.host,
            db_config.db_name,
            db_config.table,
            db_config.hours_history,
        )


        ingestion_artifact = di_component.initiate_data_ingestion()
        logger.info(
            "DataIngestion completed: path=%s, rows=%d",
            ingestion_artifact.raw_data_path,
            ingestion_artifact.n_rows,
        )

        # 2. Data Transformation
        dt_config = self.config_manager.get_data_transformation_config()
        dt_component = DataTransformation(config=dt_config)
        transformation_artifact = dt_component.initiate_data_transformation(
            ingestion_artifact
        )
        logger.info(
            "DataTransformation completed: train_rows=%d, test_rows=%d",
            transformation_artifact.n_train_rows,
            transformation_artifact.n_test_rows,
        )


        # 3. Model Trainer
        mt_config = self.config_manager.get_model_trainer_config()
        mlflow_config = self.config_manager.get_mlflow_config()

        mt_component = ModelTrainer(
            config=mt_config,
            data_transformation_artifact=transformation_artifact,
            mlflow_config=mlflow_config,
        )

        trainer_artifact = mt_component.initiate_model_trainer()
        logger.info(
            "ModelTrainer completed: model=%s, metrics=%s, rmse=%.4f, mae=%.4f",
            trainer_artifact.model_path,
            trainer_artifact.metrics_path,
            trainer_artifact.rmse,
            trainer_artifact.mae,
        )

        # 4. Model Evaluation
        me_config = self.config_manager.get_model_evaluation_config()
        me_component = ModelEvaluation(
            config=me_config,
            model_trainer_artifact=trainer_artifact,
        )
        eval_artifact = me_component.initiate_model_evaluation()

        logger.info(
            "ModelEvaluation: accepted=%s, best_model=%s",
            eval_artifact.is_model_accepted,
            eval_artifact.best_model_path,
        )

        logger.info(
            "=== TrainPipeline finished (ingestion + transformation + training + evaluation) ==="
        )
