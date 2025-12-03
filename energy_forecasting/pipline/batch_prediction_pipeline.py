from energy_forecasting.logger import logger
from energy_forecasting.configuration.configuration import ConfigManager
from energy_forecasting.components.batch_prediction import BatchPrediction


class BatchPredictionPipeline:
    """
    Simple orchestration wrapper around the BatchPrediction component.

    Uses config/config.yaml for default n_future_hours,
    but we can override it via the constructor.

    Usage:
        BatchPredictionPipeline().run_pipeline()
        BatchPredictionPipeline(n_future_hours=24).run_pipeline()
    """

    def __init__(self, n_future_hours: int | None = None) -> None:
        self.config_manager = ConfigManager()
        self.pipeline_config = self.config_manager.training_pipeline_config
        self.batch_prediction_config = self.config_manager.get_batch_prediction_config(
            n_future_hours_override=n_future_hours
        )

    def run_pipeline(self):
        logger.info("=== BatchPredictionPipeline started ===")
        logger.info("Artifacts root: %s", self.pipeline_config.artifacts_root)

        batch_pred_component = BatchPrediction(config=self.batch_prediction_config)
        artifact = batch_pred_component.initiate_batch_prediction()

        logger.info(
            "BatchPredictionPipeline completed: predictions=%s, n=%d",
            artifact.predictions_path,
            artifact.n_predictions,
        )
        logger.info("=== BatchPredictionPipeline finished ===")

        return artifact
