import logging
import os
import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)

_pipeline = None

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "productivity_model"
MODEL_ALIAS = "production"

def load_model():
    global _pipeline
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    logger.info(f"Loading model from MLflow registry: {model_uri}")
    _pipeline = mlflow.sklearn.load_model(model_uri)
    logger.info("Model loaded successfully from MLflow registry")
    return _pipeline

def get_pipeline():
    if _pipeline is None:
        raise RuntimeError(
            "Model not loaded. Ensure load_model() is called before prediction."
        )
    return _pipeline