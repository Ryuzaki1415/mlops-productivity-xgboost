import joblib
import logging
from pathlib import Path
import mlflow.pyfunc

from utils.config import MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)

_pipeline = None

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# def load_model(model_path: str):
#     global _pipeline
#     path = Path(model_path)

#     if not path.exists():
#         raise FileNotFoundError(
#             f"Model file not found at: {model_path}\n"
#             f"Make sure your .pkl file is at artifacts/model.pkl"
#         )

#     _pipeline = joblib.load(path)

#     logger.info(f"Model loaded successfully from {model_path}")
#     return _pipeline
MODEL_NAME = "productivity_model"
MODEL_VERSION = "1"  # replace MODEL_STAGE with this

def load_model():
    global _pipeline
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    _pipeline = mlflow.sklearn.load_model(model_uri)  # returns raw sklearn pipeline
    logger.info(f"Loaded model from MLflow: {model_uri}")
    return _pipeline

def get_pipeline():
    if _pipeline is None:
        raise RuntimeError(
            "Model not loaded. Ensure load_model() is called during app startup."
        )
    return _pipeline