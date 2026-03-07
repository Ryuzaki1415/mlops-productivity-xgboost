import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

_pipeline = None

MODEL_PATH = Path("api/artifacts/xgboost_pipeline_v5.pkl")  # adjust if filename differs

def load_model():
    global _pipeline

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    _pipeline = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model from {MODEL_PATH}")
    return _pipeline


def get_pipeline():
    if _pipeline is None:
        raise RuntimeError(
            "Model not loaded. Ensure load_model() is called before prediction."
        )
    return _pipeline