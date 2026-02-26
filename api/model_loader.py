import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_pipeline = None


def load_model(model_path: str):
    global _pipeline
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            f"Make sure your .pkl file is at artifacts/model.pkl"
        )

    _pipeline = joblib.load(path)

    logger.info(f"Model loaded successfully from {model_path}")
    return _pipeline


def get_pipeline():
    if _pipeline is None:
        raise RuntimeError(
            "Model not loaded. Ensure load_model() is called during app startup."
        )
    return _pipeline