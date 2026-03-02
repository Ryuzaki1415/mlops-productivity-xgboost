import numpy as np
import pandas as pd

from api.celery_app import celery_app
import shap

from api.model_loader import get_pipeline
from api.feature_engineering import create_features
from api.schemas import SHAPContributor
from api.api_config import OLLAMA_BASE_URL, OLLAMA_MODEL

from utils.config import (
   NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, DERIVED_COLUMNS,
)
from api.llm_client import get_llm_insight_sync
from api.cache import cache_set_sync,make_cache_key

from api.schemas import PredictionResponse


from api.model_loader import load_model

_shap_explainer = None


# ── Helpers ────────────────────────────────────────────────────────────────────
def classify_score(score: float) -> str:
    if score < 40:
        return "Low"
    elif score < 70:
        return "Moderate"
    return "High"

def get_explainer():
    global _shap_explainer
    if _shap_explainer is None:
        pipeline = get_pipeline()
        model    = pipeline.named_steps["model"]
        _shap_explainer = shap.TreeExplainer(model)
    return _shap_explainer



def get_feature_names(pipeline) -> list[str]:
    """Recover ordered feature names from the ColumnTransformer."""
    num_features = NUMERICAL_COLUMNS + DERIVED_COLUMNS
    cat_encoder  = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_features = cat_encoder.get_feature_names_out(CATEGORICAL_COLUMNS).tolist()
    return num_features + cat_features


def compute_shap_top5(
    pipeline, X_processed_df: pd.DataFrame
) -> list[dict]:
    """
    Computes SHAP values for a single row and returns top 5 contributors
    sorted by absolute impact.
    """
    explainer   = get_explainer()
    shap_values = explainer.shap_values(X_processed_df)  # shape: (1, n_features)
    shap_row    = shap_values[0]

    feature_names = X_processed_df.columns.tolist()
    feature_vals  = X_processed_df.iloc[0].tolist()

    pairs = sorted(
        zip(feature_names, feature_vals, shap_row),
        key=lambda x: abs(x[2]),
        reverse=True
    )

    return [
        {"feature": f, "value": round(float(v), 3), "shap_value": round(float(s), 3)}
        for f, v, s in pairs[:5]
    ]



@celery_app.task(name="api.tasks.run_prediction")
def run_prediction(raw_dict: dict):
    print("Task Received")
    
    
    try:
        pipeline = get_pipeline()
    except RuntimeError:
        print("loaded Model")
        load_model()
        pipeline = get_pipeline()
    print("making cache")
    cache_key = make_cache_key(raw_dict)
    pipeline = get_pipeline()

    print("Initiated Feature engineering")
    df_input = pd.DataFrame([raw_dict])
    df_engineered = create_features(df_input)

    feature_cols = (
        NUMERICAL_COLUMNS +
        DERIVED_COLUMNS +
        CATEGORICAL_COLUMNS
    )

    df_features = df_engineered[feature_cols]

    preprocessor = pipeline.named_steps["preprocessor"]
    X_processed_array = preprocessor.transform(df_features)

    feature_names = get_feature_names(pipeline)
    X_processed_df = pd.DataFrame(X_processed_array, columns=feature_names)

    print("predicting score")
    model = pipeline.named_steps["model"]
    score = float(np.clip(model.predict(X_processed_df)[0], 0, 100))
    category = classify_score(score)
    print("evaluating SHAP")
    top_contributors = compute_shap_top5(pipeline, X_processed_df)

    print("Getting LLM insight")
    # LLM call (now safe — runs in worker)
    insight = get_llm_insight_sync(
        score,
        category,
        raw_dict,
        top_contributors,
        OLLAMA_BASE_URL,
        OLLAMA_MODEL
    )
    
    
    response = PredictionResponse(
        score=round(score, 2),
        score_category=category,
        top_contributors=[SHAPContributor(**c) for c in top_contributors],
        insight=insight
    )

    # cache serialized version
    
    print("setting cache")
    cache_set_sync(cache_key, response.model_dump())

    return response.model_dump()