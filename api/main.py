import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from utils.config import (
   NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, DERIVED_COLUMNS,
)


from api.api_config import  OLLAMA_BASE_URL, OLLAMA_MODEL,MODEL_PATH
from feature_engineering import create_features
from model_loader import load_model, get_pipeline
from schemas import PredictionRequest, PredictionResponse, SHAPContributor, HealthResponse
from llm_client import get_llm_insight, check_ollama_health

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── SHAP explainer cache (built once after model loads) ────────────────────────
_shap_explainer = None


def get_explainer():
    global _shap_explainer
    if _shap_explainer is None:
        pipeline = get_pipeline()
        model    = pipeline.named_steps["model"]
        _shap_explainer = shap.TreeExplainer(model)
    return _shap_explainer


# ── Lifespan: load model once at startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    load_model(MODEL_PATH)
    logger.info("Model ready. Building SHAP explainer...")
    get_explainer()   # warm up explainer at startup
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Productivity Predictor API",
    description="Predicts Work Productivity Score from lifestyle and screen-time features.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def classify_score(score: float) -> str:
    if score < 40:
        return "Low"
    elif score < 70:
        return "Moderate"
    return "High"


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


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health_check():
    try:
        get_pipeline()
        model_loaded = True
    except RuntimeError:
        model_loaded = False

    ollama_ready = await check_ollama_health(OLLAMA_BASE_URL)

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        ollama_ready=ollama_ready,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    pipeline = get_pipeline()

    # 1. Raw input → DataFrame
    raw_dict = request.model_dump()
    df_input = pd.DataFrame([raw_dict])

    # 2. Feature engineering
    df_engineered = create_features(df_input)

    # 3. Select only columns the pipeline expects
    feature_cols = NUMERICAL_COLUMNS + DERIVED_COLUMNS + CATEGORICAL_COLUMNS
    try:
        df_features = df_engineered[feature_cols]
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature column: {e}")

    # 4. Preprocess (transform only — pipeline was already fitted during training)
    preprocessor      = pipeline.named_steps["preprocessor"]
    X_processed_array = preprocessor.transform(df_features)

    feature_names    = get_feature_names(pipeline)
    X_processed_df   = pd.DataFrame(X_processed_array, columns=feature_names)

    # 5. Predict
    model = pipeline.named_steps["model"]
    score = float(np.clip(model.predict(X_processed_df)[0], 0, 100))
    category = classify_score(score)

    # 6. SHAP top 5
    top_contributors = compute_shap_top5(pipeline, X_processed_df)

    # 7. LLM insight (auto-triggered)
    insight = await get_llm_insight(
        score=score,
        category=category,
        raw_features=raw_dict,
        top_contributors=top_contributors,
        ollama_base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
    )

    return PredictionResponse(
        score=round(score, 2),
        score_category=category,
        top_contributors=[SHAPContributor(**c) for c in top_contributors],
        insight=insight,
    )
