import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.api_config import  OLLAMA_BASE_URL, OLLAMA_MODEL
from api.feature_engineering import create_features
from api.model_loader import load_model, get_pipeline
from api.schemas import PredictionRequest, PredictionResponse,HealthResponse
from api.llm_client import  check_ollama_health
from api.cache import make_cache_key, cache_get_sync
from celery.result import AsyncResult
from api.celery_app import celery_app



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Productivity Predictor API",
    description="Predicts Work Productivity Score from lifestyle and screen-time features.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def greet():
    return {"message": "Please Navigate to the Streamlit Frontend!"}

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health_check():
    try:
        get_pipeline()
        model_loaded = True
    except RuntimeError:
        model_loaded = False
    ollama_ready = await check_ollama_health(OLLAMA_BASE_URL, OLLAMA_MODEL)
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        ollama_ready=ollama_ready,
    )

@app.post("/predict")
async def predict(request: PredictionRequest):
    raw_dict = request.model_dump()
    cache_key = make_cache_key(raw_dict)
    cached = cache_get_sync(cache_key)
    if cached:
        logger.info("CACHE HIT !!")
        return PredictionResponse(**cached)

    logger.info("CACHE MISS !!")
    # ✅ call by name string — fastapi never imports tasks.py
    task = celery_app.send_task("api.tasks.run_prediction", args=[raw_dict])

    logger.info(f"TASK SENT: {task.id}")
    return {"task_id": task.id}

@app.get("/result/{task_id}", response_model=PredictionResponse)
async def get_result(task_id: str):
    task = AsyncResult(task_id, app=celery_app)
    if task.state in ["PENDING", "STARTED"]:
        raise HTTPException(status_code=202, detail="Processing")
    if task.state == "FAILURE":
        raise HTTPException(status_code=500, detail=str(task.result))
    return PredictionResponse(**task.result)