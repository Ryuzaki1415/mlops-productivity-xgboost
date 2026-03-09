import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from api.model_loader import get_pipeline
from api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from api.cache import make_cache_key, cache_get_sync
from celery.result import AsyncResult
from api.celery_app import celery_app
from api.cache import redis_client_sync
from api.llm_client import check_groq_health
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Productivity Predictor API",
    description="Predicts Work Productivity Score from lifestyle and screen-time features.",
    version="1.0.0",
)

Instrumentator().instrument(app).expose(app)

limiter = Limiter(key_func=get_remote_address, storage_uri="redis://redis:6379/0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def greet():
    return {"message": "Please Navigate to the Streamlit Frontend!"}


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
@limiter.limit("60/minute")
async def health_check(request: Request):
    # Check model loaded from MLflow registry
    try:
        get_pipeline()  
        model_loaded = True
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        model_loaded = False

    try:
        redis_client_sync.ping()
        redis_ready = True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        redis_ready = False

    groq_ready = await check_groq_health()

    if model_loaded and redis_ready and groq_ready:
        status = "ok"
    elif model_loaded and redis_ready:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        groq_ready=groq_ready,
        redis_ready=redis_ready,
    )


@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, payload: PredictionRequest):
    raw_dict = payload.model_dump()
    cache_key = make_cache_key(raw_dict)
    cached = cache_get_sync(cache_key)
    if cached:
        logger.info("CACHE HIT !!")
        return PredictionResponse(**cached)
    logger.info("CACHE MISS !!")
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