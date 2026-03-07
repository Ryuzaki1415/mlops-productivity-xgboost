import os
from celery import Celery
from celery.signals import worker_process_init

from api.model_loader import load_model

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.task_default_queue = "predictions"
celery_app.conf.task_routes = {
    "tasks.run_prediction": {"queue": "predictions"}
}

celery_app.autodiscover_tasks(["api"])


@worker_process_init.connect
def init_worker(**kwargs):
    load_model()
