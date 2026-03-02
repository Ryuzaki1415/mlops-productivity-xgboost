from celery import Celery

celery_app = Celery(
    "worker",
    broker="redis://localhost:6379/0",       
    backend="redis://localhost:6379/0"
)
celery_app.conf.task_default_queue = "predictions"

celery_app.conf.task_routes = {
    "tasks.run_prediction": {"queue": "predictions"}
}

celery_app.autodiscover_tasks(["api"])