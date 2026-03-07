import os

# Must be set BEFORE prometheus_client is imported anywhere in the process
_multiproc_dir = "/tmp/prometheus_multiproc"
os.makedirs(_multiproc_dir, exist_ok=True)
os.environ["PROMETHEUS_MULTIPROC_DIR"] = _multiproc_dir

from celery import Celery
from celery.signals import worker_process_init, worker_ready
import threading
from api.model_loader import load_model

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

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
    """Runs in each forked worker process on startup."""
    # Set up multiproc dir so each fork writes metrics to shared folder
    multiproc_dir = "/tmp/prometheus_multiproc"
    os.makedirs(multiproc_dir, exist_ok=True)
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = multiproc_dir

    load_model()


@worker_ready.connect
def start_metrics_server(**kwargs):
    """Runs once in the main process when the worker is ready."""
    from prometheus_client import start_http_server, CollectorRegistry, multiprocess

    multiproc_dir = "/tmp/prometheus_multiproc"
    os.makedirs(multiproc_dir, exist_ok=True)
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = multiproc_dir

    # Use a multiprocess-aware registry so it aggregates all forked workers
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)

    threading.Thread(
        target=start_http_server,
        args=(8001,),
        kwargs={"registry": registry},
        daemon=True
    ).start()