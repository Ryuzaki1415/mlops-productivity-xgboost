import os

# Must be set before prometheus_client is imported
multiproc_dir = "/tmp/prometheus_multiproc"
os.makedirs(multiproc_dir, exist_ok=True)
os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", multiproc_dir)

from prometheus_client import Counter, Histogram, Gauge

PREDICTION_REQUESTS = Counter(
    "celery_prediction_requests_total",
    "Total prediction tasks received"
)

PREDICTION_LATENCY = Histogram(
    "celery_prediction_duration_seconds",
    "End-to-end prediction task duration",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

PREDICTION_FAILURES = Counter(
    "celery_prediction_failures_total",
    "Total failed prediction tasks"
)

ACTIVE_WORKERS = Gauge(
    "celery_active_workers",
    "Number of active Celery workers"
)