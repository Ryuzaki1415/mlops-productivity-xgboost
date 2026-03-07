from prometheus_client import Counter, Histogram, Gauge




PREDICTION_REQUESTS = Counter(
    "celery_prediction_requests_total",
    "Total prediction tasks dispatched"
)

PREDICTION_LATENCY = Histogram(
    "celery_prediction_duration_seconds",
    "Time spent on prediction tasks",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

PREDICTION_FAILURES = Counter(
    "celery_prediction_failures_total",
    "Total failed prediction tasks"
)

ACTIVE_WORKERS = Gauge(
    "celery_active_workers",
    "Number of active Celery workers"
)