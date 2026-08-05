from prometheus_client import Counter, Gauge, Histogram

EMBEDDING_REQUESTS = Counter(
    "embedding_requests_total",
    "Total embedding requests received",
    ["request_type", "status"]
)

EMBEDDING_LATENCY = Histogram(
    "embedding_latency_seconds",
    "Latency of embedding generation in seconds",
    ["request_type"],
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

EMBEDDING_BATCH_SIZE = Histogram(
    "embedding_batch_size",
    "Distribution of batch sizes in batch requests",
    buckets=(1, 5, 10, 20, 50, 100, 200, 256)
)

MODEL_LOADED = Gauge(
    "model_loaded_status",
    "Gauge indicating whether model is loaded in memory (1 = loaded, 0 = unloaded)"
)

EMBEDDING_ERRORS = Counter(
    "embedding_errors_total",
    "Total errors occurred during processing",
    ["error_type"]
)
