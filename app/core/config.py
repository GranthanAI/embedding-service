from pydantic_settings import BaseSettings, SettingsConfigDict


class SystemSettings(BaseSettings):
    """
    Validates and stores all embedding service configurations.
    Loads environment variables from .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "graphgpt-embedding-service"
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # ── Model Config ─────────────────────────────────────────────────────────
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    MODEL_PATH: str = r"C:\Users\hp\Desktop\models\models\all-MiniLM-L6-v2"
    DEVICE: str = "cpu"
    NORMALIZE_EMBEDDINGS: bool = True
    # Embedding model selection. MODEL_NAME is a Hugging Face id (or a sentence-transformers name). MODEL_PATH is an optional LOCAL copy and is
    # used only when its last path component equals the last component of MODEL_NAME (otherwise it is ignored with a warning), so changing
    # MODEL_NAME can never silently keep loading the old weights.
    TOKENIZER_ID: str = ""             # Hugging Face tokenizer id reported by GetModelInfo; empty = derived from MODEL_NAME
    MAX_SEQ_LENGTH: int = 0            # 0 = the model's own limit; > 0 caps it (smaller = faster, less text seen per chunk)
    # Instruction prefixes some models require (e5: "query: " / "passage: "; bge: a query instruction). QUERY_PREFIX is added to the single
    # GenerateEmbedding RPC, PASSAGE_PREFIX to the batch GenerateEmbeddings RPC. Empty = text is embedded as sent (all-MiniLM-L6-v2).
    QUERY_PREFIX: str = ""
    PASSAGE_PREFIX: str = ""
    VECTOR_DIMENSION: int = 384

    # ── Batch & Limits ───────────────────────────────────────────────────────
    BATCH_SIZE: int = 64
    MAX_BATCH_SIZE: int = 256
    MAX_TEXT_LENGTH: int = 8192

    # ── Priority lanes (see docs/performance.md) ─────────────────────────────
    # Default off: preserves the original single-lock behaviour. Recommended on for production.
    PRIORITY_LANES_ENABLED: bool = False
    BULK_SLICE_SIZE: int = 16          # texts per model call inside a bulk batch
    BULK_MAX_PENDING: int = 4          # bulk batches admitted at once (running + waiting); more -> RESOURCE_EXHAUSTED

    # ── Query embedding cache ────────────────────────────────────────────────
    QUERY_CACHE_SIZE: int = 0          # in-process LRU entries, 0 disables
    QUERY_CACHE_REDIS_URL: str = ""    # empty = no Redis layer (e.g. redis://localhost:6379/0)
    QUERY_CACHE_REDIS_TTL_SECONDS: int = 3600

    # ── Observability ────────────────────────────────────────────────────────
    TRUNCATION_METRIC_ENABLED: bool = True  # counts inputs over max_seq_length; vectors unchanged

    # ── Reranker (Rerank RPC) ────────────────────────────────────────────────
    # Enabled by default but lazily loaded on first call: no memory cost until used.
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"   # the default reranker (used when a request names none)
    # Comma-separated Hugging Face ids a request may select with RerankRequest.model, in addition to RERANKER_MODEL. A model that is not
    # listed is refused (INVALID_ARGUMENT): callers can never make this service download or load an arbitrary model.
    RERANKER_ALLOWED_MODELS: str = ""
    RERANKER_MAX_LOADED: int = 1       # rerankers kept in memory at once; selecting another evicts the least recently used one
    RERANKER_PRELOAD: bool = False
    RERANKER_MAX_PASSAGES: int = 50
    RERANKER_MAX_LENGTH: int = 512
    RERANKER_BATCH_SIZE: int = 16

    # ── Server Config ────────────────────────────────────────────────────────
    REST_HOST: str = "0.0.0.0"
    REST_PORT: int = 8005
    GRPC_HOST: str = "0.0.0.0"
    GRPC_PORT: int = 50054
    GRPC_MAX_WORKERS: int = 10
    GRPC_MAX_MESSAGE_LENGTH: int = 4194304


settings = SystemSettings()
