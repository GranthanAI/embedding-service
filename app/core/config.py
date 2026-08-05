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
    VECTOR_DIMENSION: int = 384

    # ── Batch & Limits ───────────────────────────────────────────────────────
    BATCH_SIZE: int = 64
    MAX_BATCH_SIZE: int = 256
    MAX_TEXT_LENGTH: int = 8192

    # ── Server Config ────────────────────────────────────────────────────────
    REST_HOST: str = "0.0.0.0"
    REST_PORT: int = 8000
    GRPC_HOST: str = "0.0.0.0"
    GRPC_PORT: int = 50051
    GRPC_MAX_WORKERS: int = 10
    GRPC_MAX_MESSAGE_LENGTH: int = 4194304


settings = SystemSettings()
