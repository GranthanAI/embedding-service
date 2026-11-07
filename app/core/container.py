from typing import Optional
from app.core.config import settings
from app.core.logging import logger
from app.models.model_manager import ModelManager
from app.models.reranker import Reranker
from app.models.reranker_pool import RerankerPool
from app.services.embedding_service import EmbeddingService
from app.services.query_cache import QueryCache, RedisLayer


class Container:
    """
    Dependency Injection Container managing application singletons.
    """

    def __init__(self):
        self.model_manager: Optional[ModelManager] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.reranker: Optional[RerankerPool] = None

    async def init_resources(self) -> None:
        """
        Initializes and wires singletons.
        """
        logger.info("Initializing DI Container resources...")

        # 1. Initialize ModelManager (loads model once)
        self.model_manager = ModelManager(
            model_name=settings.MODEL_NAME,
            model_path=settings.MODEL_PATH,
            device=settings.DEVICE,
            normalize=settings.NORMALIZE_EMBEDDINGS,
            priority_lanes=settings.PRIORITY_LANES_ENABLED,
            bulk_slice_size=settings.BULK_SLICE_SIZE,
            bulk_max_pending=settings.BULK_MAX_PENDING,
            truncation_metric=settings.TRUNCATION_METRIC_ENABLED,
            max_seq_length=settings.MAX_SEQ_LENGTH,
            tokenizer_id=settings.TOKENIZER_ID or None,
        )
        self.model_manager.load_model()

        # 2. Initialize Service Layer
        self.embedding_service = EmbeddingService(
            model_manager=self.model_manager,
            max_text_length=settings.MAX_TEXT_LENGTH,
            max_batch_size=settings.MAX_BATCH_SIZE,
            query_cache=QueryCache(
                size=settings.QUERY_CACHE_SIZE,
                redis_layer=RedisLayer(settings.QUERY_CACHE_REDIS_URL, settings.QUERY_CACHE_REDIS_TTL_SECONDS)
                if settings.QUERY_CACHE_REDIS_URL else None,
            ),
            query_prefix=settings.QUERY_PREFIX,
            passage_prefix=settings.PASSAGE_PREFIX,
        )

        # 3. Reranker (lazy unless RERANKER_PRELOAD)
        if settings.RERANKER_ENABLED:
            self.reranker = RerankerPool(
                default_model=settings.RERANKER_MODEL,
                allowed_models=settings.RERANKER_ALLOWED_MODELS.split(","),
                max_loaded=settings.RERANKER_MAX_LOADED,
                factory=lambda name: Reranker(
                    model_name=name,
                    device=settings.DEVICE,
                    max_passages=settings.RERANKER_MAX_PASSAGES,
                    max_length=settings.RERANKER_MAX_LENGTH,
                    batch_size=settings.RERANKER_BATCH_SIZE,
                ),
            )
            if settings.RERANKER_PRELOAD:
                self.reranker.load()

        logger.info("DI Container resources initialized successfully.")

    async def shutdown_resources(self) -> None:
        """
        Graceful cleanup.
        """
        logger.info("Shutting down DI Container resources...")
        if self.model_manager:
            self.model_manager.unload()
        logger.info("DI Container teardown complete.")
