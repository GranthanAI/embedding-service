from typing import Optional
from app.core.config import settings
from app.core.logging import logger
from app.models.model_manager import ModelManager
from app.services.embedding_service import EmbeddingService


class Container:
    """
    Dependency Injection Container managing application singletons.
    """

    def __init__(self):
        self.model_manager: Optional[ModelManager] = None
        self.embedding_service: Optional[EmbeddingService] = None

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
            normalize=settings.NORMALIZE_EMBEDDINGS
        )
        self.model_manager.load_model()

        # 2. Initialize Service Layer
        self.embedding_service = EmbeddingService(
            model_manager=self.model_manager,
            max_text_length=settings.MAX_TEXT_LENGTH,
            max_batch_size=settings.MAX_BATCH_SIZE
        )

        logger.info("DI Container resources initialized successfully.")

    async def shutdown_resources(self) -> None:
        """
        Graceful cleanup.
        """
        logger.info("Shutting down DI Container resources...")
        if self.model_manager:
            self.model_manager.unload()
        logger.info("DI Container teardown complete.")
