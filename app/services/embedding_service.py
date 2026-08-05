import time
from typing import List

from app.core.exceptions import (
    BatchSizeExceededException,
    EmptyTextException,
    TextTooLongException,
    ValidationException,
)
from app.core.logging import logger
from app.core.metrics import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_ERRORS,
    EMBEDDING_LATENCY,
    EMBEDDING_REQUESTS,
)
from app.models.model_manager import ModelManager
from app.schemas.embedding import (
    BatchEmbeddingResponse,
    EmbeddingResponse,
)


class EmbeddingService:
    """
    Business service layer responsible for validating incoming requests,
    invoking the ModelManager singleton, and gathering latency metrics.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        max_text_length: int = 8192,
        max_batch_size: int = 256
    ):
        self.model_manager = model_manager
        self.max_text_length = max_text_length
        self.max_batch_size = max_batch_size

    def validate_text(self, text: str) -> str:
        if text is None:
            EMBEDDING_ERRORS.labels(error_type="EmptyTextException").inc()
            raise EmptyTextException("Text field cannot be null.")
        
        stripped = text.strip()
        if not stripped:
            EMBEDDING_ERRORS.labels(error_type="EmptyTextException").inc()
            raise EmptyTextException("Text cannot be empty or whitespace-only.")

        if len(stripped) > self.max_text_length:
            EMBEDDING_ERRORS.labels(error_type="TextTooLongException").inc()
            raise TextTooLongException(self.max_text_length, len(stripped))

        return stripped

    def generate_single(self, text: str) -> EmbeddingResponse:
        start_time = time.perf_counter()
        EMBEDDING_REQUESTS.labels(request_type="single", status="attempt").inc()

        try:
            valid_text = self.validate_text(text)
            vector = self.model_manager.encode_single(valid_text)
            
            duration = time.perf_counter() - start_time
            EMBEDDING_LATENCY.labels(request_type="single").observe(duration)
            EMBEDDING_REQUESTS.labels(request_type="single", status="success").inc()

            return EmbeddingResponse(
                model=self.model_manager.model_name,
                dimension=self.model_manager.get_dimension(),
                embedding=vector
            )
        except Exception as e:
            EMBEDDING_REQUESTS.labels(request_type="single", status="failure").inc()
            raise e

    def generate_batch(self, texts: List[str]) -> BatchEmbeddingResponse:
        start_time = time.perf_counter()
        EMBEDDING_REQUESTS.labels(request_type="batch", status="attempt").inc()

        try:
            if not texts:
                EMBEDDING_ERRORS.labels(error_type="EmptyBatchException").inc()
                raise ValidationException("Batch 'texts' list cannot be empty.")

            if len(texts) > self.max_batch_size:
                EMBEDDING_ERRORS.labels(error_type="BatchSizeExceededException").inc()
                raise BatchSizeExceededException(self.max_batch_size, len(texts))

            EMBEDDING_BATCH_SIZE.observe(len(texts))

            validated_texts = [self.validate_text(t) for t in texts]
            vectors = self.model_manager.encode_batch(validated_texts)

            duration = time.perf_counter() - start_time
            EMBEDDING_LATENCY.labels(request_type="batch").observe(duration)
            EMBEDDING_REQUESTS.labels(request_type="batch", status="success").inc()

            return BatchEmbeddingResponse(
                model=self.model_manager.model_name,
                dimension=self.model_manager.get_dimension(),
                embeddings=vectors
            )
        except Exception as e:
            EMBEDDING_REQUESTS.labels(request_type="batch", status="failure").inc()
            raise e
