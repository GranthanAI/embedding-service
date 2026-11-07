import threading
import time
from typing import List

from app.core.exceptions import ModelLoadException, ValidationException
from app.core.logging import logger
from app.core.metrics import (
    RERANK_ERRORS,
    RERANK_LATENCY,
    RERANK_MODEL_LOADED,
    RERANK_PASSAGES,
    RERANK_REQUESTS,
)


class Reranker:
    """
    Cross-encoder reranker. The model is loaded lazily on the first call
    (or at startup when preload is requested) so deployments that never call
    Rerank pay no memory. Texts are never logged.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        max_passages: int = 50,
        max_length: int = 512,
        batch_size: int = 16,
        model_factory=None,
    ):
        self.model_name = model_name
        self.device = device
        self.max_passages = max_passages
        self.max_length = max_length
        self.batch_size = batch_size
        self._factory = model_factory
        self._model = None
        self._lock = threading.Lock()  # guards loading
        self._predict_lock = threading.Lock()  # the model object is not safely concurrent

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                factory = self._factory
                if factory is None:
                    from sentence_transformers import CrossEncoder as factory
                start = time.perf_counter()
                self._model = factory(self.model_name, device=self.device, max_length=self.max_length)
                RERANK_MODEL_LOADED.set(1)
                logger.info(f"Loaded reranker '{self.model_name}' in {time.perf_counter() - start:.1f}s")
            except Exception as e:
                RERANK_MODEL_LOADED.set(0)
                logger.error(f"Failed to load reranker: {e}", exc_info=True)
                raise ModelLoadException(f"Failed to load reranker '{self.model_name}': {e}")

    def unload(self) -> None:
        """Frees the model (waits for an in-flight predict). A later rerank() lazily reloads it."""
        with self._predict_lock:
            with self._lock:
                self._model = None
                RERANK_MODEL_LOADED.set(0)

    def validate(self, query: str, passages: List[str]) -> None:
        if query is None or not query.strip():
            RERANK_ERRORS.labels(error_type="EmptyQuery").inc()
            raise ValidationException("Rerank 'query' cannot be empty.")
        if not passages:
            RERANK_ERRORS.labels(error_type="NoPassages").inc()
            raise ValidationException("Rerank 'passages' must contain at least one passage.")
        if len(passages) > self.max_passages:
            RERANK_ERRORS.labels(error_type="TooManyPassages").inc()
            raise ValidationException(
                f"Rerank got {len(passages)} passages, maximum is {self.max_passages}."
            )

    def rerank(self, query: str, passages: List[str]) -> List[float]:
        """One relevance score per passage, in input order."""
        start = time.perf_counter()
        RERANK_REQUESTS.labels(status="attempt").inc()
        try:
            self.validate(query, passages)
            if self._model is None:
                self.load()
            RERANK_PASSAGES.observe(len(passages))
            pairs = [(query, p) for p in passages]
            with self._predict_lock:
                scores = self._model.predict(
                    pairs, batch_size=self.batch_size, show_progress_bar=False
                )
            out = [float(s) for s in scores]
            RERANK_LATENCY.observe(time.perf_counter() - start)
            RERANK_REQUESTS.labels(status="success").inc()
            return out
        except Exception:
            RERANK_REQUESTS.labels(status="failure").inc()
            raise
