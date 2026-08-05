import os
import threading
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.exceptions import ModelLoadException
from app.core.logging import logger
from app.core.metrics import MODEL_LOADED


class ModelManager:
    """
    Singleton manager for SentenceTransformer embedding model.
    Loads the model once during startup, keeps it memory-resident,
    and provides thread-safe single and batch vector encoding.
    """

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        device: str = "cpu",
        normalize: bool = True
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.normalize = normalize
        self._model: Optional[SentenceTransformer] = None
        self._lock = threading.RLock()
        self._dimension: int = 384

    def load_model(self) -> None:
        """
        Loads model into memory once.
        Checks model_path first if specified and exists, otherwise falls back to model_name.
        """
        with self._lock:
            if self._model is not None:
                logger.info("Model is already loaded in memory.")
                return

            target = self.model_name
            if self.model_path and os.path.exists(self.model_path):
                target = self.model_path
                logger.info(f"Loading SentenceTransformer from configured local path: {target}")
            else:
                logger.info(f"Local path '{self.model_path}' not found or unspecified. Loading from model name: {target}")

            try:
                self._model = SentenceTransformer(target, device=self.device)
                self._dimension = self._model.get_embedding_dimension()
                MODEL_LOADED.set(1)
                logger.info(
                    f"Successfully loaded model '{self.model_name}' (dimension={self._dimension}) on device '{self.device}'."
                )
                self._warmup()
            except Exception as e:
                MODEL_LOADED.set(0)
                logger.error(f"Failed to load embedding model: {str(e)}", exc_info=True)
                raise ModelLoadException(f"Failed to load model '{target}': {str(e)}")

    def _warmup(self) -> None:
        """Runs dummy encoding to warm up PyTorch JIT execution graphs."""
        if self._model is not None:
            logger.info("Warming up embedding model...")
            self.encode_single("GraphGPT Warmup")
            logger.info("Model warmup complete.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_dimension(self) -> int:
        return self._dimension

    def encode_single(self, text: str) -> List[float]:
        """
        Encodes a single string into an embedding vector.
        """
        if self._model is None:
            raise ModelLoadException("Model is not loaded.")

        with self._lock:
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False
            )
        return embedding.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encodes a batch of strings into embedding vectors.
        """
        if self._model is None:
            raise ModelLoadException("Model is not loaded.")

        if not texts:
            return []

        with self._lock:
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                batch_size=len(texts),
                show_progress_bar=False
            )
        return embeddings.tolist()

    def unload(self) -> None:
        with self._lock:
            self._model = None
            MODEL_LOADED.set(0)
            logger.info("Embedding model unloaded from memory.")
