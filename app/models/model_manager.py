import os
import threading
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.exceptions import ModelLoadException, OverloadedException
from app.core.logging import logger
from app.core.metrics import (
    BULK_REJECTED,
    EMBEDDING_INPUTS,
    EMBEDDING_TRUNCATED,
    MODEL_LOADED,
)
from app.models.priority_gate import PriorityGate


def resolve_load_target(model_name: str, model_path: Optional[str]) -> tuple[str, str]:
    """(what to load, why). A local MODEL_PATH is used only if it names the same model as MODEL_NAME (compared by last path component,
    case-insensitively); otherwise MODEL_NAME is loaded. Without this, setting MODEL_NAME=<another model> while MODEL_PATH still pointed at
    the previous one would load the OLD weights under the NEW name and index vectors that are mislabelled."""
    if not model_path or not os.path.exists(model_path):
        return model_name, "model_path_unset_or_missing"
    if os.path.basename(os.path.normpath(model_path)).lower() == model_name.replace("\\", "/").rstrip("/").split("/")[-1].lower():
        return model_path, "model_path_matches_model_name"
    return model_name, "model_path_is_a_different_model"


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
        normalize: bool = True,
        tokenizer_id: Optional[str] = None,
        priority_lanes: bool = False,
        bulk_slice_size: int = 16,
        bulk_max_pending: int = 4,
        truncation_metric: bool = True,
        max_seq_length: int = 0,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.normalize = normalize
        self._model: Optional[SentenceTransformer] = None
        self._lock = threading.RLock()
        self._dimension: int = 384
        self._tokenizer_id = tokenizer_id
        self.priority_lanes = priority_lanes
        self.bulk_slice_size = max(1, bulk_slice_size)
        self.bulk_max_pending = bulk_max_pending
        self.truncation_metric = truncation_metric
        self.max_seq_length_cap = max(0, max_seq_length)
        self.gate = PriorityGate()
        self._pending_bulk = 0
        self._pending_lock = threading.Lock()

    def load_model(self) -> None:
        """
        Loads model into memory once.
        Checks model_path first if specified and exists, otherwise falls back to model_name.
        """
        with self._lock:
            if self._model is not None:
                logger.info("Model is already loaded in memory.")
                return

            target, why = resolve_load_target(self.model_name, self.model_path)
            if why == "model_path_is_a_different_model":
                logger.warning(
                    f"MODEL_PATH '{self.model_path}' is a different model than MODEL_NAME '{self.model_name}'; ignoring MODEL_PATH and "
                    "loading MODEL_NAME."
                )
            elif target == self.model_path:
                logger.info(f"Loading SentenceTransformer from configured local path: {target}")
            else:
                logger.info(f"Local path '{self.model_path}' not found or unspecified. Loading from model name: {target}")

            try:
                self._model = SentenceTransformer(target, device=self.device)
                self._dimension = self._model.get_embedding_dimension()
                if 0 < self.max_seq_length_cap < int(self._model.max_seq_length):
                    self._model.max_seq_length = self.max_seq_length_cap
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

    def get_max_seq_length(self) -> int:
        """Tokens the loaded model reads per text (longer input is truncated by the model, not rejected)."""
        if self._model is None:
            raise ModelLoadException("Model is not loaded.")
        return int(self._model.max_seq_length)

    def get_tokenizer_id(self) -> str:
        """Hugging Face id of the tokenizer for this model. A bare sentence-transformers name gets its org prefix."""
        if self._tokenizer_id:
            return self._tokenizer_id
        return self.model_name if "/" in self.model_name else f"sentence-transformers/{self.model_name}"

    def count_truncated(self, texts: List[str]) -> int:
        """
        Counts inputs whose token length exceeds the model's max_seq_length (the model silently
        truncates them). Observational only: never changes the vectors, never raises.
        """
        if not self.truncation_metric or self._model is None:
            return 0
        try:
            limit = int(self._model.max_seq_length)
            ids = self._model.tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
            over = sum(1 for x in ids if len(x) > limit)
            EMBEDDING_INPUTS.inc(len(texts))
            if over:
                EMBEDDING_TRUNCATED.inc(over)
            return over
        except Exception:
            return 0

    def encode_single(self, text: str) -> List[float]:
        """
        Encodes a single string into an embedding vector (interactive lane when priority lanes are on).
        """
        if self._model is None:
            raise ModelLoadException("Model is not loaded.")

        self.count_truncated([text])
        guard = self.gate.interactive() if self.priority_lanes else self._lock
        with guard:
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False
            )
        return embedding.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encodes a batch of strings into embedding vectors (bulk lane when priority lanes are on:
        bounded admission, processed in slices so interactive calls can interleave).
        """
        if self._model is None:
            raise ModelLoadException("Model is not loaded.")

        if not texts:
            return []

        self.count_truncated(texts)
        if not self.priority_lanes:
            with self._lock:
                embeddings = self._model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize,
                    batch_size=len(texts),
                    show_progress_bar=False
                )
            return embeddings.tolist()

        with self._pending_lock:
            if self._pending_bulk >= self.bulk_max_pending:
                BULK_REJECTED.inc()
                raise OverloadedException(
                    f"Bulk embedding queue is full ({self.bulk_max_pending} batches pending); retry later."
                )
            self._pending_bulk += 1
        try:
            out: List[List[float]] = []
            for start in range(0, len(texts), self.bulk_slice_size):
                piece = texts[start:start + self.bulk_slice_size]
                with self.gate.bulk():
                    embeddings = self._model.encode(
                        piece,
                        convert_to_numpy=True,
                        normalize_embeddings=self.normalize,
                        batch_size=len(piece),
                        show_progress_bar=False
                    )
                out.extend(embeddings.tolist())
            return out
        finally:
            with self._pending_lock:
                self._pending_bulk -= 1

    def unload(self) -> None:
        with self._lock:
            self._model = None
            MODEL_LOADED.set(0)
            logger.info("Embedding model unloaded from memory.")
