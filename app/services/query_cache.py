import hashlib
import threading
import time
from collections import OrderedDict
from typing import List, Optional

import numpy as np
from prometheus_client import Counter

from app.core.logging import logger

QUERY_CACHE_HITS = Counter(
    "embedding_query_cache_hits_total", "Query embedding cache hits", ["layer"]
)
QUERY_CACHE_MISSES = Counter(
    "embedding_query_cache_misses_total", "Query embedding cache misses"
)


def normalise_text(text: str) -> str:
    """Whitespace-normalised form used as cache key (the tokenizer ignores whitespace runs)."""
    return " ".join(text.split())


class RedisLayer:
    """
    Optional shared cache. Any Redis failure (import, connect, timeout) degrades
    to 'not cached' and the layer is skipped for `cooldown` seconds. Keys are
    hashes, so no text is stored in key names or logs.
    """

    def __init__(self, url: str, ttl_seconds: int = 3600, cooldown: float = 30.0, client=None):
        self.ttl = ttl_seconds
        self.cooldown = cooldown
        self._down_until = 0.0
        self._client = client
        if client is None:
            try:
                import redis  # optional dependency

                self._client = redis.Redis.from_url(
                    url, socket_connect_timeout=0.05, socket_timeout=0.05
                )
            except Exception as e:
                logger.warning(f"Query cache Redis disabled: {type(e).__name__}")

    def _usable(self) -> bool:
        return self._client is not None and time.monotonic() >= self._down_until

    def _trip(self, e: Exception) -> None:
        self._down_until = time.monotonic() + self.cooldown
        logger.warning(f"Query cache Redis unavailable, skipping for {self.cooldown:.0f}s: {type(e).__name__}")

    def get(self, key: str) -> Optional[List[float]]:
        if not self._usable():
            return None
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            return np.frombuffer(raw, dtype=np.float32).astype(float).tolist()
        except Exception as e:
            self._trip(e)
            return None

    def set(self, key: str, vector: List[float]) -> None:
        if not self._usable():
            return
        try:
            self._client.set(key, np.asarray(vector, dtype=np.float32).tobytes(), ex=self.ttl)
        except Exception as e:
            self._trip(e)


class QueryCache:
    """LRU cache of single-query embeddings keyed by (model, normalised text). size 0 disables."""

    def __init__(self, size: int = 0, redis_layer: Optional[RedisLayer] = None):
        self.size = size
        self.redis = redis_layer
        self._data: "OrderedDict[tuple, List[float]]" = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.size > 0 or self.redis is not None

    def _redis_key(self, key: tuple) -> str:
        return "emb:" + hashlib.sha256("\x00".join(key).encode("utf-8")).hexdigest()

    def get(self, model: str, text: str) -> Optional[List[float]]:
        if not self.enabled:
            return None
        key = (model, normalise_text(text))
        if self.size > 0:
            with self._lock:
                vec = self._data.get(key)
                if vec is not None:
                    self._data.move_to_end(key)
                    QUERY_CACHE_HITS.labels(layer="memory").inc()
                    return list(vec)
        if self.redis is not None:
            vec = self.redis.get(self._redis_key(key))
            if vec is not None:
                self._put_memory(key, vec)
                QUERY_CACHE_HITS.labels(layer="redis").inc()
                return vec
        QUERY_CACHE_MISSES.inc()
        return None

    def _put_memory(self, key: tuple, vector: List[float]) -> None:
        if self.size <= 0:
            return
        with self._lock:
            self._data[key] = list(vector)
            self._data.move_to_end(key)
            while len(self._data) > self.size:
                self._data.popitem(last=False)

    def put(self, model: str, text: str, vector: List[float]) -> None:
        if not self.enabled:
            return
        key = (model, normalise_text(text))
        self._put_memory(key, vector)
        if self.redis is not None:
            self.redis.set(self._redis_key(key), vector)
