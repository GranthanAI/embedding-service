import threading
from collections import OrderedDict
from typing import Callable, List, Optional

from app.core.exceptions import ValidationException
from app.core.logging import logger
from app.models.reranker import Reranker


class RerankerPool:
    """
    The rerankers this server may use: a default (RERANKER_MODEL) plus an allowlist (RERANKER_ALLOWED_MODELS) that a request can select
    with RerankRequest.model. Models load lazily and at most `max_loaded` stay in memory: selecting another evicts the least recently used.
    A model outside the allowlist is refused, so callers can never make this service download or load an arbitrary model.

    The pool exposes the same surface the servicer used on a single Reranker (`model_name`, `rerank`, `load`, `is_loaded`) for the default
    model, plus `for_model()` for a requested one.
    """

    def __init__(
        self,
        default_model: str,
        allowed_models: Optional[List[str]] = None,
        max_loaded: int = 1,
        factory: Optional[Callable[[str], Reranker]] = None,
    ):
        self.model_name = default_model
        self.allowed = list(dict.fromkeys([default_model] + [m.strip() for m in (allowed_models or []) if m.strip()]))
        self.max_loaded = max(1, max_loaded)
        self._factory = factory or (lambda name: Reranker(model_name=name))
        self._rerankers: "OrderedDict[str, Reranker]" = OrderedDict()
        self._lock = threading.Lock()

    def for_model(self, model: str = "") -> Reranker:
        name = (model or "").strip() or self.model_name
        if name not in self.allowed:
            raise ValidationException(f"Reranker '{name}' is not enabled on this server. Enabled: {', '.join(self.allowed)}.")
        with self._lock:
            reranker = self._rerankers.get(name)
            if reranker is None:
                reranker = self._factory(name)
                self._rerankers[name] = reranker
            self._rerankers.move_to_end(name)
            while len([r for r in self._rerankers.values() if r.is_loaded() or r is reranker]) > self.max_loaded:
                victim_name = next(n for n, r in self._rerankers.items() if r is not reranker and r.is_loaded())
                logger.info(f"Evicting reranker '{victim_name}' to stay within RERANKER_MAX_LOADED={self.max_loaded}")
                self._rerankers[victim_name].unload()
                del self._rerankers[victim_name]
        return reranker

    def is_loaded(self) -> bool:
        return any(r.is_loaded() for r in self._rerankers.values())

    def load(self) -> None:
        self.for_model(self.model_name).load()

    def rerank(self, query: str, passages: List[str], model: str = "") -> List[float]:
        return self.for_model(model).rerank(query, passages)
