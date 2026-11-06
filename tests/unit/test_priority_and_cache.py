import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.exceptions import OverloadedException
from app.models.model_manager import ModelManager
from app.services.query_cache import QueryCache, RedisLayer


class FakeModel:
    """Sleeps proportionally to the number of texts; records call sizes."""
    max_seq_length = 8

    def __init__(self, per_text=0.01):
        self.per_text = per_text
        self.calls = []
        self.tokenizer = lambda texts, **kw: {"input_ids": [t.split() for t in texts]}

    def encode(self, texts, **kw):
        n = 1 if isinstance(texts, str) else len(texts)
        self.calls.append(n)
        time.sleep(self.per_text * n)
        return np.zeros(4) if isinstance(texts, str) else np.zeros((n, 4))


def manager(model, **kw):
    m = ModelManager("fake", **kw)
    m._model = model
    return m


def test_query_is_not_blocked_behind_a_long_bulk_batch():
    model = FakeModel(per_text=0.01)  # 128 texts = 1.28 s as one call
    m = manager(model, priority_lanes=True, bulk_slice_size=8)
    t = threading.Thread(target=lambda: m.encode_batch(["x"] * 128))
    t.start()
    time.sleep(0.15)
    start = time.perf_counter()
    m.encode_single("q")
    waited = time.perf_counter() - start
    t.join()
    assert waited < 0.3, waited  # at most one 8-text slice (0.08 s) plus its own encode
    assert max(model.calls) == 8 and sum(c for c in model.calls if c > 1) == 128


def test_without_lanes_query_waits_for_whole_batch():
    m = manager(FakeModel(per_text=0.01))
    t = threading.Thread(target=lambda: m.encode_batch(["x"] * 60))
    t.start()
    time.sleep(0.1)
    start = time.perf_counter()
    m.encode_single("q")
    waited = time.perf_counter() - start
    t.join()
    assert waited > 0.3  # existing behaviour preserved by default


def test_bulk_vectors_keep_order_and_count():
    class Ordered(FakeModel):
        def encode(self, texts, **kw):
            return np.array([[float(t)] * 4 for t in texts])

    m = manager(Ordered(), priority_lanes=True, bulk_slice_size=3)
    out = m.encode_batch([str(i) for i in range(10)])
    assert [v[0] for v in out] == [float(i) for i in range(10)]


def test_bounded_bulk_queue_rejects_when_full():
    m = manager(FakeModel(per_text=0.05), priority_lanes=True, bulk_slice_size=4, bulk_max_pending=1)
    t = threading.Thread(target=lambda: m.encode_batch(["x"] * 8))
    t.start()
    time.sleep(0.05)
    with pytest.raises(OverloadedException):
        m.encode_batch(["y"] * 4)
    t.join()
    assert len(m.encode_batch(["z"] * 2)) == 2  # capacity is released afterwards


def test_servicer_maps_overload_to_resource_exhausted():
    import grpc

    from app.grpc.generated import embedding_pb2 as pb
    from app.grpc.servicer import EmbeddingServiceServicer

    svc = MagicMock()
    svc.generate_batch.side_effect = OverloadedException("full")
    ctx = MagicMock()
    EmbeddingServiceServicer(svc).GenerateEmbeddings(pb.BatchEmbeddingRequest(texts=["a"]), ctx)
    assert ctx.abort.call_args[0][0] == grpc.StatusCode.RESOURCE_EXHAUSTED


def test_truncation_is_counted_without_changing_output():
    from app.core.metrics import EMBEDDING_TRUNCATED

    model = FakeModel(per_text=0)
    m = manager(model)
    before = EMBEDDING_TRUNCATED._value.get()
    assert m.count_truncated(["a b c", " ".join("w" * 20)]) == 1  # 20 tokens > 8
    m.encode_single("one two")
    assert EMBEDDING_TRUNCATED._value.get() == before + 1


def test_truncation_counter_never_breaks_encoding():
    model = FakeModel(per_text=0)
    model.tokenizer = MagicMock(side_effect=RuntimeError("boom"))
    assert len(manager(model).encode_single("hello")) == 4


def test_cache_hit_miss_and_normalisation():
    c = QueryCache(size=4)
    assert c.get("m", "hello  world") is None
    c.put("m", "hello  world", [1.0, 2.0])
    assert c.get("m", " hello world ") == [1.0, 2.0]
    assert c.get("other-model", "hello world") is None


def test_cache_lru_eviction():
    c = QueryCache(size=2)
    c.put("m", "a", [1.0])
    c.put("m", "b", [2.0])
    c.get("m", "a")  # a is now most recent
    c.put("m", "c", [3.0])  # evicts b
    assert c.get("m", "b") is None and c.get("m", "a") == [1.0] and c.get("m", "c") == [3.0]


def test_cache_size_zero_is_disabled():
    c = QueryCache(size=0)
    c.put("m", "a", [1.0])
    assert c.get("m", "a") is None and not c.enabled


def test_service_uses_cache_and_skips_model():
    from app.services.embedding_service import EmbeddingService

    mm = MagicMock(model_name="m")
    mm.encode_single.return_value = [0.5, 0.5]
    mm.get_dimension.return_value = 2
    svc = EmbeddingService(mm, query_cache=QueryCache(size=8))
    assert svc.generate_single("hi").embedding == [0.5, 0.5]
    assert svc.generate_single("hi").embedding == [0.5, 0.5]
    assert mm.encode_single.call_count == 1


class DeadRedis:
    calls = 0

    def get(self, k):
        DeadRedis.calls += 1
        raise ConnectionError("down")

    set = get


def test_redis_failure_degrades_silently_and_backs_off():
    DeadRedis.calls = 0
    c = QueryCache(size=0, redis_layer=RedisLayer("x", client=DeadRedis(), cooldown=60))
    assert c.get("m", "a") is None
    c.put("m", "a", [1.0])  # must not raise
    assert c.get("m", "a") is None
    assert DeadRedis.calls == 1  # circuit open: no further attempts during cooldown


def test_redis_roundtrip_with_fake_client():
    store = {}

    class Fake:
        def get(self, k):
            return store.get(k)

        def set(self, k, v, ex=None):
            store[k] = v

    layer = RedisLayer("x", client=Fake())
    QueryCache(size=0, redis_layer=layer).put("m", "a", [0.25, -1.5])
    assert QueryCache(size=2, redis_layer=layer).get("m", "a") == [0.25, -1.5]
    assert all(k.startswith("emb:") and "a" != k for k in store)


def test_unreachable_redis_url_does_not_raise():
    layer = RedisLayer("redis://127.0.0.1:1/0")
    c = QueryCache(size=1, redis_layer=layer)
    assert c.get("m", "a") is None
    c.put("m", "a", [1.0])
    assert c.get("m", "a") == [1.0]
