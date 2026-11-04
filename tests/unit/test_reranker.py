from unittest.mock import MagicMock

import grpc
import pytest

from app.core.exceptions import ValidationException
from app.grpc.generated import embedding_pb2 as pb
from app.grpc.servicer import EmbeddingServiceServicer
from app.models.reranker import Reranker


class FakeCrossEncoder:
    instances = 0

    def __init__(self, name, device="cpu", max_length=512):
        FakeCrossEncoder.instances += 1
        self.name, self.max_length = name, max_length

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        # score = number of query words that appear in the passage
        return [float(sum(w in p for w in q.split())) for q, p in pairs]


def make(**kw):
    FakeCrossEncoder.instances = 0
    return Reranker(model_name="fake/ce", model_factory=FakeCrossEncoder, **kw)


def test_scores_one_per_passage_in_request_order():
    r = make()
    scores = r.rerank("alpha beta", ["none", "alpha beta", "alpha"])
    assert scores == [0.0, 2.0, 1.0]
    assert all(isinstance(s, float) for s in scores)


def test_loading_is_lazy_and_happens_once():
    r = make()
    assert not r.is_loaded() and FakeCrossEncoder.instances == 0
    r.rerank("a", ["a"])
    r.rerank("a", ["a"])
    assert r.is_loaded() and FakeCrossEncoder.instances == 1


def test_explicit_load_preloads():
    r = make()
    r.load()
    assert FakeCrossEncoder.instances == 1


@pytest.mark.parametrize("query,passages", [("", ["a"]), ("   ", ["a"]), ("q", [])])
def test_validation_errors(query, passages):
    r = make()
    with pytest.raises(ValidationException):
        r.rerank(query, passages)
    assert FakeCrossEncoder.instances == 0  # rejected before any model load


def test_too_many_passages_rejected():
    r = make(max_passages=3)
    with pytest.raises(ValidationException):
        r.rerank("q", ["a"] * 4)
    assert len(r.rerank("q", ["a"] * 3)) == 3


def test_servicer_returns_model_and_scores():
    s = EmbeddingServiceServicer(MagicMock(), reranker=make())
    res = s.Rerank(pb.RerankRequest(query="x y", passages=["x", "y x", "z"]), MagicMock())
    assert res.model == "fake/ce" and list(res.scores) == [1.0, 2.0, 0.0]


def test_servicer_invalid_argument_on_validation_error():
    ctx = MagicMock()
    ctx.abort.side_effect = RuntimeError("aborted")
    s = EmbeddingServiceServicer(MagicMock(), reranker=make(max_passages=1))
    with pytest.raises(RuntimeError):
        s.Rerank(pb.RerankRequest(query="q", passages=["a", "b"]), ctx)
    assert ctx.abort.call_args[0][0] == grpc.StatusCode.INVALID_ARGUMENT


def test_servicer_disabled_is_unimplemented():
    ctx = MagicMock()
    s = EmbeddingServiceServicer(MagicMock(), reranker=None)
    s.Rerank(pb.RerankRequest(query="q", passages=["a"]), ctx)
    assert ctx.abort.call_args[0][0] == grpc.StatusCode.UNIMPLEMENTED
    assert "disabled" in ctx.abort.call_args[0][1]
