"""Which models run is configuration: the embedding model (path trap, prefixes, sequence cap) and the reranker (default, allowlist, eviction)."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.config import SystemSettings as Settings
from app.core.exceptions import ValidationException
from app.grpc.generated import embedding_pb2
from app.grpc.servicer import EmbeddingServiceServicer
from app.models.model_manager import ModelManager, resolve_load_target
from app.models.reranker import Reranker
from app.models.reranker_pool import RerankerPool
from app.services.embedding_service import EmbeddingService
from app.services.query_cache import QueryCache


# ---- the model path can no longer silently keep the old weights ---------------------------------------------------------------
def test_a_local_path_is_used_only_when_it_names_the_configured_model(tmp_path):
    minilm = tmp_path / "all-MiniLM-L6-v2"
    minilm.mkdir()
    assert resolve_load_target("all-MiniLM-L6-v2", str(minilm)) == (str(minilm), "model_path_matches_model_name")
    assert resolve_load_target("sentence-transformers/all-MiniLM-L6-v2", str(minilm))[0] == str(minilm)
    # MODEL_NAME changed to another model but MODEL_PATH still points at the old one: the NAME wins.
    assert resolve_load_target("intfloat/multilingual-e5-large", str(minilm)) == ("intfloat/multilingual-e5-large", "model_path_is_a_different_model")
    assert resolve_load_target("any/model", str(tmp_path / "missing"))[1] == "model_path_unset_or_missing"
    assert resolve_load_target("any/model", None)[1] == "model_path_unset_or_missing"


@patch("app.models.model_manager.SentenceTransformer")
def test_loading_a_different_model_name_ignores_the_stale_path(mock_st, tmp_path):
    old = tmp_path / "all-MiniLM-L6-v2"
    old.mkdir()
    mock_st.return_value = MagicMock(max_seq_length=512, get_embedding_dimension=MagicMock(return_value=1024), encode=MagicMock(return_value=np.zeros(1024)))
    ModelManager(model_name="intfloat/multilingual-e5-large", model_path=str(old)).load_model()
    assert mock_st.call_args.args[0] == "intfloat/multilingual-e5-large"


@patch("app.models.model_manager.SentenceTransformer")
def test_max_seq_length_can_only_lower_the_models_own_limit(mock_st):
    model = MagicMock(max_seq_length=512, get_embedding_dimension=MagicMock(return_value=384), encode=MagicMock(return_value=np.zeros(384)))
    mock_st.return_value = model
    mm = ModelManager(model_name="m", max_seq_length=256)
    mm.load_model()
    assert model.max_seq_length == 256 and mm.get_max_seq_length() == 256
    model2 = MagicMock(max_seq_length=128, get_embedding_dimension=MagicMock(return_value=384), encode=MagicMock(return_value=np.zeros(384)))
    mock_st.return_value = model2
    ModelManager(model_name="m", max_seq_length=256).load_model()
    assert model2.max_seq_length == 128                              # a cap above the model's limit does nothing
    model3 = MagicMock(max_seq_length=512, get_embedding_dimension=MagicMock(return_value=384), encode=MagicMock(return_value=np.zeros(384)))
    mock_st.return_value = model3
    ModelManager(model_name="m", max_seq_length=0).load_model()
    assert model3.max_seq_length == 512                              # 0 = the model's own limit


# ---- prefixes ----------------------------------------------------------------------------------------------------------------
class Recorder:
    model_name, normalize = "e5", True

    def __init__(self):
        self.single, self.batch = [], []

    def get_dimension(self):
        return 3

    def encode_single(self, text):
        self.single.append(text)
        return [1.0, 0.0, 0.0]

    def encode_batch(self, texts):
        self.batch.extend(texts)
        return [[0.0, 1.0, 0.0] for _ in texts]


def test_query_prefix_goes_on_single_calls_and_passage_prefix_on_batches_and_defaults_change_nothing():
    rec = Recorder()
    svc = EmbeddingService(rec, query_prefix="query: ", passage_prefix="passage: ")
    svc.generate_single("  what is x  ")
    svc.generate_batch(["a", "b"])
    assert rec.single == ["query: what is x"] and rec.batch == ["passage: a", "passage: b"]
    plain = Recorder()
    EmbeddingService(plain).generate_single("what is x")
    EmbeddingService(plain).generate_batch(["a"])
    assert plain.single == ["what is x"] and plain.batch == ["a"]


def test_the_query_cache_is_keyed_by_prefix_so_changing_it_cannot_serve_stale_vectors():
    cache = QueryCache(size=10)
    a, b = Recorder(), Recorder()
    EmbeddingService(a, query_cache=cache, query_prefix="query: ").generate_single("hello")
    EmbeddingService(b, query_cache=cache, query_prefix="").generate_single("hello")
    assert a.single == ["query: hello"] and b.single == ["hello"]    # the second service did NOT get the first one's cached vector
    again = Recorder()
    EmbeddingService(again, query_cache=cache, query_prefix="query: ").generate_single("hello")
    assert again.single == []                                         # same prefix: served from cache


# ---- reranker pool -----------------------------------------------------------------------------------------------------------
class FakeCE:
    loads = []

    def __init__(self, name, device="cpu", max_length=512):
        FakeCE.loads.append(name)
        self.name = name

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        return np.array([float(len(p[1])) for p in pairs])


def pool(**kw):
    FakeCE.loads.clear()
    return RerankerPool("org/default", kw.pop("allowed", ["org/small"]), kw.pop("max_loaded", 1),
                        factory=lambda name: Reranker(model_name=name, model_factory=FakeCE), **kw)


def test_empty_model_means_the_default_and_an_allowed_model_can_be_selected():
    p = pool()
    assert p.for_model("").model_name == "org/default" and p.for_model("org/small").model_name == "org/small"
    assert p.allowed == ["org/default", "org/small"]


def test_a_model_outside_the_allowlist_is_refused_before_anything_is_loaded():
    p = pool()
    with pytest.raises(ValidationException, match="not enabled"):
        p.for_model("evil/other-model")
    assert FakeCE.loads == []


def test_only_max_loaded_rerankers_stay_in_memory_and_the_least_recently_used_is_evicted():
    p = pool(max_loaded=1)
    p.rerank("q", ["abc"], model="org/default")
    p.rerank("q", ["abc"], model="org/small")
    assert FakeCE.loads == ["org/default", "org/small"]
    assert not any(r.model_name == "org/default" for r in p._rerankers.values() if r.is_loaded())
    assert p.for_model("org/small").is_loaded()
    roomy = pool(max_loaded=2)
    roomy.rerank("q", ["abc"], model="org/default")
    roomy.rerank("q", ["abc"], model="org/small")
    roomy.rerank("q", ["abc"], model="org/default")
    assert FakeCE.loads == ["org/default", "org/small"]               # nothing was reloaded


# ---- the Rerank RPC ----------------------------------------------------------------------------------------------------------
def rpc(p, **fields):
    ctx = MagicMock()
    ctx.abort.side_effect = RuntimeError("aborted")
    return EmbeddingServiceServicer(MagicMock(), reranker=p).Rerank(embedding_pb2.RerankRequest(query="q", passages=["aa", "bbbb"], **fields), ctx), ctx


def test_rpc_uses_the_requested_model_and_reports_which_one_scored():
    resp, _ = rpc(pool(), model="org/small")
    assert resp.model == "org/small" and list(resp.scores) == [2.0, 4.0]
    resp, _ = rpc(pool())
    assert resp.model == "org/default"


def test_rpc_refuses_a_model_that_is_not_enabled_with_invalid_argument():
    import grpc

    with pytest.raises(RuntimeError):
        rpc(pool(), model="evil/other-model")
    p = pool()
    ctx = MagicMock()
    ctx.abort.side_effect = RuntimeError("aborted")
    with pytest.raises(RuntimeError):
        EmbeddingServiceServicer(MagicMock(), reranker=p).Rerank(embedding_pb2.RerankRequest(query="q", passages=["a"], model="evil/other-model"), ctx)
    assert ctx.abort.call_args.args[0] == grpc.StatusCode.INVALID_ARGUMENT


def test_a_single_reranker_still_works_and_only_accepts_its_own_name():
    single = Reranker(model_name="org/only", model_factory=FakeCE)
    resp, _ = rpc(single)
    assert resp.model == "org/only"
    resp, _ = rpc(single, model="org/only")
    assert resp.model == "org/only"
    with pytest.raises(RuntimeError):
        rpc(single, model="org/other")


# ---- everything above is a setting ---------------------------------------------------------------------------------------------
def test_every_model_choice_is_a_setting_with_a_default_that_changes_nothing(monkeypatch):
    for k in ("MODEL_NAME", "QUERY_PREFIX", "PASSAGE_PREFIX", "MAX_SEQ_LENGTH", "RERANKER_ALLOWED_MODELS"):
        monkeypatch.delenv(k, raising=False)
    s = Settings(_env_file=None)
    assert (s.QUERY_PREFIX, s.PASSAGE_PREFIX, s.MAX_SEQ_LENGTH, s.TOKENIZER_ID, s.RERANKER_ALLOWED_MODELS, s.RERANKER_MAX_LOADED) == ("", "", 0, "", "", 1)
    monkeypatch.setenv("MODEL_NAME", "intfloat/multilingual-e5-large")
    monkeypatch.setenv("QUERY_PREFIX", "query: ")
    monkeypatch.setenv("RERANKER_ALLOWED_MODELS", "a/b, c/d")
    s = Settings(_env_file=None)
    assert s.MODEL_NAME == "intfloat/multilingual-e5-large" and s.QUERY_PREFIX == "query: " and s.RERANKER_ALLOWED_MODELS == "a/b, c/d"
