from unittest.mock import MagicMock

from app.grpc.generated import embedding_pb2
from app.grpc.servicer import EmbeddingServiceServicer


def test_get_model_info_returns_what_a_client_needs_to_chunk():
    manager = MagicMock(model_name="all-MiniLM-L6-v2", normalize=True)
    manager.get_dimension.return_value = 384
    manager.get_max_seq_length.return_value = 256
    manager.get_tokenizer_id.return_value = "sentence-transformers/all-MiniLM-L6-v2"
    servicer = EmbeddingServiceServicer(MagicMock(model_manager=manager))

    info = servicer.GetModelInfo(embedding_pb2.ModelInfoRequest(), MagicMock())

    assert (info.model, info.dimension, info.max_seq_length, info.tokenizer_id, info.normalized) == (
        "all-MiniLM-L6-v2", 384, 256, "sentence-transformers/all-MiniLM-L6-v2", True)


def test_existing_rpcs_are_unchanged():
    names = {m.name for m in embedding_pb2.DESCRIPTOR.services_by_name["EmbeddingService"].methods}
    assert {"GenerateEmbedding", "GenerateEmbeddings"} <= names and "GetModelInfo" in names


def test_rerank_contract_exists_and_is_unimplemented_when_no_reranker_configured():
    import grpc
    import pytest

    from app.grpc.generated import embedding_pb2 as pb

    assert {"Rerank"} <= {m.name for m in pb.DESCRIPTOR.services_by_name["EmbeddingService"].methods}
    request = pb.RerankRequest(query="q", passages=["a", "b"])
    assert list(request.passages) == ["a", "b"] and list(pb.RerankResponse(scores=[0.1, 0.9]).scores) == pytest.approx([0.1, 0.9])
    context = MagicMock()
    servicer = EmbeddingServiceServicer(MagicMock())
    servicer.Rerank(request, context)
    assert context.abort.call_args[0][0] == grpc.StatusCode.UNIMPLEMENTED
