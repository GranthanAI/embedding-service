from unittest.mock import MagicMock
import pytest

from app.core.exceptions import (
    BatchSizeExceededException,
    EmptyTextException,
    TextTooLongException,
    ValidationException,
)
from app.services.embedding_service import EmbeddingService


@pytest.fixture
def mock_model_manager():
    mm = MagicMock()
    mm.model_name = "all-MiniLM-L6-v2"
    mm.get_dimension.return_value = 384
    mm.encode_single.return_value = [0.1] * 384
    mm.encode_batch.return_value = [[0.1] * 384, [0.2] * 384]
    return mm


def test_embedding_service_generate_single_success(mock_model_manager):
    service = EmbeddingService(model_manager=mock_model_manager, max_text_length=100)
    res = service.generate_single("Hello world")
    assert res.model == "all-MiniLM-L6-v2"
    assert res.dimension == 384
    assert len(res.embedding) == 384


def test_embedding_service_empty_text(mock_model_manager):
    service = EmbeddingService(model_manager=mock_model_manager)
    with pytest.raises(EmptyTextException):
        service.generate_single("   ")


def test_embedding_service_text_too_long(mock_model_manager):
    service = EmbeddingService(model_manager=mock_model_manager, max_text_length=10)
    with pytest.raises(TextTooLongException):
        service.generate_single("This text is longer than 10 characters")


def test_embedding_service_generate_batch_success(mock_model_manager):
    service = EmbeddingService(model_manager=mock_model_manager, max_batch_size=5)
    res = service.generate_batch(["Text 1", "Text 2"])
    assert res.model == "all-MiniLM-L6-v2"
    assert len(res.embeddings) == 2


def test_embedding_service_batch_size_exceeded(mock_model_manager):
    service = EmbeddingService(model_manager=mock_model_manager, max_batch_size=2)
    with pytest.raises(BatchSizeExceededException):
        service.generate_batch(["1", "2", "3"])
