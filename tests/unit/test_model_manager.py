from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app.core.exceptions import ModelLoadException
from app.models.model_manager import ModelManager


def test_model_manager_init():
    mm = ModelManager(model_name="all-MiniLM-L6-v2", device="cpu")
    assert not mm.is_loaded()
    assert mm.get_dimension() == 384


@patch("app.models.model_manager.SentenceTransformer")
def test_model_manager_load_and_encode(mock_st):
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.encode.return_value = np.zeros(384)
    mock_st.return_value = mock_model

    mm = ModelManager(model_name="all-MiniLM-L6-v2", device="cpu")
    mm.load_model()

    assert mm.is_loaded()
    vec = mm.encode_single("Test string")
    assert len(vec) == 384


def test_encode_before_load_raises():
    mm = ModelManager(model_name="all-MiniLM-L6-v2", device="cpu")
    with pytest.raises(ModelLoadException):
        mm.encode_single("Test string")


@patch("app.models.model_manager.SentenceTransformer")
def test_model_info_reports_max_seq_length_and_tokenizer_id(mock_st):
    mock_model = MagicMock()
    mock_model.max_seq_length = 256
    mock_st.return_value = mock_model
    mm = ModelManager(model_name="all-MiniLM-L6-v2", device="cpu")
    mm.load_model()

    assert mm.get_max_seq_length() == 256
    assert mm.get_tokenizer_id() == "sentence-transformers/all-MiniLM-L6-v2"
    assert ModelManager(model_name="BAAI/bge-m3").get_tokenizer_id() == "BAAI/bge-m3"
    assert ModelManager(model_name="x", tokenizer_id="org/tok").get_tokenizer_id() == "org/tok"


def test_max_seq_length_before_load_raises():
    with pytest.raises(ModelLoadException):
        ModelManager(model_name="all-MiniLM-L6-v2").get_max_seq_length()
