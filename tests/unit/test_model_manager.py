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
