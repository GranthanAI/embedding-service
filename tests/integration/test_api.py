from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.services.embedding_service import EmbeddingService


@pytest.fixture
def client_with_mocked_service(monkeypatch):
    mock_service = MagicMock()
    mock_service.generate_single.return_value.model = "all-MiniLM-L6-v2"
    mock_service.generate_single.return_value.dimension = 384
    mock_service.generate_single.return_value.embedding = [0.0] * 384

    mock_service.generate_batch.return_value.model = "all-MiniLM-L6-v2"
    mock_service.generate_batch.return_value.dimension = 384
    mock_service.generate_batch.return_value.embeddings = [[0.0] * 384, [0.0] * 384]

    mock_container = MagicMock()
    mock_container.embedding_service = mock_service
    mock_container.model_manager.is_loaded.return_value = True
    mock_container.model_manager.get_dimension.return_value = 384

    async def mock_init_resources(self):
        self.embedding_service = mock_service
        self.model_manager = mock_container.model_manager

    async def mock_shutdown_resources(self):
        pass

    # Patch Container so the lifespan startup does not load the real SentenceTransformer
    from app.core.container import Container
    monkeypatch.setattr(Container, "init_resources", mock_init_resources)
    monkeypatch.setattr(Container, "shutdown_resources", mock_shutdown_resources)

    # Disable gRPC server start in test lifespan to avoid port conflicts / background threads
    from app.grpc.server import GRPCServer
    monkeypatch.setattr(GRPCServer, "start", lambda self: None)
    monkeypatch.setattr(GRPCServer, "stop", lambda self: None)

    with TestClient(app) as client:
        client.app.state.container = mock_container
        yield client


def test_health_check(client_with_mocked_service):
    res = client_with_mocked_service.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "healthy"}


def test_readiness_check(client_with_mocked_service):
    res = client_with_mocked_service.get("/ready")
    assert res.status_code == 200
    assert res.json()["status"] == "UP"


def test_embed_single_endpoint(client_with_mocked_service):
    res = client_with_mocked_service.post("/v1/embed", json={"text": "User prefers FastAPI"})
    assert res.status_code == 200
    data = res.json()
    assert data["model"] == "all-MiniLM-L6-v2"
    assert data["dimension"] == 384
    assert len(data["embedding"]) == 384


def test_embed_batch_endpoint(client_with_mocked_service):
    res = client_with_mocked_service.post("/v1/embed/batch", json={"texts": ["Text 1", "Text 2"]})
    assert res.status_code == 200
    data = res.json()
    assert data["model"] == "all-MiniLM-L6-v2"
    assert len(data["embeddings"]) == 2
