from fastapi import Request
from app.services.embedding_service import EmbeddingService


def get_container(request: Request):
    return request.app.state.container


def get_embedding_service(request: Request) -> EmbeddingService:
    container = get_container(request)
    return container.embedding_service
