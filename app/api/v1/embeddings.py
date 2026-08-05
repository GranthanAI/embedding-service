from fastapi import APIRouter, Depends, status
from app.api.dependencies import get_embedding_service
from app.schemas.embedding import (
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)
from app.services.embedding_service import EmbeddingService

router = APIRouter()


@router.post(
    "/embed",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate single text embedding vector",
    description="Encodes a single text string into a 384-dimensional vector using all-MiniLM-L6-v2."
)
async def generate_single_embedding(
    request: EmbeddingRequest,
    service: EmbeddingService = Depends(get_embedding_service)
) -> EmbeddingResponse:
    return service.generate_single(request.text)


@router.post(
    "/embed/batch",
    response_model=BatchEmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate batch text embedding vectors",
    description="Encodes multiple text strings in a single batch request into 384-dimensional vectors."
)
async def generate_batch_embeddings(
    request: BatchEmbeddingRequest,
    service: EmbeddingService = Depends(get_embedding_service)
) -> BatchEmbeddingResponse:
    return service.generate_batch(request.texts)
