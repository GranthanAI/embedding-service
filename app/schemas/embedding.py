from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    text: str = Field(..., description="Text input to generate embedding vector for", examples=["User prefers FastAPI"])


class EmbeddingResponse(BaseModel):
    model: str = Field(..., examples=["all-MiniLM-L6-v2"])
    dimension: int = Field(..., examples=[384])
    embedding: List[float] = Field(..., description="Generated 384-dimensional float vector")


class BatchEmbeddingRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        description="List of text inputs to generate embeddings for",
        examples=[["User prefers FastAPI", "User builds GraphGPT"]]
    )


class BatchEmbeddingResponse(BaseModel):
    model: str = Field(..., examples=["all-MiniLM-L6-v2"])
    dimension: int = Field(..., examples=[384])
    embeddings: List[List[float]] = Field(..., description="List of generated 384-dimensional float vectors")


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])


class ReadinessResponse(BaseModel):
    status: str = Field(..., examples=["UP", "DOWN"])
    details: Dict[str, str] = Field(..., examples=[{"model": "UP", "dimension": "384"}])


class ErrorResponse(BaseModel):
    error: str
    code: str
    details: Optional[str] = None
