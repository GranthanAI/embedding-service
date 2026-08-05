from fastapi import APIRouter
from app.api.v1 import embeddings

api_router = APIRouter()
api_router.include_router(embeddings.router, prefix="/v1", tags=["Embeddings"])
