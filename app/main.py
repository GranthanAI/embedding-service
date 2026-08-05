from fastapi import FastAPI, Response, status
from prometheus_client import make_asgi_app
import uvicorn

from app.api.router import api_router
from app.core.config import settings
from app.core.logging import logger, setup_logging
from app.lifespan import lifespan
from app.middleware.correlation import CorrelationIDMiddleware
from app.middleware.exception_handler import register_exception_handlers
from app.schemas.embedding import HealthResponse, ReadinessResponse

setup_logging(debug_mode=settings.DEBUG, log_level=settings.LOG_LEVEL)

app = FastAPI(
    title="GraphGPT Embedding Service",
    description="Standalone internal microservice for generating text embeddings using all-MiniLM-L6-v2.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── Middleware ───────────────────────────────────────────────────────────────
app.add_middleware(CorrelationIDMiddleware)
register_exception_handlers(app)

# ── API Routes ───────────────────────────────────────────────────────────────
app.include_router(api_router)

# ── Prometheus Metrics ───────────────────────────────────────────────────────
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ── Health & Readiness Probes ────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Observability"],
    summary="Liveness Probe"
)
async def health_check():
    """Shallow process liveness probe."""
    return HealthResponse(status="healthy")


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    tags=["Observability"],
    summary="Readiness Probe"
)
async def readiness_check(response: Response):
    """Deep readiness probe checking model loading status."""
    container = getattr(app.state, "container", None)
    model_ok = container is not None and container.model_manager is not None and container.model_manager.is_loaded()

    if not model_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="DOWN",
            details={"model": "UNLOADED"}
        )

    return ReadinessResponse(
        status="UP",
        details={
            "model": settings.MODEL_NAME,
            "dimension": str(container.model_manager.get_dimension()),
            "device": settings.DEVICE
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.REST_HOST,
        port=settings.REST_PORT,
        reload=False
    )
