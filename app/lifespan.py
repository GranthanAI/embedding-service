from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.core.container import Container
from app.core.logging import logger
from app.grpc.server import GRPCServer


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Starting Embedding Service application lifespan ===")

    # 1. Bootstrap DI container (loads model once)
    container = Container()
    await container.init_resources()
    app.state.container = container

    # 2. Start gRPC server background thread
    grpc_server = GRPCServer(embedding_service=container.embedding_service)
    grpc_server.start()
    app.state.grpc_server = grpc_server

    logger.info("=== Application bootstrap completed successfully ===")

    yield

    logger.info("=== Stopping application and tearing down resources ===")

    # 3. Shutdown gRPC server and container
    if hasattr(app.state, "grpc_server"):
        app.state.grpc_server.stop()

    await container.shutdown_resources()

    logger.info("=== Application shutdown complete ===")
