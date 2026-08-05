from concurrent import futures
import grpc

from app.core.config import settings
from app.core.logging import logger
from app.grpc.generated import embedding_pb2_grpc
from app.grpc.servicer import EmbeddingServiceServicer
from app.services.embedding_service import EmbeddingService


class GRPCServer:
    """
    gRPC Server lifecycle wrapper.
    """

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.server: grpc.Server = None

    def start(self) -> None:
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=settings.GRPC_MAX_WORKERS),
            options=[
                ('grpc.max_send_message_length', settings.GRPC_MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', settings.GRPC_MAX_MESSAGE_LENGTH),
            ]
        )
        servicer = EmbeddingServiceServicer(self.embedding_service)
        embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(servicer, self.server)

        bind_address = f"{settings.GRPC_HOST}:{settings.GRPC_PORT}"
        self.server.add_insecure_port(bind_address)
        self.server.start()
        logger.info(f"gRPC server started on {bind_address} with {settings.GRPC_MAX_WORKERS} workers.")

    def stop(self, grace: float = 5.0) -> None:
        if self.server:
            logger.info("Stopping gRPC server...")
            self.server.stop(grace)
            logger.info("gRPC server stopped.")
