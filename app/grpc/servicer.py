import grpc
from app.core.exceptions import EmbeddingServiceException
from app.core.logging import logger
from app.grpc.generated import embedding_pb2, embedding_pb2_grpc
from app.services.embedding_service import EmbeddingService


class EmbeddingServiceServicer(embedding_pb2_grpc.EmbeddingServiceServicer):
    """
    gRPC Servicer implementing EmbeddingService RPCs.
    """

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    def GenerateEmbedding(
        self, request: embedding_pb2.EmbeddingRequest, context: grpc.ServicerContext
    ) -> embedding_pb2.EmbeddingResponse:
        try:
            res = self.embedding_service.generate_single(request.text)
            return embedding_pb2.EmbeddingResponse(
                model=res.model,
                dimension=res.dimension,
                embedding=res.embedding
            )
        except EmbeddingServiceException as e:
            logger.warning(f"gRPC GenerateEmbedding validation error: {e.message}")
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, e.message)
        except Exception as e:
            logger.error(f"gRPC GenerateEmbedding error: {str(e)}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, f"Internal server error: {str(e)}")

    def GenerateEmbeddings(
        self, request: embedding_pb2.BatchEmbeddingRequest, context: grpc.ServicerContext
    ) -> embedding_pb2.BatchEmbeddingResponse:
        try:
            res = self.embedding_service.generate_batch(list(request.texts))
            vectors = [
                embedding_pb2.FloatVector(values=v) for v in res.embeddings
            ]
            return embedding_pb2.BatchEmbeddingResponse(
                model=res.model,
                dimension=res.dimension,
                embeddings=vectors
            )
        except EmbeddingServiceException as e:
            logger.warning(f"gRPC GenerateEmbeddings validation error: {e.message}")
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, e.message)
        except Exception as e:
            logger.error(f"gRPC GenerateEmbeddings error: {str(e)}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, f"Internal server error: {str(e)}")
