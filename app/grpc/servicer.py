import grpc
from app.core.exceptions import EmbeddingServiceException, ValidationException
from app.core.logging import logger
from app.grpc.generated import embedding_pb2, embedding_pb2_grpc
from app.services.embedding_service import EmbeddingService


class EmbeddingServiceServicer(embedding_pb2_grpc.EmbeddingServiceServicer):
    """
    gRPC Servicer implementing EmbeddingService RPCs.
    """

    def __init__(self, embedding_service: EmbeddingService, reranker=None):
        self.embedding_service = embedding_service
        self.reranker = reranker

    def Rerank(
        self, request: embedding_pb2.RerankRequest, context: grpc.ServicerContext
    ) -> embedding_pb2.RerankResponse:
        if self.reranker is None:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "Reranker is disabled on this server (RERANKER_ENABLED=false).",
            )
            return embedding_pb2.RerankResponse()
        try:
            scores = self.reranker.rerank(request.query, list(request.passages))
            return embedding_pb2.RerankResponse(model=self.reranker.model_name, scores=scores)
        except EmbeddingServiceException as e:
            logger.warning(f"gRPC Rerank error: {e.message}")
            code = (
                grpc.StatusCode.INVALID_ARGUMENT
                if isinstance(e, ValidationException)
                else grpc.StatusCode.INTERNAL
            )
            context.abort(code, e.message)
        except Exception as e:
            logger.error(f"gRPC Rerank error: {type(e).__name__}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, "Internal server error")

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

    def GetModelInfo(
        self, request: embedding_pb2.ModelInfoRequest, context: grpc.ServicerContext
    ) -> embedding_pb2.ModelInfoResponse:
        try:
            manager = self.embedding_service.model_manager
            return embedding_pb2.ModelInfoResponse(
                model=manager.model_name,
                dimension=manager.get_dimension(),
                max_seq_length=manager.get_max_seq_length(),
                tokenizer_id=manager.get_tokenizer_id(),
                normalized=manager.normalize,
            )
        except Exception as e:
            logger.error(f"gRPC GetModelInfo error: {str(e)}", exc_info=True)
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
