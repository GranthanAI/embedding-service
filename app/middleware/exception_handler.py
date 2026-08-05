from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    BatchSizeExceededException,
    EmptyTextException,
    ModelLoadException,
    TextTooLongException,
    ValidationException,
)
from app.core.logging import logger
from app.schemas.embedding import ErrorResponse


def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(EmptyTextException)
    async def empty_text_handler(request: Request, exc: EmptyTextException):
        logger.warning(f"Empty text validation failure: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(error="Empty Text", code=exc.code, details=exc.message).model_dump()
        )

    @app.exception_handler(TextTooLongException)
    async def text_too_long_handler(request: Request, exc: TextTooLongException):
        logger.warning(f"Text length limit exceeded: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(error="Text Exceeds Limit", code=exc.code, details=exc.message).model_dump()
        )

    @app.exception_handler(BatchSizeExceededException)
    async def batch_size_handler(request: Request, exc: BatchSizeExceededException):
        logger.warning(f"Batch size limit exceeded: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(error="Batch Size Exceeded", code=exc.code, details=exc.message).model_dump()
        )

    @app.exception_handler(ValidationException)
    async def validation_handler(request: Request, exc: ValidationException):
        logger.warning(f"Validation error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(error="Validation Error", code=exc.code, details=exc.message).model_dump()
        )

    @app.exception_handler(ModelLoadException)
    async def model_load_handler(request: Request, exc: ModelLoadException):
        logger.error(f"Model load exception: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(error="Model Service Unavailable", code=exc.code, details=exc.message).model_dump()
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled internal server error: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(error="Internal Server Error", code="INTERNAL_ERROR", details=str(exc)).model_dump()
        )
