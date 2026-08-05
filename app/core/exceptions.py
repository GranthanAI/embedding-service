class EmbeddingServiceException(Exception):
    """Base exception class for Embedding Service."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR"):
        super().__init__(message)
        self.message = message
        self.code = code


class ModelLoadException(EmbeddingServiceException):
    """Raised when loading model fails."""
    def __init__(self, message: str):
        super().__init__(message, code="MODEL_LOAD_ERROR")


class ValidationException(EmbeddingServiceException):
    """Raised on invalid input parameters."""
    def __init__(self, message: str):
        super().__init__(message, code="VALIDATION_ERROR")


class EmptyTextException(ValidationException):
    """Raised when empty or whitespace-only text is provided."""
    def __init__(self, message: str = "Text cannot be null or empty"):
        super().__init__(message)


class TextTooLongException(ValidationException):
    """Raised when text exceeds character length limit."""
    def __init__(self, max_length: int, actual_length: int):
        super().__init__(f"Text length {actual_length} exceeds maximum limit of {max_length}")


class BatchSizeExceededException(ValidationException):
    """Raised when batch size exceeds limit."""
    def __init__(self, max_batch: int, actual_batch: int):
        super().__init__(f"Batch size {actual_batch} exceeds maximum limit of {max_batch}")
