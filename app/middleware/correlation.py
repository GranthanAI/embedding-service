import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.constants import REQUEST_ID_HEADER
from app.core.logging import set_request_id


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware extracting or generating X-Request-ID and binding it to log context.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        set_request_id(request_id)

        try:
            response: Response = await call_next(request)
        finally:
            set_request_id(None)

        response.headers[REQUEST_ID_HEADER] = request_id
        return response
