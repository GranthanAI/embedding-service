# Canonical dependency source: pyproject.toml + uv.lock (requires-python >= 3.12).

# --- Build Stage ---
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Install exactly what uv.lock pins (runtime dependencies only).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Generate the gRPC stubs (not committed), keeping the package-style import the
# service expects: `from app.grpc.generated import embedding_pb2`.
COPY protos/ ./protos/
RUN mkdir -p app/grpc/generated \
    && /opt/venv/bin/python -m grpc_tools.protoc -Iprotos \
        --python_out=app/grpc/generated --grpc_python_out=app/grpc/generated protos/embedding.proto \
    && sed -i 's/^import embedding_pb2 as embedding__pb2/from app.grpc.generated import embedding_pb2 as embedding__pb2/' \
        app/grpc/generated/embedding_pb2_grpc.py

# --- Final Production Stage ---
FROM python:3.12-slim AS runner

RUN useradd --system --uid 10001 --create-home appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

COPY app/ ./app/
COPY protos/ ./protos/
COPY --from=builder /build/app/grpc/generated/ ./app/grpc/generated/

# No .env is baked into the image: supply configuration (MODEL_PATH, ports, ...)
# through the environment and mount the model weights at MODEL_PATH.
USER appuser

EXPOSE 8005 50054

CMD ["python", "-m", "app.main"]
