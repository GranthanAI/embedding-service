# GraphGPT Embedding Service

Standalone internal microservice responsible ONLY for generating text embeddings using `all-MiniLM-L6-v2`.

## Architecture & Features
- **FastAPI REST API** (`/v1/embed`, `/v1/embed/batch`)
- **gRPC API** (`GenerateEmbedding`, `GenerateEmbeddings`)
- **Singleton ModelManager** with once-at-startup memory resident loading
- **CPU PyTorch Inference**
- **Prometheus Observability** (`/metrics`)
- **Correlation ID Tracking** (`X-Request-ID`)

## Environment Variables
Configured via `.env`:
- `MODEL_NAME`: `all-MiniLM-L6-v2`
- `MODEL_PATH`: Local directory path to the `all-MiniLM-L6-v2` weights (required; `.env.example` has a placeholder)
- `DEVICE`: `cpu`
- `BATCH_SIZE`: `64`
- `MAX_BATCH_SIZE`: `256`
- `MAX_TEXT_LENGTH`: `8192`
- `GRPC_PORT`: `50054`
- `REST_PORT`: `8005`

## Commands
```bash
make setup     # Copy .env.example to .env if missing, then `uv sync`
make protos    # Compile gRPC protos
make dev       # Start service
make test      # Run test suite
```
