.PHONY: help setup dev test clean protos docker-build

help:
	@echo "========================================================================"
	@echo "GraphGPT Embedding Service - Setup & Execution Commands"
	@echo "========================================================================"
	@echo "  make setup        - Install dependencies via uv"
	@echo "  make protos       - Compile gRPC proto definitions"
	@echo "  make dev          - Launch FastAPI REST and gRPC servers"
	@echo "  make test         - Run pytest unit & integration test suite"
	@echo "  make clean        - Clean cache and build artifacts"
	@echo "  make docker-build - Build production Docker image"
	@echo "========================================================================"

setup:
	@python -c "import os, shutil; os.path.exists('.env') or shutil.copy('.env.example', '.env')"
	uv sync

protos:
	python -m grpc_tools.protoc -Iprotos --python_out=app/grpc/generated --grpc_python_out=app/grpc/generated protos/embedding.proto
	@python -c "import os; p='app/grpc/generated/embedding_pb2_grpc.py'; open(p, 'w').write(open(p).read().replace('import embedding_pb2 as embedding__pb2', 'from app.grpc.generated import embedding_pb2 as embedding__pb2')) if os.path.exists(p) else None"

dev:
	uv run python -m app.main

test:
	uv run python -m pytest tests/

clean:
	@python -c "import shutil, glob; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True)]"

docker-build:
	docker build -t graphgpt-embedding-service:latest .
