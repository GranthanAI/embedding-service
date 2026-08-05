# GraphGPT Embedding Service API Documentation

This document describes the REST API endpoints provided by the GraphGPT Embedding Service. The service generates high-quality text embedding vectors using the `all-MiniLM-L6-v2` model.

By default, the REST server runs at:
* **Base URL**: `http://localhost:8000` (or `http://127.0.0.1:8000`)
* **Interactive Swagger UI**: `http://localhost:8000/docs`
* **Alternative Redoc UI**: `http://localhost:8000/redoc`

---

## Endpoint Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/v1/embed` | Generate single text embedding vector |
| `POST` | `/v1/embed/batch` | Generate batch text embedding vectors |
| `GET` | `/health` | Liveness probe (shallow) |
| `GET` | `/ready` | Readiness probe (deep check including model status) |
| `GET` | `/metrics` | Prometheus metrics endpoint |

---

## 1. Single Embedding Generation

### `POST /v1/embed`

Generate a 384-dimensional floating-point embedding vector for a single string input.

#### Request Headers
```http
Content-Type: application/json
```

#### Request Body (JSON)
* `text` (string, required): The text content to generate an embedding for.

**Example Request:**
```json
{
  "text": "User prefers FastAPI"
}
```

#### Response Body (JSON)
* `model` (string): The identifier of the model used (e.g. `all-MiniLM-L6-v2`).
* `dimension` (integer): The size of the embedding vector (384).
* `embedding` (array of floats): The generated 384-dimensional vector.

**Example Response (200 OK):**
```json
{
  "model": "all-MiniLM-L6-v2",
  "dimension": 384,
  "embedding": [
    0.01234567,
    -0.08765432,
    0.00543210,
    ...
  ]
}
```

#### cURL Example
```bash
curl -X POST http://localhost:8000/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "User prefers FastAPI"}'
```

---

## 2. Batch Embedding Generation

### `POST /v1/embed/batch`

Generate 384-dimensional floating-point embedding vectors for multiple string inputs in a single request.

#### Request Headers
```http
Content-Type: application/json
```

#### Request Body (JSON)
* `texts` (array of strings, required): A list of text strings to embed.

**Example Request:**
```json
{
  "texts": [
    "User prefers FastAPI",
    "User builds GraphGPT"
  ]
}
```

#### Response Body (JSON)
* `model` (string): The identifier of the model used.
* `dimension` (integer): The size of each embedding vector (384).
* `embeddings` (array of float arrays): List containing a 384-dimensional vector for each input text in the request.

**Example Response (200 OK):**
```json
{
  "model": "all-MiniLM-L6-v2",
  "dimension": 384,
  "embeddings": [
    [0.0123, -0.0876, 0.0054, ...],
    [0.0987, -0.0654, 0.0032, ...]
  ]
}
```

#### cURL Example
```bash
curl -X POST http://localhost:8000/v1/embed/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["User prefers FastAPI", "User builds GraphGPT"]}'
```

---

## 3. Liveness Probe

### `GET /health`

Shallow liveness check to verify the process is alive.

#### Response Body (JSON)
**Example Response (200 OK):**
```json
{
  "status": "healthy"
}
```

#### cURL Example
```bash
curl -X GET http://localhost:8000/health
```

---

## 4. Readiness Probe

### `GET /ready`

Deep check verifying if the model is fully loaded in memory and ready to serve requests.

#### Response Body (JSON)

**Example Response (200 OK - Ready):**
```json
{
  "status": "UP",
  "details": {
    "model": "all-MiniLM-L6-v2",
    "dimension": "384",
    "device": "cpu"
  }
}
```

**Example Response (503 Service Unavailable - Not Ready):**
```json
{
  "status": "DOWN",
  "details": {
    "model": "UNLOADED"
  }
}
```

#### cURL Example
```bash
curl -X GET http://localhost:8000/ready
```

---

## 5. Error Response Format

For validation or server errors, the API returns a structured JSON error body.

#### Response Body (JSON)
* `error` (string): Summary of the error.
* `code` (string): Machine-readable error code.
* `details` (string, optional): Extra details or trace back.

**Example Response (422 Unprocessable Entity - Validation Error):**
```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Example Custom Error Response:**
```json
{
  "error": "Failed to load embedding model",
  "code": "MODEL_LOAD_ERROR",
  "details": "Failed to load model 'all-MiniLM-L6-v2' on CPU: out of memory"
}
```
