# Embedding service performance under load

Problem: `ModelManager` serialised every request behind one lock, so a 64-256 text ingestion batch blocked single query
embeddings (retrieval-side timeout 0.4 s).

## Approach (chosen and measured)

One model instance, two lanes (`app/models/priority_gate.py`): bulk batches are cut into slices of `BULK_SLICE_SIZE`
texts; each slice takes the model for one call, and a waiting query always goes before the next slice. A query therefore
waits for at most one slice, not the whole batch. Admission to the bulk lane is bounded (`BULK_MAX_PENDING` batches,
running + waiting); beyond it the request fails with RESOURCE_EXHAUSTED (gRPC) / 429 (REST).
A second model instance was not used: it would double model memory and the two instances would contend for the same CPU
cores, so it does not remove the wait. Memory cost of the chosen approach: none beyond the existing model.
Limits: a query still waits for the slice in flight; the bulk lane can starve under a continuous query flood.
Rerank has its own lock and is not behind the embedding lane (it is a separate model).

## Measurement (`scripts/bench_under_load.py`, real all-MiniLM-L6-v2, CPU, 128 ingestion texts of ~80 words looping, one query every 100 ms)

| Case | Query samples | p50 | p95 | max | queries over 400 ms |
|---|---|---|---|---|---|
| Before (single lock) | 3 | 8852 ms | 10481 ms | 10481 ms | 3 of 3 |
| After, lanes on, slice 8 | 52 | 210 ms | 726 ms | 3550 ms | 8 |
| After, lanes on, slice 2 | 96 | 102 ms | 487 ms | 1191 ms | 10 |

Idle query p50 was 120 ms (before run) and 33-35 ms (after runs): the machine is shared with other benchmarks, so
absolute numbers are noisy and only the order of magnitude is reliable. Samples are 25 s windows, not repeated.
Bulk batch time rose from 7.2 s (mean) to 7.0-9.2 s with slice 2. The "before" run only completed 3 queries in 25 s
because each blocked for a whole batch, so its sample is small. The 0.4 s target was not met at p95 on this loaded
machine; smaller slices help (try `BULK_SLICE_SIZE=2`), idle-machine numbers were not measured.

## Flags

| Flag | Default | Note |
|---|---|---|
| `PRIORITY_LANES_ENABLED` | false | Off keeps the original single-lock behaviour; turn on in production after review |
| `BULK_SLICE_SIZE` | 16 | texts per model call in the bulk lane; 2-8 recommended |
| `BULK_MAX_PENDING` | 4 | bulk batches admitted; more -> RESOURCE_EXHAUSTED |
| `QUERY_CACHE_SIZE` | 0 | in-process LRU of single-query vectors keyed by (model, whitespace-normalised text); 0 disables |
| `QUERY_CACHE_REDIS_URL` | empty | optional shared layer (e.g. `redis://localhost:6379/0`, `graphgpt-redis` runs locally); any Redis error silently falls back to no cache with a 30 s back-off; keys are SHA-256 hashes, no text stored in keys or logged |
| `QUERY_CACHE_REDIS_TTL_SECONDS` | 3600 | |
| `TRUNCATION_METRIC_ENABLED` | true | counts inputs over `max_seq_length` tokens; vectors unchanged |

The `redis` Python package is optional and not a declared dependency; without it the Redis layer is disabled.

## Metrics

`embedding_truncated_total` (with `embedding_inputs_total` as denominator), `embedding_query_cache_hits_total{layer}`,
`embedding_query_cache_misses_total`, `embedding_bulk_rejected_total`.
