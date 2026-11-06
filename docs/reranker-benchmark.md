# Reranker CPU benchmark

`scripts/bench_rerank.py MODEL` : 30 synthetic passages of about 200 tokens, one query, batch size 16, max_length 512,
CPU (6 torch threads), warm-up 3 calls. The machine was shared with other benchmarks, so latencies are pessimistic and noisy.

| Model | Load | p50 | p95 | Peak RSS | Calls |
|---|---|---|---|---|---|
| cross-encoder/ms-marco-MiniLM-L-6-v2 (English) | 7.2 s | 2.59 s | 4.35 s | 610 MB | 30 |
| BAAI/bge-reranker-v2-m3 (multilingual) | not obtained | not obtained | not obtained | not obtained | 0 |

bge-reranker-v2-m3: the model is in the local Hugging Face cache (model.safetensors, 2.27 GB; a few `.incomplete` blobs from an
earlier interrupted download remain in the same folder, nothing was deleted). Two benchmark runs were started
but neither produced a result line (output lost to a pipe/timeout, not a model failure that I could confirm),
and I stopped re-running on the user's request to finish quickly. Re-run: `HF_HUB_OFFLINE=1 python scripts/bench_rerank.py BAAI/bge-reranker-v2-m3 --calls 20`.
Expect it to be several times slower and larger than MiniLM (about 568M parameters vs 22M); this is an expectation, not a measurement.
