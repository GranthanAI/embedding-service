"""Single-query embedding latency while 128-text ingestion batches run.

  python scripts/bench_under_load.py [--lanes] [--slice 16] [--seconds 40] [--bulk-threads 1]

Uses the real ModelManager and model (MODEL_NAME / MODEL_PATH from settings), synthetic texts.
Without --lanes this is the original single-lock behaviour ("before").
"""
import argparse
import inspect
import json
import random
import statistics
import threading
import time

from app.core.config import settings
from app.models.model_manager import ModelManager

WORDS = "the of and to in is that for it as was with be by on not he this are or his from at which but have an had they you were".split()


def text(rnd, n):
    return " ".join(rnd.choice(WORDS) for _ in range(n))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * len(xs))) - 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", action="store_true")
    ap.add_argument("--slice", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--words", type=int, default=80, help="words per ingestion text")
    ap.add_argument("--interval", type=float, default=0.1, help="seconds between queries")
    ap.add_argument("--bulk-threads", type=int, default=1)
    a = ap.parse_args()

    kw = dict(model_name=settings.MODEL_NAME, model_path=settings.MODEL_PATH, device="cpu")
    if "priority_lanes" in inspect.signature(ModelManager.__init__).parameters:
        kw.update(priority_lanes=a.lanes, bulk_slice_size=a.slice, bulk_max_pending=8)
    elif a.lanes:
        raise SystemExit("this checkout has no priority lanes")
    mm = ModelManager(**kw)
    mm.load_model()

    rnd = random.Random(1)
    bulk = [text(rnd, a.words) for _ in range(a.batch)]
    queries = [text(rnd, 8) for _ in range(2000)]

    idle = []
    for i in range(30):
        s = time.perf_counter()
        mm.encode_single(queries[i])
        idle.append(time.perf_counter() - s)

    stop = threading.Event()
    batches = []
    bulk_lat = []

    def ingest():
        while not stop.is_set():
            s = time.perf_counter()
            mm.encode_batch(bulk)
            bulk_lat.append(time.perf_counter() - s)

    threads = [threading.Thread(target=ingest, daemon=True) for _ in range(a.bulk_threads)]
    for t in threads:
        t.start()
    time.sleep(0.2)
    lat = []
    end = time.perf_counter() + a.seconds
    i = 0
    while time.perf_counter() < end:
        s = time.perf_counter()
        mm.encode_single(queries[i % len(queries)])
        lat.append(time.perf_counter() - s)
        i += 1
        time.sleep(a.interval)
    stop.set()
    for t in threads:
        t.join()

    print(json.dumps({
        "lanes": a.lanes, "slice": a.slice if a.lanes else None, "batch": a.batch,
        "idle_p50_ms": round(1000 * statistics.median(idle), 1),
        "queries": len(lat),
        "p50_ms": round(1000 * statistics.median(lat), 1),
        "p95_ms": round(1000 * pct(lat, 0.95), 1),
        "max_ms": round(1000 * max(lat), 1),
        "over_400ms": sum(1 for x in lat if x > 0.4),
        "bulk_batches_done": len(bulk_lat),
        "bulk_batch_mean_s": round(statistics.mean(bulk_lat), 2) if bulk_lat else None,
    }))


if __name__ == "__main__":
    main()
