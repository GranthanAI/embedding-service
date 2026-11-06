"""CPU rerank benchmark: python scripts/bench_rerank.py MODEL [--passages 30] [--tokens 200] [--calls 40]

Run one model per process so peak RSS is attributable. Synthetic passages (no real data).
"""
import argparse
import ctypes
import json
import os
import random
import statistics
import sys
import time


def peak_rss_mb() -> float:
    if sys.platform == "win32":
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32, ps = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 1e6
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def make_passages(n: int, words: int, seed: int = 7):
    rnd = random.Random(seed)
    vocab = "the of and to in is that for it as was with be by on not he this are or his from at which but have an had they you were their one all we can her has there been if more when will would who so no".split()
    return [" ".join(rnd.choice(vocab) for _ in range(words)) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--passages", type=int, default=30)
    ap.add_argument("--words", type=int, default=200, help="words per passage (~200 tokens after wordpiece)")
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    from sentence_transformers import CrossEncoder

    t0 = time.perf_counter()
    model = CrossEncoder(a.model, device="cpu", max_length=a.max_length)
    load_s = time.perf_counter() - t0
    passages = make_passages(a.passages, a.words)
    tok = model.tokenizer
    avg_tokens = statistics.mean(len(tok(p)["input_ids"]) for p in passages)
    pairs = [("what is the main topic of this document", p) for p in passages]
    for _ in range(a.warmup):
        model.predict(pairs, batch_size=a.batch_size, show_progress_bar=False)
    lat = []
    for _ in range(a.calls):
        s = time.perf_counter()
        model.predict(pairs, batch_size=a.batch_size, show_progress_bar=False)
        lat.append(time.perf_counter() - s)
    lat.sort()
    p95 = lat[min(len(lat) - 1, int(round(0.95 * len(lat))) - 1)]
    print(json.dumps({
        "model": a.model, "passages": a.passages, "avg_passage_tokens": round(avg_tokens),
        "calls": a.calls, "load_s": round(load_s, 1),
        "p50_s": round(statistics.median(lat), 3), "p95_s": round(p95, 3),
        "peak_rss_mb": round(peak_rss_mb()), "torch_threads": __import__("torch").get_num_threads(),
    }))


if __name__ == "__main__":
    main()
