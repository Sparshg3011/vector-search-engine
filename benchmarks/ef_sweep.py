"""Sweep ef and print the recall/latency tradeoff, with brute force as
the reference point."""

import time

import numpy as np

from vecstore import FlatIndex, HNSWIndex, recall


def main(n=10000, dim=32, n_queries=100, k=10):
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

    flat = FlatIndex(dim=dim)
    flat.add(vectors)

    index = HNSWIndex(dim=dim, seed=0)
    t0 = time.perf_counter()
    for v in vectors:
        index.add(v)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    truth = [flat.search(q, k=k)[0] for q in queries]
    flat_ms = (time.perf_counter() - t0) / n_queries * 1000

    print(f"{n} vectors, dim={dim}, k={k}, build took {build_s:.1f}s")
    print(f"brute force: {flat_ms:.3f} ms/query, recall 1.0 by definition")
    print(f"{'ef':>4}  {'recall@10':>9}  {'ms/query':>8}")
    for ef in [10, 20, 50, 100, 200]:
        t0 = time.perf_counter()
        results = [index.search(q, k=k, ef=ef)[0] for q in queries]
        ms = (time.perf_counter() - t0) / n_queries * 1000
        mean = sum(recall(t, g) for t, g in zip(truth, results)) / n_queries
        print(f"{ef:>4}  {mean:>9.3f}  {ms:>8.3f}")


if __name__ == "__main__":
    main()
