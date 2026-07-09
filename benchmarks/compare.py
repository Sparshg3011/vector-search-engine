"""Compare vecstore against faiss on an ann-benchmarks dataset.

Same data, same machine, same M and ef_construction, single thread,
median latency over the same queries, recall against the official
ground truth. Writes a json + png to results/.
"""

import argparse
import json
import os
import statistics
import time

import faiss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vecstore import HNSWIndex, recall
from vecstore.datasets import load

M = 16
EF_CONSTRUCTION = 100
EF_SWEEP = [10, 20, 50, 100, 200]
K = 10


def timed(search_fn, queries):
    """Median per-query latency in ms, after a small warmup."""
    for q in queries[:20]:
        search_fn(q)
    results, times = [], []
    for q in queries:
        t0 = time.perf_counter()
        results.append(search_fn(q))
        times.append(time.perf_counter() - t0)
    return results, statistics.median(times) * 1000


def mean_recall(truth, results):
    return float(np.mean([recall(t[:K], got) for t, got in zip(truth, results)]))


def normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def build_ours(train, dim, cache, metric):
    if os.path.exists(cache):
        print(f"loading cached index from {cache}")
        return HNSWIndex.load(cache)
    index = HNSWIndex(dim=dim, metric=metric, M=M, ef_construction=EF_CONSTRUCTION, seed=0)
    t0 = time.perf_counter()
    for i, v in enumerate(train):
        index.add(v)
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(train)}")
    print(f"vecstore build: {time.perf_counter() - t0:.0f}s")
    index.save(cache)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fashion-mnist-784-euclidean")
    parser.add_argument("--queries", type=int, default=500)
    args = parser.parse_args()

    train, queries, truth = load(args.dataset)
    queries, truth = queries[: args.queries], truth[: args.queries]
    dim = train.shape[1]
    faiss.omp_set_num_threads(1)

    # angular datasets rank by cosine — normalize and score by inner
    # product so recall isn't measured under the wrong metric
    angular = args.dataset.endswith("angular")
    if angular:
        train, queries = normalize(train), normalize(queries)
    metric = "ip" if angular else "l2"
    faiss_metric = faiss.METRIC_INNER_PRODUCT if angular else faiss.METRIC_L2

    cache = os.path.join("data", f"{args.dataset}-M{M}-ef{EF_CONSTRUCTION}-{metric}.npz")
    ours = build_ours(train, dim, cache, metric)

    theirs = faiss.IndexHNSWFlat(dim, M, faiss_metric)
    theirs.hnsw.efConstruction = EF_CONSTRUCTION
    t0 = time.perf_counter()
    theirs.add(train)
    print(f"faiss build: {time.perf_counter() - t0:.0f}s")

    flat = faiss.IndexFlatIP(dim) if angular else faiss.IndexFlatL2(dim)
    flat.add(train)
    _, flat_ms = timed(lambda q: flat.search(q[None], K)[1][0], queries)
    print(f"exact search: {flat_ms:.2f} ms/query")

    # one full untimed pass per index first — on 1M vectors the first
    # timed config otherwise pays all the page faults
    theirs.hnsw.efSearch = 50
    for q in queries:
        ours.search(q, k=K, ef=50)
        theirs.search(q[None], K)

    rows = []
    for ef in EF_SWEEP:
        results, ms = timed(lambda q: ours.search(q, k=K, ef=ef)[0], queries)
        rows.append(
            {"index": "vecstore", "ef": ef, "recall": mean_recall(truth, results), "ms": ms}
        )
        theirs.hnsw.efSearch = ef
        results, ms = timed(lambda q: theirs.search(q[None], K)[1][0], queries)
        rows.append(
            {"index": "faiss", "ef": ef, "recall": mean_recall(truth, results), "ms": ms}
        )
        print(f"ef={ef}: vecstore r={rows[-2]['recall']:.3f} {rows[-2]['ms']:.2f}ms"
              f" | faiss r={rows[-1]['recall']:.3f} {rows[-1]['ms']:.2f}ms")

    out = {
        "dataset": args.dataset,
        "n": len(train),
        "dim": dim,
        "M": M,
        "ef_construction": EF_CONSTRUCTION,
        "queries": len(queries),
        "flat_ms": flat_ms,
        "rows": rows,
    }
    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.dataset}.json", "w") as f:
        json.dump(out, f, indent=2)
    plot(out, f"results/{args.dataset}.png")


def plot(out, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, marker in [("vecstore", "o"), ("faiss", "s")]:
        pts = [(r["ms"], r["recall"]) for r in out["rows"] if r["index"] == name]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=marker, label=name)
    ax.axvline(out["flat_ms"], ls="--", c="gray")
    high = max(r["recall"] for r in out["rows"])
    ax.text(out["flat_ms"] * 0.92, high, f"exact search ({out['flat_ms']:.1f} ms)",
            rotation=90, va="top", ha="right", color="gray", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("median ms / query (log scale)")
    ax.set_ylabel("recall@10")
    ax.set_title(
        f"{out['dataset']}: {out['n']} vectors, dim {out['dim']}, M={out['M']}"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
