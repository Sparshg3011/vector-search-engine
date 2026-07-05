# vector-search-engine

Approximate nearest neighbor search engine built from scratch — the HNSW graph
index implemented in Python/NumPy and benchmarked against FAISS on standard ANN
datasets. No search library does the searching: the layered graph, the insert
heuristic and the query algorithm all live in this repo, in readable NumPy.

## Results

Fashion-MNIST from ann-benchmarks (60,000 vectors, 784 dims, official ground
truth). Both indexes built with M=16, ef_construction=100; single thread;
median latency over 500 queries.

| ef | recall@10 (vecstore) | recall@10 (faiss) | ms/query (vecstore) | ms/query (faiss) |
|---:|---:|---:|---:|---:|
| 10 | 0.932 | 0.932 | 0.19 | 0.03 |
| 20 | 0.978 | 0.980 | 0.25 | 0.04 |
| 50 | 0.996 | 0.995 | 0.47 | 0.08 |
| 100 | 0.998 | 0.998 | 0.78 | 0.13 |
| 200 | 0.999 | 1.000 | 1.36 | 0.23 |

![recall vs latency](results/fashion-mnist-784-euclidean.png)

Exact search costs 2.6 ms/query on this machine. Reading the curve honestly:
this implementation matches FAISS on recall point for point (same algorithm,
same parameters), answers at 99.6% recall ~5x faster than exact search, and
sits ~6x behind FAISS's C++ on raw latency.

Reproduce with `python benchmarks/compare.py` (downloads the dataset on first
run).

## Status

Work in progress — web playground and more datasets on the way.
