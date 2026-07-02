import numpy as np

from .distances import METRICS


class FlatIndex:
    """Exact brute-force search. Slow but always right — this is the
    ground truth that HNSW recall gets measured against."""

    def __init__(self, dim, metric="l2"):
        if metric not in METRICS:
            raise ValueError(f"unknown metric: {metric}")
        self.dim = dim
        self.metric = metric
        self._dist = METRICS[metric]
        self._vectors = np.empty((0, dim), dtype=np.float32)

    def __len__(self):
        return len(self._vectors)

    def add(self, vectors):
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        self._vectors = np.vstack([self._vectors, vectors])

    def search(self, query, k=10):
        """Returns (ids, distances) of the k nearest vectors, closest first."""
        if len(self._vectors) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        query = np.asarray(query, dtype=np.float32)
        dists = self._dist(query, self._vectors)
        k = min(k, len(dists))
        # argpartition is O(n) vs O(n log n) for a full sort, we only
        # sort the k winners
        ids = np.argpartition(dists, k - 1)[:k]
        ids = ids[np.argsort(dists[ids])]
        return ids, dists[ids]
