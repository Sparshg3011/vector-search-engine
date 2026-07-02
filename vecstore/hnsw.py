import math
import random

import numpy as np

from .distances import METRICS


def random_level(m_l, rng):
    """Coin-flip layer assignment: most nodes stay at layer 0, each
    level up is ~M times rarer. This is what makes the graph a
    hierarchy without any explicit balancing."""
    return int(-math.log(1.0 - rng.random()) * m_l)


class HNSWIndex:
    """Hierarchical Navigable Small World graph, built one insert at a
    time. M is the max links per node — more links means better recall
    but more memory and slower inserts."""

    def __init__(self, dim, metric="l2", M=16, seed=0):
        if metric not in METRICS:
            raise ValueError(f"unknown metric: {metric}")
        self.dim = dim
        self.metric = metric
        self.M = M
        self._dist = METRICS[metric]
        self._ml = 1.0 / math.log(M)
        self._rng = random.Random(seed)
        self._vectors = np.empty((0, dim), dtype=np.float32)
        # one adjacency dict per layer: {node id: [neighbor ids]}
        self._layers = []
        self._entry = None

    def __len__(self):
        return len(self._vectors)

    def add(self, vector):
        """Insert one vector. Construction is sequential by nature:
        each insert navigates the graph built so far."""
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vector.shape[1]}")
        node = len(self._vectors)
        self._vectors = np.vstack([self._vectors, vector])

        level = random_level(self._ml, self._rng)
        prev_max = len(self._layers) - 1
        while len(self._layers) <= level:
            self._layers.append({})
        for l in range(level + 1):
            self._layers[l][node] = []

        if self._entry is None or level > prev_max:
            self._entry = node
        return node

    def _greedy_search(self, query, start, layer):
        """Hop to whichever neighbor is closest to the query, repeat
        until no neighbor improves. Every move strictly shrinks the
        distance, so this always terminates — but it can stop at a
        local minimum, which is why search keeps a candidate list at
        layer 0 instead of relying on this alone."""
        best = start
        best_dist = self._dist(query, self._vectors[best : best + 1])[0]
        while True:
            neighbors = self._layers[layer][best]
            if not neighbors:
                return best, best_dist
            dists = self._dist(query, self._vectors[neighbors])
            i = int(np.argmin(dists))
            if dists[i] >= best_dist:
                return best, best_dist
            best, best_dist = neighbors[i], dists[i]
