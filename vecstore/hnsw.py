import heapq
import json
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

    def __init__(self, dim, metric="l2", M=16, ef_construction=100, seed=0):
        if metric not in METRICS:
            raise ValueError(f"unknown metric: {metric}")
        self.dim = dim
        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction
        self._dist = METRICS[metric]
        self._ml = 1.0 / math.log(M)
        self._rng = random.Random(seed)
        # preallocated buffer, doubled when full — appending with
        # vstack would copy everything on every single insert
        self._vectors = np.empty((0, dim), dtype=np.float32)
        self._size = 0
        # one adjacency dict per layer: {node id: [neighbor ids]}
        self._layers = []
        self._entry = None

    def __len__(self):
        return self._size

    @property
    def vectors(self):
        """View of the live rows, without the spare buffer capacity."""
        return self._vectors[: self._size]

    def add(self, vector):
        """Insert one vector: zoom down to its level with greedy hops,
        then at each of its layers find the ef_construction closest
        nodes and link up with the best M of them."""
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vector.shape[1]}")
        node = self._size
        if self._size == len(self._vectors):
            grown = np.empty(
                (max(64, 2 * len(self._vectors)), self.dim), dtype=np.float32
            )
            grown[: self._size] = self._vectors[: self._size]
            self._vectors = grown
        self._vectors[node] = vector
        self._size += 1

        level = random_level(self._ml, self._rng)
        prev_entry = self._entry
        prev_max = len(self._layers) - 1
        while len(self._layers) <= level:
            self._layers.append({})
        for l in range(level + 1):
            self._layers[l][node] = []

        if prev_entry is not None:
            query = self._vectors[node]
            entry = prev_entry
            for l in range(prev_max, level, -1):
                entry, _ = self._greedy_search(query, entry, l)
            for l in range(min(level, prev_max), -1, -1):
                found = self._search_layer(query, entry, l, self.ef_construction)
                neighbors = self._select_neighbors(query, found, self.M)
                self._layers[l][node] = neighbors
                # links go both ways, and the receiving node may now
                # have too many — trim it back
                for nb in neighbors:
                    self._layers[l][nb].append(node)
                    self._prune(nb, l)
                entry = found[0][1]

        if prev_entry is None or level > prev_max:
            self._entry = node
        return node

    def save(self, path):
        """One .npz file: vectors as a real array, graph and params as
        json. No pickle — the file stays portable and safe to open."""
        meta = {
            "dim": self.dim,
            "metric": self.metric,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "entry": self._entry,
            "layers": [
                {str(n): nbrs for n, nbrs in layer.items()} for layer in self._layers
            ],
        }
        np.savez_compressed(path, vectors=self.vectors, meta=json.dumps(meta))

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(data["meta"].item())
        index = cls(
            dim=meta["dim"],
            metric=meta["metric"],
            M=meta["M"],
            ef_construction=meta["ef_construction"],
        )
        index._vectors = data["vectors"].copy()
        index._size = len(index._vectors)
        index._entry = meta["entry"]
        index._layers = [
            {int(n): nbrs for n, nbrs in layer.items()} for layer in meta["layers"]
        ]
        return index

    def _select_neighbors(self, query, candidates, M):
        """Diversity heuristic from the paper. Walk candidates nearest
        first and keep one only if it's closer to the query than to
        everyone already kept — i.e. it opens a new direction. A pile
        of candidates in one cluster collapses to a single link."""
        chosen = []
        for d, node in candidates:
            if len(chosen) == M:
                break
            if chosen:
                to_chosen = self._dist(self._vectors[node], self._vectors[chosen])
                if to_chosen.min() <= d:
                    continue
            chosen.append(node)
        return chosen

    def _prune(self, node, layer):
        """Re-select a node's links when it has too many. Layer 0 gets
        twice the budget — that's where the fine-grained search happens."""
        cap = 2 * self.M if layer == 0 else self.M
        links = self._layers[layer][node]
        if len(links) <= cap:
            return
        dists = self._dist(self._vectors[node], self._vectors[links])
        candidates = sorted(zip(dists, links))
        self._layers[layer][node] = self._select_neighbors(
            self._vectors[node], candidates, cap
        )

    def search(self, query, k=10, ef=50):
        """Greedy-descend the upper layers, then run one careful
        ef-wide search on layer 0 and return the k best (ids, dists)."""
        if self._entry is None:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        query = np.asarray(query, dtype=np.float32)
        entry = self._entry
        for l in range(len(self._layers) - 1, 0, -1):
            entry, _ = self._greedy_search(query, entry, l)
        found = self._search_layer(query, entry, 0, max(ef, k))[:k]
        ids = np.array([n for _, n in found], dtype=np.int64)
        dists = np.array([d for d, _ in found], dtype=np.float32)
        return ids, dists

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

    def _search_layer(self, query, entry, layer, ef):
        """Best-first search over one layer. Two heaps: candidates to
        expand (closest first) and the ef best results so far (worst
        first, so it can be evicted in O(log ef)). Python heaps are
        min-only, so result distances are negated. Stops once the
        closest unexpanded candidate is worse than the worst result —
        expanding it could only add nodes we'd throw away."""
        d = self._dist(query, self._vectors[entry : entry + 1])[0]
        candidates = [(d, entry)]
        results = [(-d, entry)]
        visited = {entry}
        while candidates:
            d, node = heapq.heappop(candidates)
            if d > -results[0][0]:
                break
            fresh = [n for n in self._layers[layer][node] if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            dists = self._dist(query, self._vectors[fresh])
            for nd, nb in zip(dists, fresh):
                if len(results) < ef or nd < -results[0][0]:
                    heapq.heappush(candidates, (nd, nb))
                    heapq.heappush(results, (-nd, nb))
                    if len(results) > ef:
                        heapq.heappop(results)
        return sorted((-d, n) for d, n in results)
