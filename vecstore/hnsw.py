import heapq
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
        self._vectors = np.empty((0, dim), dtype=np.float32)
        # one adjacency dict per layer: {node id: [neighbor ids]}
        self._layers = []
        self._entry = None

    def __len__(self):
        return len(self._vectors)

    def add(self, vector):
        """Insert one vector: zoom down to its level with greedy hops,
        then at each of its layers find the ef_construction closest
        nodes and link up with the best M of them."""
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vector.shape[1]}")
        node = len(self._vectors)
        self._vectors = np.vstack([self._vectors, vector])

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
                neighbors = [n for _, n in found[: self.M]]
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

    def _prune(self, node, layer):
        """Cap a node's links, keeping the closest. Layer 0 gets twice
        the budget — that's where the fine-grained search happens."""
        cap = 2 * self.M if layer == 0 else self.M
        links = self._layers[layer][node]
        if len(links) <= cap:
            return
        dists = self._dist(self._vectors[node], self._vectors[links])
        keep = np.argsort(dists)[:cap]
        self._layers[layer][node] = [links[i] for i in keep]

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
