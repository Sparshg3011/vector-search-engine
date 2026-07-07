import heapq

import numpy as np


def traced_search(index, query, k=10, ef=50):
    """The same walk index.search does, with a flight recorder taped
    on. Lives apart so the measured search path stays lean — the demo
    pays for its own bookkeeping. Returns (ids, dists, trace); trace
    holds one step per layer with every hop taken."""
    query = np.asarray(query, dtype=np.float32)
    if index._entry is None:
        return [], [], []

    trace = []
    entry = int(index._entry)
    for layer in range(len(index._layers) - 1, 0, -1):
        hops = []
        best = entry
        best_dist = float(index._dist(query, index._vectors[best : best + 1])[0])
        while True:
            neighbors = index._layers[layer][best]
            if not neighbors:
                break
            dists = index._dist(query, index._vectors[neighbors])
            i = int(np.argmin(dists))
            if dists[i] >= best_dist:
                break
            hops.append([best, int(neighbors[i])])
            best, best_dist = int(neighbors[i]), float(dists[i])
        trace.append(
            {"layer": layer, "entry": entry, "hops": hops,
             "visited": [entry] + [h[1] for h in hops]}
        )
        entry = best

    ef = max(ef, k)
    d0 = float(index._dist(query, index._vectors[entry : entry + 1])[0])
    candidates = [(d0, entry)]
    results = [(-d0, entry)]
    visited = {entry}
    hops = []
    while candidates:
        d, node = heapq.heappop(candidates)
        if d > -results[0][0]:
            break
        fresh = [n for n in index._layers[0][node] if n not in visited]
        if not fresh:
            continue
        visited.update(fresh)
        dists = index._dist(query, index._vectors[fresh])
        for nd, nb in zip(dists, fresh):
            if len(results) < ef or nd < -results[0][0]:
                hops.append([int(node), int(nb)])
                heapq.heappush(candidates, (float(nd), int(nb)))
                heapq.heappush(results, (-float(nd), int(nb)))
                if len(results) > ef:
                    heapq.heappop(results)
    trace.append(
        {"layer": 0, "entry": entry, "hops": hops,
         "visited": sorted(int(v) for v in visited)}
    )

    found = sorted((-d, n) for d, n in results)[:k]
    ids = [int(n) for _, n in found]
    return ids, [float(d) for d, _ in found], trace
