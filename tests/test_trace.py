import numpy as np

from vecstore import HNSWIndex
from vecstore.trace import traced_search


rng = np.random.default_rng(21)


def build_2d(n=200):
    index = HNSWIndex(dim=2, M=8, ef_construction=64, seed=3)
    for v in rng.uniform(0, 100, (n, 2)):
        index.add(v)
    return index


def test_traced_results_match_the_real_search():
    index = build_2d()
    for q in rng.uniform(0, 100, (10, 2)):
        for ef in (5, 20, 50):
            ids, dists, _ = traced_search(index, q, k=5, ef=ef)
            real_ids, real_dists = index.search(q, k=5, ef=ef)
            assert ids == list(real_ids)
            np.testing.assert_allclose(dists, real_dists, rtol=1e-6)


def test_trace_covers_every_layer_top_down():
    index = build_2d()
    _, _, trace = traced_search(index, [50, 50], k=5, ef=20)
    layers = [step["layer"] for step in trace]
    assert layers == list(range(len(index._layers) - 1, -1, -1))


def test_hops_connect_and_stay_valid():
    index = build_2d()
    _, _, trace = traced_search(index, [30, 70], k=5, ef=20)
    for step in trace:
        for a, b in step["hops"]:
            assert 0 <= a < len(index) and 0 <= b < len(index)
        if step["layer"] > 0 and step["hops"]:
            # greedy hops chain: each hop starts where the last ended
            for prev, nxt in zip(step["hops"], step["hops"][1:]):
                assert prev[1] == nxt[0]


def test_visited_is_a_small_fraction_of_the_index():
    index = build_2d(n=400)
    _, _, trace = traced_search(index, [50, 50], k=5, ef=10)
    visited = {v for step in trace for v in step["visited"]}
    assert len(visited) < 150  # the whole point of the structure
