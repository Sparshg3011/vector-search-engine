import math
import random

import numpy as np
import pytest

from vecstore.hnsw import HNSWIndex, random_level


rng = np.random.default_rng(3)


def build_index(n=100, dim=8, **kwargs):
    index = HNSWIndex(dim=dim, **kwargs)
    for v in rng.standard_normal((n, dim)):
        index.add(v)
    return index


def test_most_nodes_stay_at_level_zero():
    rng = random.Random(0)
    m_l = 1 / math.log(16)
    levels = [random_level(m_l, rng) for _ in range(20000)]
    promoted = sum(1 for l in levels if l >= 1) / len(levels)
    # with M=16, about 1 in 16 nodes should climb above layer 0
    assert abs(promoted - 1 / 16) < 0.01
    assert max(levels) <= 6


def test_bigger_m_gives_flatter_hierarchy():
    rng = random.Random(1)
    tall = [random_level(1 / math.log(4), rng) for _ in range(5000)]
    rng = random.Random(1)
    flat = [random_level(1 / math.log(64), rng) for _ in range(5000)]
    assert sum(flat) < sum(tall)


def test_every_node_lives_in_layer_zero():
    index = build_index(n=50)
    assert len(index) == 50
    assert set(index._layers[0]) == set(range(50))


def test_upper_layers_are_subsets_of_lower_ones():
    index = build_index(n=300)
    for l in range(1, len(index._layers)):
        assert set(index._layers[l]) <= set(index._layers[l - 1])


def test_entry_point_sits_on_the_top_layer():
    index = build_index(n=300)
    assert index._entry in index._layers[-1]


def test_dim_mismatch_raises():
    index = HNSWIndex(dim=8)
    with pytest.raises(ValueError):
        index.add(np.zeros(16))


def test_greedy_search_walks_a_chain_toward_the_query():
    index = HNSWIndex(dim=1)
    for x in [0.0, 1.0, 2.0, 3.0]:
        index.add([x])
    index._layers[0] = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    best, dist = index._greedy_search(np.array([2.9], np.float32), start=0, layer=0)
    assert best == 3


def test_degrees_never_exceed_the_cap():
    index = build_index(n=300, M=8)
    for l, layer in enumerate(index._layers):
        cap = 16 if l == 0 else 8
        assert all(len(links) <= cap for links in layer.values())


def test_links_point_at_real_nodes_in_the_same_layer():
    index = build_index(n=200)
    for layer in index._layers:
        for node, links in layer.items():
            assert node not in links
            assert all(nb in layer for nb in links)


def test_every_node_is_reachable_from_the_entry_point():
    index = build_index(n=300)
    seen = {index._entry}
    frontier = [index._entry]
    while frontier:
        node = frontier.pop()
        for nb in index._layers[0][node]:
            if nb not in seen:
                seen.add(nb)
                frontier.append(nb)
    assert len(seen) == len(index)


def trap_graph():
    # A is closer to the query than B, so a pure greedy walk never
    # crosses B to reach C, the true nearest
    index = HNSWIndex(dim=2)
    index.add([0.0, 0.0])  # A
    index.add([3.0, 0.0])  # B
    index.add([0.0, 9.0])  # C
    index._layers[0] = {0: [1], 1: [0, 2], 2: [1]}
    return index, np.array([0.0, 10.0], np.float32)


def test_search_layer_escapes_the_greedy_trap():
    index, query = trap_graph()
    found = index._search_layer(query, entry=0, layer=0, ef=3)
    assert found[0][1] == 2


def test_search_layer_with_ef_one_degenerates_to_greedy():
    index, query = trap_graph()
    found = index._search_layer(query, entry=0, layer=0, ef=1)
    assert found[0][1] == 0


def test_greedy_search_stops_at_local_minimum():
    index, query = trap_graph()
    best, dist = index._greedy_search(query, start=0, layer=0)
    assert best == 0  # stuck, even though C is far closer
