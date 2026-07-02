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
