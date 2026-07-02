import math
import random

from vecstore.hnsw import random_level


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
