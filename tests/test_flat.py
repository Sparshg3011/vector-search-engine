import numpy as np
import pytest

from vecstore.flat import FlatIndex


rng = np.random.default_rng(7)


def test_finds_exact_nearest_neighbor():
    index = FlatIndex(dim=4)
    index.add([[0, 0, 0, 0], [10, 10, 10, 10], [1, 0, 0, 0]])
    ids, dists = index.search([0.9, 0, 0, 0], k=1)
    assert ids[0] == 2


def test_results_sorted_by_distance():
    index = FlatIndex(dim=32)
    index.add(rng.standard_normal((500, 32)))
    _, dists = index.search(rng.standard_normal(32), k=10)
    assert np.all(np.diff(dists) >= 0)


def test_k_larger_than_index():
    index = FlatIndex(dim=8)
    index.add(rng.standard_normal((3, 8)))
    ids, dists = index.search(rng.standard_normal(8), k=10)
    assert len(ids) == 3


def test_empty_index_returns_nothing():
    index = FlatIndex(dim=8)
    ids, dists = index.search(rng.standard_normal(8), k=5)
    assert len(ids) == 0


def test_dim_mismatch_raises():
    index = FlatIndex(dim=8)
    with pytest.raises(ValueError):
        index.add(rng.standard_normal((5, 16)))


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        FlatIndex(dim=8, metric="manhattan")


def test_cosine_metric_ranks_by_direction():
    index = FlatIndex(dim=2, metric="cosine")
    # same direction as query but huge magnitude should still win
    index.add([[1000.0, 1000.0], [0.1, 0.0]])
    ids, _ = index.search([1.0, 1.0], k=1)
    assert ids[0] == 0


def test_incremental_adds_accumulate():
    index = FlatIndex(dim=8)
    index.add(rng.standard_normal((10, 8)))
    index.add(rng.standard_normal((5, 8)))
    assert len(index) == 15
