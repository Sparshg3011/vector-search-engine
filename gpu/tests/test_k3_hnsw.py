import numpy as np
import pytest

from vecstore.eval import recall
from vecstore.flat import FlatIndex
from vecstore.hnsw import HNSWIndex

import vecstore_gpu

if not vecstore_gpu.is_available():
    pytest.skip(
        "_vecstore_gpu is not built here — this file only runs on a cuda machine",
        allow_module_level=True,
    )

pytestmark = pytest.mark.gpu

N = 20_000
DIM = 32
NQ = 200
K = 10


@pytest.fixture(scope="module")
def index():
    # 20k python-side inserts is the slow part of this file — build once
    # and share it across the whole module
    gen = np.random.default_rng(4)
    idx = HNSWIndex(dim=DIM, seed=0)
    for v in gen.standard_normal((N, DIM)).astype(np.float32):
        idx.add(v)
    return idx


@pytest.fixture(scope="module")
def gi(index, tmp_path_factory):
    # gaussian data is not angular, so l2 with no normalization
    path = str(tmp_path_factory.mktemp("k3") / "index.gpu.npz")
    vecstore_gpu.export_index(index, path)
    loaded = vecstore_gpu.load_gpu_index(path)
    loaded.validate()
    return loaded


@pytest.fixture(scope="module")
def device(gi):
    dev = vecstore_gpu.DeviceIndex(gi.vectors, gi.dim, gi.metric)
    dev.set_graph(gi.adjacency, gi.degrees, int(gi.entry))
    return dev


@pytest.fixture(scope="module")
def queries():
    return np.random.default_rng(17).standard_normal((NQ, DIM)).astype(np.float32)


@pytest.fixture(scope="module")
def entries(gi, queries):
    return vecstore_gpu.descend(gi, queries)


@pytest.fixture(scope="module")
def truth(gi, queries):
    # exact neighbors of the rows the device actually holds: both sides
    # of the gate get scored against the same data, per spec
    flat = FlatIndex(dim=gi.dim)
    flat.add(gi.vectors[:, : gi.dim].astype(np.float32))
    return [flat.search(q, k=K)[0] for q in queries]


def pad(queries, stride):
    out = np.zeros((len(queries), stride), dtype=np.float16)
    out[:, : queries.shape[1]] = queries
    return out


@pytest.mark.parametrize("ef", [50, 200])
def test_recall_tracks_the_cpu_index(index, gi, device, queries, entries, truth, ef):
    gpu_ids, _ = device.hnsw(pad(queries, gi.stride), entries, K, ef)
    gpu = np.mean([recall(t, ids) for t, ids in zip(truth, gpu_ids)])
    cpu = np.mean(
        [recall(t, index.search(q, k=K, ef=ef)[0]) for t, q in zip(truth, queries)]
    )
    # spec: a gap past 0.01 is a bug in the beam search, not "the gpu is
    # different". same ef, same graph, same entry points, same vectors
    assert abs(gpu - cpu) <= 0.01, f"ef={ef}: gpu recall {gpu:.4f} vs cpu {cpu:.4f}"


@pytest.mark.parametrize("ef", [50, 200])
def test_results_are_well_formed(gi, device, queries, entries, ef):
    ids, dists = device.hnsw(pad(queries, gi.stride), entries, K, ef)
    assert ids.shape == (NQ, K)
    assert dists.shape == (NQ, K)
    assert ids.dtype == np.int32
    assert dists.dtype == np.float32
    assert ids.min() >= 0 and ids.max() < len(gi.degrees)
    # the result list dedups by id — that is what makes the global
    # visited hash safe to collide
    assert all(len(set(row.tolist())) == K for row in ids)
    assert np.all(np.diff(dists, axis=1) >= 0)


def test_distances_belong_to_the_ids_returned(gi, device, queries, entries):
    ids, dists = device.hnsw(pad(queries[:16], gi.stride), entries[:16], K, 64)
    rows = gi.vectors[:, : gi.dim].astype(np.float32)
    for qi, q in enumerate(queries[:16]):
        diff = rows[ids[qi]] - q
        want = np.einsum("ij,ij->i", diff, diff)
        np.testing.assert_allclose(dists[qi], want, rtol=1e-3, atol=1e-2)


def test_k_above_ef_raises(gi, device, queries, entries):
    # a beam of width ef cannot hand back more than ef results
    with pytest.raises(ValueError):
        device.hnsw(pad(queries[:4], gi.stride), entries[:4], 32, 16)


def test_ef_above_the_cap_raises(gi, device, queries, entries):
    # MAX_EF is 256; the candidate list lives in shared memory and is
    # sized from it
    with pytest.raises(ValueError):
        device.hnsw(pad(queries[:4], gi.stride), entries[:4], K, 257)


def test_entries_must_cover_every_query(gi, device, queries, entries):
    with pytest.raises(ValueError):
        device.hnsw(pad(queries[:8], gi.stride), entries[:4], K, 50)


def test_hnsw_without_a_graph_raises(gi, queries, entries):
    bare = vecstore_gpu.DeviceIndex(gi.vectors, gi.dim, gi.metric)
    with pytest.raises((RuntimeError, ValueError)) as err:
        bare.hnsw(pad(queries[:4], gi.stride), entries[:4], K, 50)
    # the graph check belongs in the binding, ahead of the kernel launch —
    # a stub's "K3 not implemented" reaching here means it is missing
    assert "not implemented" not in str(err.value)
