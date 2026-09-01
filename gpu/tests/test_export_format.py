import numpy as np
import pytest

from vecstore.hnsw import HNSWIndex
from vecstore_gpu import descend, export_index, load_gpu_index


rng = np.random.default_rng(11)


def build_index(n=300, dim=20, M=6, metric="l2", seed=11):
    # dim 20 rounds up to stride 24, so every case here carries four pad
    # columns; M=6 gives a layer-0 cap of 12
    gen = np.random.default_rng(seed)
    index = HNSWIndex(dim=dim, metric=metric, M=M, seed=0)
    for v in gen.standard_normal((n, dim)):
        index.add(v)
    return index


@pytest.fixture(scope="module")
def index():
    return build_index()


@pytest.fixture(scope="module")
def exported(index, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("export") / "index.gpu.npz")
    export_index(index, path)
    return path


@pytest.fixture
def gi(exported):
    # reloaded per test: the validate() cases mutate what they get
    return load_gpu_index(exported)


def test_stride_rounds_up_to_a_multiple_of_eight(gi):
    assert gi.dim == 20
    assert gi.stride == 24
    assert gi.stride % 8 == 0
    # smallest legal stride, not just any multiple of 8
    assert gi.stride - gi.dim < 8


def test_padding_columns_are_zero(gi):
    assert gi.vectors.shape == (300, gi.stride)
    assert np.all(gi.vectors[:, gi.dim :] == 0)


def test_vectors_are_the_fp16_cast_of_the_source_rows(index, gi):
    np.testing.assert_array_equal(gi.vectors[:, : gi.dim], index.vectors.astype(np.float16))


def test_dtypes_match_the_spec_table(exported):
    # the loader unwraps scalars to python types, so the dtype contract
    # can only be checked on the raw npz
    raw = np.load(exported, allow_pickle=False)
    assert raw["vectors"].dtype == np.float16
    assert raw["adjacency"].dtype == np.int32
    assert raw["degrees"].dtype == np.int32
    for key in ("dim", "stride", "M", "entry"):
        assert raw[key].dtype == np.int64, key
        assert raw[key].shape == (), key
    assert raw["normalized"].dtype == np.bool_
    assert raw["metric"].dtype.kind == "U"
    assert raw["upper_layers"].dtype.kind == "U"
    assert set(raw.files) == {
        "vectors",
        "dim",
        "stride",
        "adjacency",
        "degrees",
        "M",
        "entry",
        "metric",
        "normalized",
        "upper_layers",
    }


def test_adjacency_width_is_exactly_two_m(index, gi):
    assert gi.M == index.M
    assert gi.adjacency.shape == (300, 2 * index.M)


def test_layer_zero_degree_never_exceeds_two_m(gi):
    assert gi.degrees.min() >= 0
    assert gi.degrees.max() <= 2 * gi.M


def test_adjacency_padding_is_minus_one_past_each_degree(index, gi):
    for node, nbrs in index._layers[0].items():
        assert gi.degrees[node] == len(nbrs)
        assert list(gi.adjacency[node, : len(nbrs)]) == nbrs
        assert np.all(gi.adjacency[node, len(nbrs) :] == -1)
    # -1 marks padding and nothing else
    live = np.arange(gi.adjacency.shape[1])[None, :] < gi.degrees[:, None]
    assert np.all(gi.adjacency[live] >= 0)
    assert np.all(gi.adjacency[~live] == -1)


def test_neighbor_ids_stay_in_range(gi):
    live = np.arange(gi.adjacency.shape[1])[None, :] < gi.degrees[:, None]
    ids = gi.adjacency[live]
    assert ids.min() >= 0
    assert ids.max() < len(gi.degrees)


def test_upper_layers_round_trip(index, gi):
    # layer 0 belongs to the gpu; the json carries layers 1..top
    assert len(gi.upper_layers) == len(index._layers) - 1
    assert gi.upper_layers == index._layers[1:]


def test_entry_is_preserved(index, gi):
    assert gi.entry == index._entry
    assert gi.entry in gi.upper_layers[-1]


def test_metric_and_normalized_default_to_l2_unnormalized(gi):
    assert gi.metric == "l2"
    assert gi.normalized is False


def test_exported_index_validates(gi):
    gi.validate()


def corrupt_stride(gi):
    # not a multiple of 8, so rows stop being 16-byte aligned
    gi.stride = 20


def corrupt_stride_below_dim(gi):
    gi.stride = 16


def corrupt_adjacency_width(gi):
    gi.M = gi.M + 1


def corrupt_entry(gi):
    gi.entry = len(gi.degrees)


def corrupt_degree_overflow(gi):
    gi.degrees = gi.degrees.copy()
    gi.degrees[0] = 2 * gi.M + 1


def corrupt_hole_in_live_prefix(gi):
    gi.adjacency = gi.adjacency.copy()
    gi.adjacency[int(np.argmax(gi.degrees)), 0] = -1


def corrupt_live_id_past_degree(gi):
    gi.adjacency = gi.adjacency.copy()
    node = int(np.flatnonzero(gi.degrees < gi.adjacency.shape[1])[0])
    gi.adjacency[node, -1] = 0


def corrupt_neighbor_id(gi):
    gi.adjacency = gi.adjacency.copy()
    gi.adjacency[int(np.argmax(gi.degrees)), 0] = len(gi.degrees)


def corrupt_degrees_length(gi):
    gi.degrees = gi.degrees[:-1]


def corrupt_padding_column(gi):
    gi.vectors = gi.vectors.copy()
    gi.vectors[3, gi.dim] = 1.0


@pytest.mark.parametrize(
    "mutate",
    [
        corrupt_stride,
        corrupt_stride_below_dim,
        corrupt_adjacency_width,
        corrupt_entry,
        corrupt_degree_overflow,
        corrupt_hole_in_live_prefix,
        corrupt_live_id_past_degree,
        corrupt_neighbor_id,
        corrupt_degrees_length,
        corrupt_padding_column,
    ],
)
def test_validate_catches_corruption(gi, mutate):
    # the gpu reads out of bounds instead of raising, so every one of
    # these has to be caught on the host
    mutate(gi)
    with pytest.raises(ValueError):
        gi.validate()


def test_normalize_makes_rows_unit_norm(tmp_path):
    index = build_index()
    path = str(tmp_path / "normalized.gpu.npz")
    export_index(index, path, normalize=True)
    gi = load_gpu_index(path)
    gi.validate()
    norms = np.linalg.norm(gi.vectors[:, : gi.dim].astype(np.float32), axis=1)
    # normalization runs in fp32 before the cast, so only fp16 rounding
    # separates these from 1.0
    np.testing.assert_allclose(norms, 1.0, atol=2e-3)
    assert gi.normalized is True
    assert np.all(gi.vectors[:, gi.dim :] == 0)


def test_cosine_exports_as_ip_with_normalized_rows(tmp_path):
    index = build_index(metric="cosine")
    path = str(tmp_path / "cosine.gpu.npz")
    export_index(index, path, normalize=True)
    gi = load_gpu_index(path)
    # no cosine on the device: unit rows make ip rank identically
    assert gi.metric == "ip"
    assert gi.normalized is True


def test_cosine_without_normalize_raises(tmp_path):
    index = build_index(metric="cosine")
    with pytest.raises(ValueError):
        export_index(index, str(tmp_path / "cosine.gpu.npz"), normalize=False)


def test_live_index_and_saved_path_export_identically(index, tmp_path):
    saved = str(tmp_path / "index.npz")
    index.save(saved)
    from_live = str(tmp_path / "live.gpu.npz")
    from_path = str(tmp_path / "path.gpu.npz")
    export_index(index, from_live)
    export_index(saved, from_path)

    a = np.load(from_live, allow_pickle=False)
    b = np.load(from_path, allow_pickle=False)
    assert set(a.files) == set(b.files)
    for key in a.files:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def mirror_of(gi):
    """A cpu HNSWIndex holding the exported rows and the exported graph:
    same vectors the kernels read, same links, so _greedy_search here is
    an independent replay of what descend() has to reproduce."""
    mirror = HNSWIndex(dim=gi.dim, metric=gi.metric, M=gi.M)
    mirror._vectors = gi.vectors[:, : gi.dim].astype(np.float32)
    mirror._size = len(mirror._vectors)
    layer0 = {
        node: gi.adjacency[node, : gi.degrees[node]].tolist()
        for node in range(len(gi.degrees))
    }
    mirror._layers = [layer0] + [dict(layer) for layer in gi.upper_layers]
    mirror._entry = int(gi.entry)
    return mirror


def test_descend_replays_the_cpu_upper_layer_walk(gi):
    queries = np.random.default_rng(23).standard_normal((20, gi.dim)).astype(np.float32)
    entries = descend(gi, queries)
    assert entries.dtype == np.int32
    assert entries.shape == (20,)

    mirror = mirror_of(gi)
    for qi, q in enumerate(queries):
        # HNSWIndex.search's descent verbatim: layers top..1, layer 0 is
        # the gpu's job
        node = mirror._entry
        for layer in range(len(mirror._layers) - 1, 0, -1):
            node, _ = mirror._greedy_search(q, node, layer)
        assert entries[qi] == node


def test_descend_accepts_padded_queries(gi):
    queries = np.random.default_rng(23).standard_normal((20, gi.dim)).astype(np.float32)
    padded = np.zeros((20, gi.stride), dtype=np.float32)
    padded[:, : gi.dim] = queries
    np.testing.assert_array_equal(descend(gi, padded), descend(gi, queries))


def test_descend_rejects_a_wrong_width(gi):
    with pytest.raises(ValueError):
        descend(gi, np.zeros((2, gi.dim + 1), dtype=np.float32))


def test_export_of_an_empty_index_raises(tmp_path):
    index = HNSWIndex(dim=8)
    with pytest.raises(ValueError):
        export_index(index, str(tmp_path / "empty.gpu.npz"))
