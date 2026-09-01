import numpy as np
import pytest

import vecstore_gpu

if not vecstore_gpu.is_available():
    pytest.skip(
        "_vecstore_gpu is not built here — this file only runs on a cuda machine",
        allow_module_level=True,
    )

pytestmark = pytest.mark.gpu

rng = np.random.default_rng(2)


def assert_topk(dmat, k, ids, dists):
    """The spec's k2 gate. np.argpartition picks one arbitrary winner
    among tied distances and so does the kernel, so the id sets only have
    to agree away from the k-th value: anything sitting on the boundary
    within 1e-3 relative is an equally correct answer. Distances come
    back ascending and have to be the matrix's own values."""
    assert ids.shape == (len(dmat), k)
    assert dists.shape == (len(dmat), k)
    assert ids.dtype == np.int32
    assert dists.dtype == np.float32
    for row in range(len(dmat)):
        d = dmat[row]
        got = [int(i) for i in ids[row]]
        assert len(set(got)) == k, f"row {row}: repeated ids {got}"
        assert min(got) >= 0 and max(got) < d.shape[0], f"row {row}: id out of range"
        # k2 only selects, it never recomputes — the values must come
        # back bit-identical to the ones it was handed
        np.testing.assert_array_equal(
            dists[row], d[got], err_msg=f"row {row}: dists != dmat[ids]"
        )
        assert np.all(np.diff(dists[row]) >= 0), f"row {row}: not ascending"
        want = set(np.argpartition(d, k - 1)[:k].tolist())
        cut = float(np.sort(d)[k - 1])
        # relative to the boundary value, floored so a cut near zero
        # doesn't demand bit-exactness
        tol = 1e-3 * max(abs(cut), 1.0)
        for i in set(got) ^ want:
            assert abs(float(d[i]) - cut) <= tol, (
                f"row {row}: id {i} at distance {d[i]} is not a boundary tie "
                f"with the k-th distance {cut}"
            )


def test_matches_argpartition():
    dmat = rng.standard_normal((16, 4000)).astype(np.float32)
    ids, dists = vecstore_gpu.topk(dmat, 10)
    assert_topk(dmat, 10, ids, dists)


def test_prime_ish_row_length():
    # 1013 columns leaves a ragged tail for whatever tile width k2 uses
    dmat = rng.standard_normal((5, 1013)).astype(np.float32)
    ids, dists = vecstore_gpu.topk(dmat, 32)
    assert_topk(dmat, 32, ids, dists)


def test_k_of_one():
    dmat = rng.standard_normal((64, 777)).astype(np.float32)
    ids, dists = vecstore_gpu.topk(dmat, 1)
    assert_topk(dmat, 1, ids, dists)
    np.testing.assert_array_equal(dists[:, 0], dmat.min(axis=1))


def test_k_at_the_cap():
    # 128 is MAX_K exactly — the last legal k, so the per-thread lists
    # are full
    dmat = rng.standard_normal((8, 5000)).astype(np.float32)
    ids, dists = vecstore_gpu.topk(dmat, 128)
    assert_topk(dmat, 128, ids, dists)


def test_k_above_the_cap_raises():
    dmat = rng.standard_normal((4, 5000)).astype(np.float32)
    with pytest.raises(ValueError):
        vecstore_gpu.topk(dmat, 129)


def test_heavy_duplicate_values():
    # five distinct values across 600 columns: the k-th distance is tied
    # dozens of times over, so the boundary rule is doing real work here
    dmat = rng.integers(0, 5, (12, 600)).astype(np.float32)
    ids, dists = vecstore_gpu.topk(dmat, 10)
    assert_topk(dmat, 10, ids, dists)
    # every winner has to be a minimum when there are that many of them
    assert np.all(dists == dmat.min(axis=1, keepdims=True))


def test_all_values_identical():
    dmat = np.full((4, 300), 2.5, dtype=np.float32)
    ids, dists = vecstore_gpu.topk(dmat, 16)
    assert_topk(dmat, 16, ids, dists)


def test_k_larger_than_the_row_raises():
    # the output is a fixed (nq, k) block, so there is nothing sane to
    # return when the row is shorter than k — clamping would lie about
    # the shape
    dmat = rng.standard_normal((4, 8)).astype(np.float32)
    with pytest.raises(ValueError):
        vecstore_gpu.topk(dmat, 16)


def test_non_float32_matrix_is_rejected():
    # k2 takes the fp32 distance matrix k1 produced; a silent downcast of
    # fp64 would quietly change which ids win at the boundary
    dmat = rng.standard_normal((4, 500))
    with pytest.raises((TypeError, ValueError)):
        vecstore_gpu.topk(dmat, 10)
