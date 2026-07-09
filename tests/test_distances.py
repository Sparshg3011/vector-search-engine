import numpy as np

from vecstore.distances import l2, inner_product, cosine


rng = np.random.default_rng(42)


def test_l2_matches_naive_loop():
    q = rng.standard_normal(16).astype(np.float32)
    vs = rng.standard_normal((100, 16)).astype(np.float32)
    expected = np.array([np.sum((v - q) ** 2) for v in vs])
    np.testing.assert_allclose(l2(q, vs), expected, rtol=1e-4)


def test_l2_is_zero_for_identical_vector():
    q = rng.standard_normal(8).astype(np.float32)
    dists = l2(q, q[None, :])
    assert dists[0] == 0.0


def test_inner_product_prefers_larger_dot():
    q = np.array([1.0, 0.0], dtype=np.float32)
    vs = np.array([[3.0, 0.0], [1.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
    dists = inner_product(q, vs)
    # bigger dot product should mean smaller "distance"
    assert dists[0] < dists[1] < dists[2]


def test_cosine_ignores_magnitude():
    q = np.array([1.0, 1.0], dtype=np.float32)
    vs = np.array([[2.0, 2.0], [100.0, 100.0]], dtype=np.float32)
    dists = cosine(q, vs)
    np.testing.assert_allclose(dists, [0.0, 0.0], atol=1e-6)


def test_l2_ranking_survives_large_magnitudes():
    # squaring these in float32 overflows to inf; the float64 accumulator
    # keeps the true order (row 1 is nearer the origin)
    q = np.array([0.0], dtype=np.float32)
    vs = np.array([[2.5e19], [2.0e19]], dtype=np.float32)
    dists = l2(q, vs)
    assert np.isfinite(dists).all()
    assert dists[1] < dists[0]


def test_cosine_orthogonal_and_opposite():
    q = np.array([1.0, 0.0], dtype=np.float32)
    vs = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    dists = cosine(q, vs)
    np.testing.assert_allclose(dists, [1.0, 2.0], atol=1e-6)
