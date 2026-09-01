import numpy as np
import pytest

import vecstore_gpu

if not vecstore_gpu.is_available():
    pytest.skip(
        "_vecstore_gpu is not built here — this file only runs on a cuda machine",
        allow_module_level=True,
    )

pytestmark = pytest.mark.gpu

rng = np.random.default_rng(1)

NQ = 7
N = 1013

# both k1 paths answer the same contract, so every case runs twice
PATHS = ["distances", "distances_cublas"]


def quantize(data):
    """fp32 rows -> the padded fp16 block the device stores. Pad columns
    are zero, so they contribute nothing to either metric and kernels may
    loop to stride."""
    n, dim = data.shape
    stride = (dim + 7) // 8 * 8
    out = np.zeros((n, stride), dtype=np.float16)
    out[:, :dim] = data.astype(np.float16)
    return out


def reference(queries, vectors, metric):
    """fp64 numpy over the *already quantized* rows: the fp16 error is
    baked into the inputs both sides see, so what the tolerance covers is
    kernel error alone."""
    q = queries.astype(np.float64)
    x = vectors.astype(np.float64)
    if metric == "l2":
        diff = q[:, None, :] - x[None, :, :]
        return np.einsum("qnd,qnd->qn", diff, diff)
    # negated so lower = closer, the convention descend() and vecstore's
    # inner_product already use
    return -(q @ x.T)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("dim", [37, 5, 32])
def test_distances_match_fp64_numpy(path, metric, dim):
    # 37 is prime-ish and pads to 40; 5 is smaller than the 8-value pad
    # quantum; 32 is the case where stride == dim and there is no padding
    vectors = quantize(rng.standard_normal((N, dim)).astype(np.float32))
    queries = quantize(rng.standard_normal((NQ, dim)).astype(np.float32))
    device = vecstore_gpu.DeviceIndex(vectors, dim, metric)
    got = getattr(device, path)(queries)
    assert got.shape == (NQ, N)
    assert got.dtype == np.float32
    np.testing.assert_allclose(
        got, reference(queries, vectors, metric), rtol=1e-3, atol=1e-2
    )


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_sift_scale_sums_stay_in_fp32(path, metric):
    # integers 0..255 are exact in fp16, so the inputs carry no error at
    # all — but squared-l2 sums over 128 dims reach ~8.3e6, way past
    # fp16's 65504 max. accumulate in half and this comes back inf or
    # thousands of ulps off; the tolerance is tighter than the spec gate
    # because exact inputs leave nothing else to blame
    dim = 128
    raw = rng.integers(0, 256, (N, dim)).astype(np.float32)
    vectors = quantize(raw)
    queries = quantize(rng.integers(0, 256, (NQ, dim)).astype(np.float32))
    # integers below 2048 round-trip through fp16 untouched
    np.testing.assert_array_equal(vectors[:, :dim].astype(np.float32), raw)
    device = vecstore_gpu.DeviceIndex(vectors, dim, metric)
    got = getattr(device, path)(queries)
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(
        got, reference(queries, vectors, metric), rtol=1e-4, atol=1e-2
    )


@pytest.mark.parametrize("path", PATHS)
def test_l2_of_a_row_against_itself_is_zero(path):
    dim = 37
    vectors = quantize(rng.standard_normal((N, dim)).astype(np.float32))
    device = vecstore_gpu.DeviceIndex(vectors, dim, "l2")
    # the cublas path expands to |q|^2+|x|^2-2qx, where the diagonal is
    # exactly the cancellation it is worst at
    got = getattr(device, path)(vectors[:NQ].copy())
    assert np.all(np.abs(np.diag(got[:, :NQ])) < 1e-2)


@pytest.mark.parametrize("path", PATHS)
def test_single_query_keeps_its_row_shape(path):
    dim = 37
    vectors = quantize(rng.standard_normal((N, dim)).astype(np.float32))
    queries = quantize(rng.standard_normal((1, dim)).astype(np.float32))
    device = vecstore_gpu.DeviceIndex(vectors, dim, "l2")
    got = getattr(device, path)(queries)
    assert got.shape == (1, N)
    np.testing.assert_allclose(
        got, reference(queries, vectors, "l2"), rtol=1e-3, atol=1e-2
    )


def test_unknown_metric_raises():
    vectors = quantize(rng.standard_normal((16, 32)).astype(np.float32))
    with pytest.raises(ValueError):
        vecstore_gpu.DeviceIndex(vectors, 32, "cosine")


@pytest.mark.parametrize("path", PATHS)
def test_query_width_must_match_the_stride(path):
    dim = 37
    vectors = quantize(rng.standard_normal((N, dim)).astype(np.float32))
    device = vecstore_gpu.DeviceIndex(vectors, dim, "l2")
    # unpadded queries would make the kernel stride into the wrong row
    with pytest.raises(ValueError):
        getattr(device, path)(np.zeros((NQ, dim), dtype=np.float16))


@pytest.mark.parametrize("path", PATHS)
def test_fp32_queries_are_rejected(path):
    dim = 32
    vectors = quantize(rng.standard_normal((N, dim)).astype(np.float32))
    device = vecstore_gpu.DeviceIndex(vectors, dim, "l2")
    # spec: fp16 crosses the boundary, checked — a silent cast would hide
    # the quantization the whole format is built around
    with pytest.raises((TypeError, ValueError)):
        getattr(device, path)(rng.standard_normal((NQ, dim)).astype(np.float32))
