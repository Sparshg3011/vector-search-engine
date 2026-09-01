import numpy as np
import pytest

import vecstore_gpu

if not vecstore_gpu.is_available():
    pytest.skip(
        "_vecstore_gpu is not built here — this file only runs on a cuda machine",
        allow_module_level=True,
    )

pytestmark = pytest.mark.gpu

rng = np.random.default_rng(0)


def test_hello_add_matches_numpy_exactly():
    a = rng.standard_normal(100_000).astype(np.float32)
    b = rng.standard_normal(100_000).astype(np.float32)
    out = vecstore_gpu.hello_add(a, b)
    assert out.dtype == np.float32
    assert out.shape == a.shape
    # ieee fp32 add is deterministic and elementwise — nothing gets
    # reordered, so anything short of bitwise equality is a real bug
    np.testing.assert_array_equal(out, a + b)


def test_hello_add_covers_the_tail_of_the_grid():
    # 100_003 is prime: the last block runs partly out of range, which
    # is where a missing bounds guard shows up
    a = rng.standard_normal(100_003).astype(np.float32)
    b = rng.standard_normal(100_003).astype(np.float32)
    np.testing.assert_array_equal(vecstore_gpu.hello_add(a, b), a + b)


def test_hello_add_length_mismatch_raises():
    a = rng.standard_normal(64).astype(np.float32)
    b = rng.standard_normal(32).astype(np.float32)
    with pytest.raises(ValueError):
        vecstore_gpu.hello_add(a, b)
