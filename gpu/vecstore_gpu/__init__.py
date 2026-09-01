import numpy as np

from .export import export_index
from .loader import load_gpu_index

# cmake drops the compiled module next to this file; a failed import
# just means there is no cuda build on this machine (the .so links the
# cuda runtime, so it won't load without a driver either — that is the
# whole device check, per SPEC no separate probe)
try:
    from . import _vecstore_gpu as _ext
except ImportError:
    _ext = None

_EXT_NAMES = ("DeviceIndex", "topk", "hello_add")

__all__ = ["is_available", "descend", "export_index", "load_gpu_index", *_EXT_NAMES]


def is_available():
    return _ext is not None


if _ext is not None:
    DeviceIndex = _ext.DeviceIndex
    topk = _ext.topk
    hello_add = _ext.hello_add


def __getattr__(name):
    # only consulted when the name wasn't bound above, i.e. the
    # extension is missing
    if name in _EXT_NAMES:
        raise RuntimeError(
            f"vecstore_gpu.{name} needs the _vecstore_gpu extension, which "
            "is not built here — build on a cuda machine via gpu/setup_pod.sh"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def descend(index, queries):
    """Host half of the gpu search: greedy-walk the upper layers of a
    loaded GpuIndex and return each query's layer-0 entry id, int32
    shape (nq,). Mirrors HNSWIndex.search's descent (layers top..1),
    but runs on the exported fp16 rows cast to fp32 — the same values
    the kernels see, so cpu and gpu start layer 0 at the same node."""
    queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
    dim = index.dim
    if queries.shape[1] not in (dim, index.stride):
        raise ValueError(
            f"expected queries of dim {dim} (or padded to {index.stride}), "
            f"got {queries.shape[1]}"
        )
    # pad columns are zero on the index side and carry no signal
    queries = queries[:, :dim]

    if index.metric == "l2":

        def dist(q, rows):
            diff = rows - q
            # fp32 accumulation to match the kernels, not the fp64 the
            # cpu index uses
            return np.einsum("ij,ij->i", diff, diff)

    elif index.metric == "ip":

        def dist(q, rows):
            # negated so lower = closer, consistent with l2
            return -(rows @ q)

    else:
        raise ValueError(f"unknown gpu metric: {index.metric}")

    vectors = index.vectors

    def rows(ids):
        # gather then cast: the walks touch few rows, no point casting
        # all N up front
        return vectors[ids, :dim].astype(np.float32)

    entries = np.empty(len(queries), dtype=np.int32)
    for qi, q in enumerate(queries):
        node = int(index.entry)
        best = dist(q, rows([node]))[0]
        # upper_layers[i] holds layer i+1; walk top..1, layer 0 belongs
        # to the gpu
        for layer in reversed(index.upper_layers):
            # hop to the closest neighbor until none improves; strictly
            # shrinking distance, so this terminates
            while True:
                neighbors = layer[node]
                if not neighbors:
                    break
                d = dist(q, rows(neighbors))
                i = int(np.argmin(d))
                if d[i] >= best:
                    break
                node, best = neighbors[i], d[i]
        entries[qi] = node
    return entries
