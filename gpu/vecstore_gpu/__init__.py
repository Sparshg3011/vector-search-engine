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


def _layer_tables(index):
    """Per upper layer: the layer's rows as a compact fp32 array, an
    adjacency table of row indices into it (-1 padded), and the sorted
    global ids those rows correspond to. Built once per loaded index.
    The upper layers hold a few percent of the nodes, so the copies are
    small, and descend() then never touches the fp16 array or maps ids
    inside its hop loop."""
    tables = getattr(index, "_descend_tables", None)
    if tables is None:
        tables = []
        dim = index.dim
        for layer in index.upper_layers:
            nodes = np.fromiter(layer.keys(), dtype=np.int64, count=len(layer))
            nodes.sort()
            width = max((len(nbrs) for nbrs in layer.values()), default=0)
            adj = np.full((len(nodes), max(width, 1)), -1, dtype=np.int32)
            for row, node in enumerate(nodes):
                nbrs = layer[int(node)]
                if nbrs:
                    adj[row, : len(nbrs)] = np.searchsorted(nodes, nbrs)
            vecs = index.vectors[nodes, :dim].astype(np.float32)
            tables.append((nodes, adj, vecs))
        index._descend_tables = tables
    return tables


def descend(index, queries):
    """Host half of the gpu search: greedy-walk the upper layers of a
    loaded GpuIndex and return each query's layer-0 entry id, int32
    shape (nq,). Mirrors HNSWIndex.search's descent (layers top..1),
    but runs on the exported fp16 rows cast to fp32 — the same values
    the kernels see, so cpu and gpu start layer 0 at the same node.

    The whole batch walks together: one array op per hop for every
    query still moving, instead of a python loop per query per hop."""
    queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
    dim = index.dim
    if queries.shape[1] not in (dim, index.stride):
        raise ValueError(
            f"expected queries of dim {dim} (or padded to {index.stride}), "
            f"got {queries.shape[1]}"
        )
    # pad columns are zero on the index side and carry no signal
    queries = np.ascontiguousarray(queries[:, :dim])
    if index.metric not in ("l2", "ip"):
        raise ValueError(f"unknown gpu metric: {index.metric}")
    l2 = index.metric == "l2"
    tables = _layer_tables(index)
    entry = int(index.entry)

    if not tables:
        return np.full(len(queries), entry, dtype=np.int32)

    out = np.empty(len(queries), dtype=np.int32)
    # chunk so the (queries x neighbors x dim) gather stays small
    for c0 in range(0, len(queries), 4096):
        q = queries[c0 : c0 + 4096]
        nq = len(q)
        # upper_layers[i] holds layer i+1; walk top..1, layer 0 belongs
        # to the gpu. cur is a row index into the current layer's table.
        nodes, adj, vecs = tables[-1]
        cur = np.full(nq, int(np.searchsorted(nodes, entry)), dtype=np.int64)
        if l2:
            diff = vecs[cur] - q
            best = np.einsum("ij,ij->i", diff, diff)
        else:
            # negated so lower = closer, consistent with l2
            best = -np.einsum("ij,ij->i", vecs[cur], q)
        for li in range(len(tables) - 1, -1, -1):
            nodes, adj, vecs = tables[li]
            if li != len(tables) - 1:
                # the node we stopped on exists one layer down too
                cur = np.searchsorted(nodes, prev_nodes[cur])
            active = np.arange(nq)
            while len(active):
                nbrs = adj[cur[active]]
                valid = nbrs >= 0
                gathered = vecs[np.where(valid, nbrs, 0)]
                if l2:
                    diff = gathered - q[active][:, None, :]
                    d = np.einsum("abd,abd->ab", diff, diff)
                else:
                    d = -np.einsum("abd,ad->ab", gathered, q[active])
                d[~valid] = np.inf
                j = np.argmin(d, axis=1)
                dmin = d[np.arange(len(active)), j]
                # hop to the closest neighbor only while it strictly
                # improves; a query that stops stays put for this layer
                improved = dmin < best[active]
                if not improved.any():
                    break
                moved = active[improved]
                cur[moved] = nbrs[improved, j[improved]]
                best[moved] = dmin[improved]
                active = moved
            prev_nodes = nodes
        out[c0 : c0 + nq] = prev_nodes[cur]
    return out
