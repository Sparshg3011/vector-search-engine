"""Benchmark adapters — one small class per baseline, all the same shape:

    .name                       table label
    .available / .reason        why a baseline is missing, never silence
    .build(data)                never raises; failure disables with a reason
    .prepare(queries)           host-side dtype/padding, outside the timer
    .search(queries, k, **knobs) -> (ids, wall_ms, kernel_ms_or_None)

wall_ms is end-to-end (h2d + kernels + d2h + any host half of the
search); kernel_ms is cuda-event time from the extension, or None where
the library does not expose one. SPEC: the two are reported as separate
columns and never blended, so None means "no number", never "same as
wall".

Every optional import lives inside a build and its failure is stored on
the adapter — SPEC forbids a baseline quietly vanishing from the table.
"""

import os
import sys
import time

import numpy as np

GPU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(GPU_DIR)
# nothing under gpu/ is pip-installed; vecstore lives at the repo root
for _p in (GPU_DIR, REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class BenchData:
    """What every adapter builds from.

    `vectors` is the single source of truth for all baselines *and* for
    ground truth: the exported fp16 rows cast back to fp32, unpadded.
    Those are the exact numbers the kernels see, so a recall gap between
    cpu and gpu is an algorithm difference, not a quantization one. For
    a normalized export it is also the same normalized data the SPEC
    requires every cpu baseline to run on.
    """

    def __init__(self, gpu_index, source_index_path=None):
        self.gpu = gpu_index
        self.source_index_path = source_index_path
        self.dim = gpu_index.dim
        self.stride = gpu_index.stride
        self.metric = gpu_index.metric
        self.normalized = gpu_index.normalized
        self.vectors = np.ascontiguousarray(
            gpu_index.vectors[:, : gpu_index.dim], dtype=np.float32
        )
        self.n = len(self.vectors)

    def device_queries(self, queries):
        # the extension checks dtype, contiguity and width — build the
        # padded fp16 block here rather than let it raise per call
        out = np.zeros((len(queries), self.stride), dtype=np.float16)
        out[:, : self.dim] = queries
        return out


class Baseline:
    name = "baseline"
    # which sweep drives this index: "ef", "nprobe", "itopk", or None
    # for an exact index with nothing to tune
    knob = None

    def __init__(self):
        self.available = True
        self.reason = None
        # static facts worth recording next to the rows (actual nlist,
        # thread caveats); last_extra is per-call (descend cost)
        self.extra = {}
        self.last_extra = None

    def disable(self, reason):
        self.available = False
        self.reason = reason
        return self

    def build(self, data):
        """Never raises: a missing wheel, a cpu-only faiss or a device
        that will not come up disables the adapter with a stored reason
        so run.py can print it instead of dropping a row."""
        try:
            self._build(data)
        except Exception as e:
            self.disable(f"{type(e).__name__}: {e}")
        return self

    def _build(self, data):
        raise NotImplementedError

    def prepare(self, queries):
        """Cast/pad a query block into this baseline's native form. Run
        once per rep *before* the timer starts — a real caller already
        holds queries in the index's dtype, and charging one baseline
        for a cast the others do not need would skew the comparison."""
        return queries

    def search(self, queries, k, **knobs):
        raise NotImplementedError


# ---------------------------------------------------------------- cpu


class NumpyFlat(Baseline):
    """vecstore FlatIndex, one query at a time. The honest single-thread
    reference: no batching tricks, the same loop the repo's own
    benchmarks use."""

    name = "numpy_flat"

    def _build(self, data):
        from vecstore.flat import FlatIndex

        self.index = FlatIndex(dim=data.dim, metric=data.metric)
        self.index.add(data.vectors)
        if os.environ.get("OMP_NUM_THREADS") != "1":
            # the ip path is a blas gemv and blas thread counts are
            # fixed before numpy imports — "single thread" is a claim
            # this process cannot make after the fact
            self.extra["threads"] = (
                "OMP_NUM_THREADS was not 1 — rerun with OMP_NUM_THREADS=1 for a "
                "strictly single-threaded number"
            )

    def search(self, queries, k, **knobs):
        t0 = time.perf_counter()
        ids = [self.index.search(q, k=k)[0] for q in queries]
        wall = (time.perf_counter() - t0) * 1000.0
        return np.asarray(ids, dtype=np.int64), wall, None


class CpuHnsw(Baseline):
    """vecstore HNSWIndex from the source .npz the .gpu.npz was exported
    from — the cpu twin of gpu_hnsw, same graph, same rows."""

    name = "cpu_hnsw"
    knob = "ef"

    def _build(self, data):
        from vecstore.distances import METRICS
        from vecstore.hnsw import HNSWIndex

        if not data.source_index_path:
            raise ValueError(
                "needs --source-index (the saved HNSWIndex the .gpu.npz came from)"
            )
        index = HNSWIndex.load(data.source_index_path)
        if len(index) != data.n or index.dim != data.dim:
            raise ValueError(
                f"source index is {len(index)}x{index.dim}, gpu index is "
                f"{data.n}x{data.dim} — not the same build"
            )
        # the graph is untouched; only the rows are swapped for the
        # exported ones so cpu and gpu score identical numbers. a
        # normalized export also turns cosine into ip, which ranks the
        # same on unit rows — the graph was built under either ordering
        index._vectors = data.vectors
        index._size = data.n
        index.metric = data.metric
        index._dist = METRICS[data.metric]
        self.index = index

    def search(self, queries, k, ef=50, **knobs):
        t0 = time.perf_counter()
        ids = [self.index.search(q, k=k, ef=ef)[0] for q in queries]
        wall = (time.perf_counter() - t0) * 1000.0
        return np.asarray(ids, dtype=np.int64), wall, None


# ---------------------------------------------------------------- ours


def _extension():
    """Import vecstore_gpu and fail loudly if the .so is not here. The
    extension only builds on a cuda pod, so this is the normal outcome
    on a laptop and must read as a skip reason, not a crash."""
    import vecstore_gpu

    if not vecstore_gpu.is_available():
        raise RuntimeError(
            "_vecstore_gpu is not built on this machine — cuda only, see "
            "gpu/setup_pod.sh"
        )
    return vecstore_gpu


class GpuFlat(Baseline):
    """DeviceIndex.brute_force — K1+K2. Two adapters share this class,
    one per distance path, because the only difference that matters is
    which kernel computes the matrix."""

    def __init__(self, use_cublas):
        super().__init__()
        self.use_cublas = bool(use_cublas)
        self.name = "gpu_flat_cublas" if use_cublas else "gpu_flat_hand"

    def _build(self, data):
        ext = _extension()
        self.data = data
        self.index = ext.DeviceIndex(data.gpu.vectors, data.dim, data.metric)

    def prepare(self, queries):
        return self.data.device_queries(queries)

    def search(self, queries, k, **knobs):
        t0 = time.perf_counter()
        ids, _ = self.index.brute_force(queries, k, self.use_cublas)
        wall = (time.perf_counter() - t0) * 1000.0
        # brute_force chunks internally; last_kernel_ms covers the whole
        # call's kernels and excludes the transfers wall already counts
        return np.asarray(ids, dtype=np.int64), wall, float(self.index.last_kernel_ms())


class GpuHnsw(Baseline):
    """descend() on the host + DeviceIndex.hnsw — K3. The upper-layer
    walk is numpy and is part of the query path, so it sits inside
    wall_ms and is reported again on its own as descend_ms; it is never
    counted as kernel time."""

    name = "gpu_hnsw"
    knob = "ef"

    def _build(self, data):
        ext = _extension()
        self.data = data
        self.descend = ext.descend
        self.index = ext.DeviceIndex(data.gpu.vectors, data.dim, data.metric)
        self.index.set_graph(data.gpu.adjacency, data.gpu.degrees, int(data.gpu.entry))

    def prepare(self, queries):
        # descend walks fp32 host rows, the kernel wants padded fp16 —
        # both blocks are built before the timer, neither is free
        host = np.ascontiguousarray(queries, dtype=np.float32)
        return host, self.data.device_queries(queries)

    def search(self, queries, k, ef=50, **knobs):
        host, device = queries
        # SPEC: k <= ef <= MAX_EF; a sweep point below k would ask the
        # kernel for more results than its candidate list can hold
        ef = max(int(ef), k)
        t0 = time.perf_counter()
        entries = self.descend(self.data.gpu, host)
        t1 = time.perf_counter()
        ids, _ = self.index.hnsw(device, entries, k, ef)
        wall = (time.perf_counter() - t0) * 1000.0
        self.last_extra = {"descend_ms": (t1 - t0) * 1000.0}
        return np.asarray(ids, dtype=np.int64), wall, float(self.index.last_kernel_ms())


# -------------------------------------------------------------- faiss


def _faiss_gpu():
    import faiss

    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError("this is the faiss-cpu wheel — needs a faiss-gpu build")
    if faiss.get_num_gpus() < 1:
        raise RuntimeError("faiss reports 0 gpus")
    return faiss


def _faiss_flat(faiss, dim, metric):
    # the metric has to match the exported index: ip on normalized rows
    # is cosine's ranking, l2 otherwise. mismatch here silently measures
    # recall under the wrong order
    return faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)


class FaissFlatGpu(Baseline):
    """faiss GpuIndexFlat — the exact-search number gpu_flat has to
    beat, or at least explain itself against."""

    name = "faiss_flat_gpu"

    def _build(self, data):
        faiss = _faiss_gpu()
        self.faiss = faiss
        self.res = faiss.StandardGpuResources()
        cpu = _faiss_flat(faiss, data.dim, data.metric)
        cpu.add(data.vectors)
        self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu)

    def prepare(self, queries):
        return np.ascontiguousarray(queries, dtype=np.float32)

    def search(self, queries, k, **knobs):
        t0 = time.perf_counter()
        _, ids = self.index.search(queries, k)
        wall = (time.perf_counter() - t0) * 1000.0
        # faiss exposes no cuda-event timer; the column stays empty
        # rather than borrowing the wall number
        return np.asarray(ids, dtype=np.int64), wall, None


class FaissIvfGpu(Baseline):
    """faiss GpuIndexIVFFlat, nlist=4096, nprobe swept. The approximate
    baseline gpu_hnsw is actually competing with."""

    name = "faiss_ivf_gpu"
    knob = "nprobe"

    def _build(self, data):
        faiss = _faiss_gpu()
        self.faiss = faiss
        nlist = 4096
        if data.n < 39 * nlist:
            # faiss wants ~39 training points per centroid; a lopsided
            # quantizer is a worse baseline than a smaller honest one,
            # and the value that ran is recorded in extra
            nlist = max(1, data.n // 39)
        self.nlist = nlist
        self.extra["nlist"] = nlist
        metric = (
            faiss.METRIC_INNER_PRODUCT if data.metric == "ip" else faiss.METRIC_L2
        )
        self.res = faiss.StandardGpuResources()
        # faiss borrows the quantizer without owning it — drop this ref
        # and the index reads freed memory
        self._quantizer = _faiss_flat(faiss, data.dim, data.metric)
        cpu = faiss.IndexIVFFlat(self._quantizer, data.dim, nlist, metric)
        try:
            self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu)
            self.index.train(data.vectors)
        except Exception:
            # older builds refuse to clone an untrained ivf; fall back
            # to the (much slower) cpu kmeans and clone afterwards
            cpu.train(data.vectors)
            self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu)
        self.index.add(data.vectors)

    def prepare(self, queries):
        return np.ascontiguousarray(queries, dtype=np.float32)

    def _set_nprobe(self, nprobe):
        # gpu faiss caps nprobe at 2048 and it can never exceed nlist
        nprobe = max(1, min(int(nprobe), self.nlist, 2048))
        try:
            self.index.nprobe = nprobe
        except (AttributeError, TypeError):
            self.faiss.GpuParameterSpace().set_index_parameter(
                self.index, "nprobe", nprobe
            )
        return nprobe

    def search(self, queries, k, nprobe=8, **knobs):
        used = self._set_nprobe(nprobe)
        t0 = time.perf_counter()
        _, ids = self.index.search(queries, k)
        wall = (time.perf_counter() - t0) * 1000.0
        self.last_extra = {"nprobe_used": used}
        return np.asarray(ids, dtype=np.int64), wall, None


# --------------------------------------------------------------- cuvs


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    # cupy arrays have .get(), cuvs device_ndarray has .copy_to_host()
    for attr in ("get", "copy_to_host"):
        fn = getattr(x, attr, None)
        if callable(fn):
            return np.asarray(fn())
    return np.asarray(x)


def _cuvs_ids(out):
    # cuvs returns (distances, neighbors) and older releases returned
    # them the other way round — pick by dtype, not by position
    for item in out:
        arr = _to_numpy(item)
        if np.issubdtype(arr.dtype, np.integer):
            return arr
    raise RuntimeError("cuvs search returned no integer neighbor array")


class CuvsCagra(Baseline):
    """cuVS CAGRA — nvidia's gpu graph index, the one gpu_hnsw is
    measured against. itopk_size is its ef, so it takes the ef sweep."""

    name = "cuvs_cagra"
    knob = "itopk"

    def _build(self, data):
        from cuvs.neighbors import cagra

        self.cagra = cagra
        # no cosine on this side either: the export normalized angular
        # data, so inner_product reproduces the ranking
        metric = "inner_product" if data.metric == "ip" else "sqeuclidean"
        build = getattr(cagra, "build", None) or cagra.build_index
        self.index = build(cagra.IndexParams(metric=metric), data.vectors)

    def prepare(self, queries):
        # host arrays on purpose: cuvs takes them via dlpack and the
        # h2d copy then lands inside wall_ms, same as our own adapters
        return np.ascontiguousarray(queries, dtype=np.float32)

    def search(self, queries, k, itopk=64, **knobs):
        # itopk_size must be >= k; cuvs rounds it up to a multiple of 32
        params = self.cagra.SearchParams(itopk_size=max(int(itopk), k))
        t0 = time.perf_counter()
        out = self.cagra.search(params, self.index, queries, k)
        wall = (time.perf_counter() - t0) * 1000.0
        return _cuvs_ids(out).astype(np.int64), wall, None


def all_baselines():
    """Fresh adapters in table order. Construction is cheap and touches
    no device — availability is decided in build()."""
    return [
        NumpyFlat(),
        CpuHnsw(),
        GpuFlat(False),
        GpuFlat(True),
        GpuHnsw(),
        FaissFlatGpu(),
        FaissIvfGpu(),
        CuvsCagra(),
    ]


BASELINE_NAMES = [b.name for b in all_baselines()]
