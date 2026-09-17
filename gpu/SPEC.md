# GPU chapter — build contract (v1)

The CUDA port of vecstore: layer-0 search on GPU, fp16 storage / fp32
accumulation, warp-per-query HNSW beam search, benchmarked against
FAISS-GPU and cuVS/CAGRA. This file is the contract between the glue, the
tests and the benchmarks on one side and the kernels on the other. When
code and this spec disagree, fix one of them in the same commit.

## Kernels

- `kernels/distances.cu` — K1, tiled distance matrix. `launchers.cu` also
  carries a cuBLAS GEMM path for the same contract, kept for comparison.
- `kernels/topk.cu` — K2, per-row top-k selection.
- `kernels/hnsw_search.cu` — K3, warp-per-query beam search: a
  global-memory visited table and a shared-memory candidate list.

## Directory layout

```
gpu/
├── SPEC.md                 # this file
├── CMakeLists.txt          # builds _vecstore_gpu (pybind11 + CUDA, C++17)
├── kernels/
│   ├── config.cuh          # constants + launch geometry shared with the host
│   ├── hello.cu            # toolchain check: vector add (complete)
│   ├── distances.cu        # K1  tiled distance matrix
│   ├── topk.cu             # K2  per-row top-k
│   └── hnsw_search.cu      # K3  warp-per-query beam search
├── src/
│   ├── launchers.h         # host API, deliberately cuda-free
│   ├── launchers.cu        # host wrappers; #includes ../kernels/*.cu (one TU)
│   └── bindings.cpp        # pybind11 module _vecstore_gpu
├── vecstore_gpu/           # python package (extension .so lands here)
│   ├── __init__.py         # is_available(), DeviceIndex re-export, descend()
│   ├── export.py           # HNSWIndex .npz → .gpu.npz (runs anywhere)
│   └── loader.py           # .gpu.npz → GpuIndex namespace + validation
├── tests/                  # pytest; conftest.py adds gpu/ to sys.path
├── bench/                  # run.py, baselines.py, plot.py, fetch_queries.py, results/
├── tools/
│   ├── precheck.py         # pre-pod structural compile against stub cuda headers
│   └── cudastub/           # permissive stand-ins for cuda_runtime.h, cuda_fp16.h, cublas_v2.h
├── slurm/
│   └── discovery.sbatch    # build + test + benchmark job for USC Discovery
├── setup_pod.sh            # deps → build → check_env.py (--deps-only / --build-only)
├── check_env.py            # phase-0 gates (see below)
├── sync.sh                 # rsync push loop, mac → pod
├── Dockerfile              # pinned alternative to setup_pod.sh
├── notes.md                # engineering log (seeded with measurements)
└── WRITEUP.md              # skeleton of the final writeup
```

Python imports: `gpu/conftest.py` and scripts insert `gpu/` into `sys.path`;
everything imports `vecstore_gpu`. Nothing under `gpu/` is pip-installed.
CMake sets the extension output directory to `gpu/vecstore_gpu/`.

## Serialized GPU index format — `<name>.gpu.npz`

Produced by `export.py` from a saved `HNSWIndex` (or a live one). Keys:

| key | dtype / shape | meaning |
|---|---|---|
| `vectors` | float16 `(N, stride)` | row-padded vectors; pad values 0.0 |
| `dim` | int64 scalar | true dimensionality |
| `stride` | int64 scalar | `dim` rounded up to a multiple of 8 (16-byte rows) |
| `adjacency` | int32 `(N, 2M)` | layer-0 neighbor ids, padded with `-1` |
| `degrees` | int32 `(N,)` | valid entries per adjacency row |
| `M` | int64 scalar | HNSW M (layer-0 cap is exactly `2M`) |
| `entry` | int64 scalar | entry-point node id |
| `metric` | str scalar | `"l2"` or `"ip"` (see normalization rule) |
| `normalized` | bool scalar | True when rows were L2-normalized at export |
| `upper_layers` | str scalar | json: list (layers 1..top) of `{node: [nbrs]}` |

Rules:
- **No cosine on the GPU.** Angular datasets are L2-normalized at export
  (`export.py --normalize`); after that, l2 and ip both give cosine's
  ranking. Every CPU baseline in tests/benchmarks must use the *same*
  normalized data.
- fp16 overflow: SIFT squared-L2 sums reach ~8.3M > fp16 max 65504 —
  **all distance accumulation is fp32**. Never accumulate in `__half`.
- Padding dims are zero, so they contribute nothing to l2 or ip; kernels
  may loop to `stride` instead of `dim` if convenient.

## Extension API (`_vecstore_gpu`, pybind11)

fp16 crosses the boundary as `np.float16` arrays (C-contiguous, checked).
ids are int32 (N < 2^31). All methods synchronize before returning and
record kernel-only time via CUDA events.

- `hello_add(a: f32[n], b: f32[n]) -> f32[n]` — device round-trip check.
- `topk(dmat: f32[nq, n], k) -> (ids i32[nq, k], dists f32[nq, k])` —
  module-level (K2 is testable without stored vectors). Rows sorted
  ascending. `k <= 128`; larger k raises.
- `class DeviceIndex(vectors: f16[N, stride], dim, metric: str)`
  - `.set_graph(adjacency: i32[N, 2M], degrees: i32[N], entry: int)`
  - `.distances(queries: f16[nq, stride]) -> f32[nq, N]` — K1 alone,
    hand-rolled path (for contract tests; caller keeps nq small).
  - `.distances_cublas(queries) -> f32[nq, N]` — GEMM path
    (‖q‖²+‖x‖²−2q·x for l2, plain GEMM for ip; norms precomputed fp32
    at construction by a helper kernel).
  - `.brute_force(queries: f16[nq, stride], k, use_cublas=False)
     -> (ids, dists)` — K1+K2 pipeline, **internally chunked** so the
    materialized matrix stays ≤ 2 GB (batch=2048 × 1M would be 8 GB).
  - `.hnsw(queries: f16[nq, stride], entries: i32[nq], k, ef)
     -> (ids, dists)` — K3. `ef <= 256` (`MAX_EF`), `k <= ef`.
  - `.last_kernel_ms() -> float` — CUDA-event time of the previous call
    (kernel(s) only, no H2D/D2H). Benchmarks report this *and* wall time,
    separately labeled.

**Before you pay for a compile.** On a machine with no nvcc, check the
structure first:

```bash
python3 gpu/tools/precheck.py
```

It compiles the device side against permissive stub headers with the
host compiler, catching typos, undeclared names, wrong argument counts
at kernel launches and unbalanced braces. It does not check cuda
semantics - nvcc on the pod does. If it rejects code nvcc accepts, widen
the stub in `tools/cudastub`, never bend the kernel to fit it.

Launch geometry (tile size, threads per block, warps per block, K3's
dynamic shared-memory size) lives in `kernels/config.cuh` because both
sides index with it — change a tile in a kernel, change it there too.

## K3 design constraints (from measurement, 2026-08-30)

Instrumented the real saved indexes (see `notes.md` for the full table):
SIFT-1M at ef=200 visits **median 2,792 / p95 3,547 / max ~3,700** nodes
per query; expansions ≈ ef; candidate pushes ≈ 3×ef. Therefore:

- **Visited set v1 lives in global memory**: per-in-flight-query hash,
  `VISITED_SLOTS = 8192` int32 slots (~2× p95 headroom), workspace
  allocated by the launcher as `(grid_queries, VISITED_SLOTS)`. A
  shared-memory evict-on-collision variant is the planned A/B — safe only
  because result-list insertion dedups by id.
- Candidate structure: one combined sorted list of length `ef`
  (dist, id, expanded-flag) in shared memory — measurement supports it
  (expansions ≈ ef). `MAX_EF = 256` → ≤ 2 KB + flags per query.
- Warp-per-query; entry points come from the host: `descend()` in
  `vecstore_gpu/__init__.py` runs the upper-layer greedy walk in NumPy on
  the fp16 vectors cast to fp32 (same data the GPU sees).

## Tolerances (fixed now, never renegotiated mid-debug)

| what | reference | gate |
|---|---|---|
| hello | numpy add | exact |
| K1 (both paths) | fp64 numpy on the same fp16-quantized inputs | `rtol=1e-3, atol=1e-2` |
| K1, SIFT-scale case only | same | `rtol=1e-4` — integers 0..255 are exact in fp16, so that test carries no quantization error to hide behind. A tighter guard on one case, not a renegotiated gate. |
| K2 | `np.argpartition` on the same fp32 matrix | ids compared as sets; at the k-boundary, ids whose distances differ by ≤1e-3 relative are interchangeable; dists sorted ascending |
| K3 | vecstore `HNSWIndex.search`, same ef, same normalized data | mean recall@10 over ≥200 queries within **0.01** of CPU at ef ∈ {50, 200}; disagreement beyond that = bug, not "GPU is different" |

Recall for the K3 gate is scored against exact `FlatIndex` neighbors of
the **exported fp16 rows**, not the fp32 originals — same-data rule.
(Measured: scoring against fp32 truth moves the CPU number by 0.0005,
20× inside the gate, so the choice cannot decide a pass.)

Argument validation raises `ValueError` from the binding layer, ahead of
the device call; the C++ core repeats the checks as a `runtime_error`
backstop. A state error (`hnsw()` before `set_graph`) is a
`RuntimeError`. Wrong dtypes are rejected, never force-cast: a silent
fp64→fp32 downcast of the distance matrix would move the top-k boundary.

## Benchmark protocol (`bench/run.py`)

- Axes: batch ∈ {1, 32, 128, 512, 2048} (first-class), ef sweep for graph
  indexes, `nprobe` sweep for faiss IVF, k=10 default.
- Baselines (each optional, skipped with a printed notice — no silent
  drops): numpy flat (single thread), vecstore CPU HNSW, gpu flat
  (K1+K2), gpu hnsw (K3), faiss-gpu flat, faiss-gpu IVF, cuVS CAGRA.
- ≥3 warmup runs, then median of ≥20 timed runs (batch 1: ≥200 queries).
- Report kernel-ms (CUDA events) and end-to-end wall-ms (incl. H2D/D2H)
  as separate labeled columns; never blend them. `gpu_hnsw`'s host
  descent counts inside wall time (it is part of the query path) and is
  also broken out as `extra.descend_ms`, so it can never be read as
  kernel time.
- **Every baseline scores under the export's own metric.** On a
  normalized angular export the faiss baselines must use
  `IndexFlatIP` / `METRIC_INNER_PRODUCT`; ranking those vectors by L2
  scores recall under the wrong ordering. Anything a baseline adapts at
  runtime (the IVF `nlist`, capped when `N < 39·nlist` by faiss's
  training-point ratio) is recorded in the results json rather than
  assumed.
- Log GPU name, driver, and SM clock via nvidia-smi into the results
  json; results land in `bench/results/*.json`, plots via `plot.py`.

## Phase-0 gates (`check_env.py`, run by `setup_pod.sh`)

1. `nvidia-smi` works; 2. `nvcc --version` ≥ 12.x; 3. CMake build
succeeds; 4. `hello_add` matches numpy exactly; 5. **`ncu` collects
counters** on a probe (rentals often block them — `ERR_NVGPUCTRPERM`
means switch instance/provider or add `--privileged`); 6.
`compute-sanitizer` runs clean on hello. Print a PASS/FAIL table; nonzero
exit on any failure. Gate 5 failing is a *provider* problem, not a code
problem — do not start renting by the hour until 5 passes.

Gate 5 reads the driver's `RmProfilingAdminOnly` from
`/proc/driver/nvidia/params` first: `1` without root means counters are
blocked, and the gate says so without spending minutes proving it.
`--profiler-optional` (forwarded by `setup_pod.sh`) reports gate 5 as
WARN instead of FAIL, for clusters where counters may be unavailable but
tests and benchmarks should still run.

## Running on a Slurm cluster (USC Discovery)

Discovery's compute nodes have no internet, so everything that downloads
runs on a login node and the gpu job only builds, tests and benchmarks.
Access needs a CARC project account — `myaccount` lists yours and gives
the `<project_id>` used below.

Once, on a login node (`ssh <netid>@discovery.usc.edu`; USC VPN when off
campus):

```bash
module purge
module load gcc/13.3.0 cuda/12.6.3 python/3.11.9
python3 -m venv ~/venvs/vecstore
source ~/venvs/vecstore/bin/activate
git clone https://github.com/Sparshg3011/vector-search-engine.git
cd vector-search-engine
bash gpu/setup_pod.sh --deps-only
python3 gpu/bench/fetch_queries.py sift-128-euclidean
```

From your own machine, upload the exported index (the transfer nodes are
the fast path for large files):

```bash
scp data/sift.gpu.npz data/sift-128-euclidean-vecstore.npz \
    <netid>@hpc-transfer1.usc.edu:~/vector-search-engine/data/
```

Then, from the repo root on a login node:

```bash
sbatch --account=<project_id> gpu/slurm/discovery.sbatch
squeue -u $USER
tail -f slurm-vecstore-gpu-<jobid>.out
```

The job loads the same modules, activates the venv, runs
`setup_pod.sh --build-only --profiler-optional`, the gpu tests, and the
SIFT benchmark into `gpu/bench/results/discovery-<jobid>.json` with its
plots. It asks for one L40S; command-line flags override the script, so
`--gpus-per-task=a100:1` or `--gpus-per-task=a40:1` work too, and the
build targets whichever card the job lands on. For a quick build-and-test
loop on the short-wait debug partition (A40, 1 hour max):

```bash
BENCH=0 sbatch --account=<project_id> --partition=debug \
    --gpus-per-task=a40:1 --time=00:45:00 gpu/slurm/discovery.sbatch
```

Load the modules again in every new login session before activating the
venv; its python is the module's.

## Conventions

Match the repo voice: no type annotations, lowercase pragmatic comments
that state constraints (not narration). Tests mirror `tests/test_flat.py`
style. GPU tests: `pytest.mark.gpu` + skip when the extension or a device
is missing; edge cases owed from the prototype suite: awkward
prime-ish shapes, dim < tile, k=1, k=cap, k>n, empty/degenerate inputs,
duplicate-distance ties, chunking equivalence.
