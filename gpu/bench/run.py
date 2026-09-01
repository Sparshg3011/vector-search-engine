#!/usr/bin/env python3
"""Run the gpu benchmark matrix and write one results json.

Protocol is SPEC's, not negotiable per run: >= 3 warmup reps, median of
>= 20 timed reps (>= 200 distinct queries at batch 1), ground truth
computed once in fp64 on the same data every baseline sees, wall-clock
and cuda-event kernel time reported as separate columns. A baseline that
cannot run prints why and still appears in the json — no silent drops.

    python gpu/bench/run.py --gpu-index data/sift-1M.gpu.npz \\
        --source-index data/sift-1M-M16-ef100-l2.npz \\
        --queries data/sift-queries.npy --baselines all
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
GPU_DIR = os.path.dirname(BENCH_DIR)
REPO_DIR = os.path.dirname(GPU_DIR)
# nothing under gpu/ is pip-installed; bench/ is not a package either
for _p in (BENCH_DIR, GPU_DIR, REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baselines import BASELINE_NAMES, BenchData, all_baselines  # noqa: E402

from vecstore import recall  # noqa: E402
from vecstore_gpu import load_gpu_index  # noqa: E402

WARMUP_REPS = 3
TIMED_REPS = 20
# batch 1 measures latency, and one query measured 20 times is one
# query — the protocol wants >= 200 distinct ones
BATCH1_REPS = 200


def parse_ints(text):
    return [int(x) for x in text.split(",") if x.strip()]


def load_queries(spec, data, min_pool, seed=0):
    """The query pool. Stays fp32: the fp16 cast belongs to the gpu path
    and its recall cost is a property of that design, so it is charged
    to the gpu baselines rather than baked into the data."""
    if spec.startswith("random:"):
        n = max(int(spec.split(":", 1)[1]), min_pool)
        rng = np.random.default_rng(seed)
        q = rng.standard_normal((n, data.dim)).astype(np.float32)
        if data.normalized:
            # index rows are unit vectors; keep queries on the same
            # sphere so this looks like the angular datasets it stands in for
            q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
        return q, {"source": spec, "pool": len(q), "wrapped": False}
    q = np.atleast_2d(np.asarray(np.load(spec), dtype=np.float32))
    if q.shape[1] not in (data.dim, data.stride):
        raise SystemExit(
            f"queries are dim {q.shape[1]}, index is dim {data.dim} "
            f"(stride {data.stride})"
        )
    q = np.ascontiguousarray(q[:, : data.dim])
    return q, {"source": spec, "pool": len(q), "wrapped": len(q) < min_pool}


def ground_truth(vectors, queries, k, metric, budget=3e7):
    """Exact neighbors in fp64, once, for the whole pool. Chunked over
    the database: the full nq x N score matrix is 16 GB at 2048 x 1M, so
    only `budget` scores are ever live and the running top-k is merged
    per chunk."""
    x = np.asarray(vectors, dtype=np.float64)
    q = np.asarray(queries, dtype=np.float64)
    nq, n = len(q), len(x)
    chunk = max(1, min(n, int(budget // max(nq, 1))))
    qsq = np.einsum("ij,ij->i", q, q)[:, None]
    best_d = np.empty((nq, 0), dtype=np.float64)
    best_i = np.empty((nq, 0), dtype=np.int64)
    for start in range(0, n, chunk):
        block = x[start : start + chunk]
        if metric == "ip":
            # negated so lower = closer, same convention as vecstore
            d = -(q @ block.T)
        else:
            # ||q||^2 + ||x||^2 - 2q.x — the gemm form. in fp64 the
            # cancellation error is ~1e-13 relative, orders below any
            # tie that could reorder neighbors
            d = qsq + np.einsum("ij,ij->i", block, block)[None, :] - 2.0 * (q @ block.T)
        kk = min(k, d.shape[1])
        part = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        best_d = np.concatenate([best_d, np.take_along_axis(d, part, axis=1)], axis=1)
        best_i = np.concatenate([best_i, part + start], axis=1)
        if best_d.shape[1] > k:
            keep = np.argpartition(best_d, k - 1, axis=1)[:, :k]
            best_d = np.take_along_axis(best_d, keep, axis=1)
            best_i = np.take_along_axis(best_i, keep, axis=1)
    order = np.argsort(best_d, axis=1)
    return np.take_along_axis(best_i, order, axis=1)


def rep_slices(pool_size, batch, reps):
    """Rep i takes the next `batch` queries and wraps around the pool, so
    consecutive reps are not re-running byte-identical work out of a warm
    cache."""
    slices, cursor = [], 0
    for _ in range(reps):
        slices.append(np.arange(cursor, cursor + batch) % pool_size)
        cursor = (cursor + batch) % pool_size
    return slices


def sweep_for(baseline, efs, nprobes):
    # cagra's itopk_size is its ef, so it rides the same sweep
    if baseline.knob in ("ef", "itopk"):
        return efs
    if baseline.knob == "nprobe":
        return nprobes
    return [None]


def run_config(baseline, pool, truth, k, batch, knob_value):
    """One (baseline, batch, knob) cell. Query blocks are prepared
    outside the timer; only the search call is measured."""
    knobs = {} if baseline.knob is None else {baseline.knob: knob_value}
    reps = BATCH1_REPS if batch == 1 else TIMED_REPS
    walls, kernels, extras, got = [], [], [], {}
    for rep, idx in enumerate(rep_slices(len(pool), batch, WARMUP_REPS + reps)):
        block = baseline.prepare(pool[idx])
        baseline.last_extra = None
        ids, wall, kernel = baseline.search(block, k, **knobs)
        if rep < WARMUP_REPS:
            continue
        walls.append(wall)
        kernels.append(kernel)
        if baseline.last_extra:
            extras.append(baseline.last_extra)
        for qi, row in zip(idx, np.atleast_2d(ids)):
            # deterministic baselines rewrite the same answer on a wrap;
            # keying by query id keeps recall an unweighted mean
            got[int(qi)] = row
    wall = statistics.median(walls)
    # one None means the library has no kernel timer at all — leave the
    # column empty instead of blending it with wall time
    kernel = None if any(x is None for x in kernels) else statistics.median(kernels)
    row = {
        "baseline": baseline.name,
        "batch": batch,
        "knob": baseline.knob,
        "knob_value": knob_value,
        "k": k,
        "recall": float(np.mean([recall(truth[qi][:k], got[qi]) for qi in got])),
        "reps": reps,
        "queries_scored": len(got),
        "wall_ms_batch": wall,
        "kernel_ms_batch": kernel,
        "wall_ms_per_query": wall / batch,
        "kernel_ms_per_query": None if kernel is None else kernel / batch,
        "qps": (batch * 1000.0 / wall) if wall > 0 else None,
    }
    if extras:
        # median each per-call fact so one stalled rep cannot define it
        row["extra"] = {
            key: statistics.median([e[key] for e in extras]) for key in extras[0]
        }
    return row


def _git(args):
    try:
        proc = subprocess.run(
            ["git", "-C", REPO_DIR] + args, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def environment():
    """Scraped after the builds so library versions come from what was
    actually imported. Clocks matter: a throttled card and a boosting
    one are different machines."""
    env = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "git_rev": _git(["rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
    }
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,clocks.sm",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            env["gpus"] = [l.strip() for l in proc.stdout.strip().splitlines() if l.strip()]
        else:
            env["gpus"] = None
            env["nvidia_smi_error"] = (proc.stderr or proc.stdout).strip()[:200]
    except (OSError, subprocess.SubprocessError) as e:
        env["gpus"] = None
        env["nvidia_smi_error"] = str(e)
    for mod in ("faiss", "cuvs", "vecstore_gpu"):
        loaded = sys.modules.get(mod)
        env[mod] = getattr(loaded, "__version__", "present") if loaded else None
    return env


def print_header(k):
    print(
        f"{'baseline':<16} {'batch':>6} {'knob':>12} {'recall@%d' % k:>10} "
        f"{'wall ms/q':>10} {'kernel ms/q':>12} {'qps':>12}"
    )


def print_row(row):
    knob = "-" if row["knob"] is None else f"{row['knob']}={row['knob_value']}"
    kernel = row["kernel_ms_per_query"]
    kernel = "-" if kernel is None else f"{kernel:.4f}"
    qps = "-" if row["qps"] is None else f"{row['qps']:.1f}"
    print(
        f"{row['baseline']:<16} {row['batch']:>6} {knob:>12} {row['recall']:>10.3f} "
        f"{row['wall_ms_per_query']:>10.4f} {kernel:>12} {qps:>12}",
        flush=True,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu-index", required=True, help="path to a .gpu.npz")
    parser.add_argument(
        "--source-index", help="saved HNSWIndex .npz the export came from (cpu_hnsw)"
    )
    parser.add_argument("--queries", default="random:1000", help="path.npy or random:N")
    parser.add_argument(
        "--baselines",
        default="all",
        help="comma list of %s, or 'all'" % ",".join(BASELINE_NAMES),
    )
    parser.add_argument("--batches", default="1,32,128,512,2048")
    parser.add_argument("--ef", default="50,100,200", help="graph-index sweep")
    parser.add_argument("--nprobe", default="1,8,32,128", help="faiss ivf sweep")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", help="results json (default bench/results/<ts>.json)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    batches = parse_ints(args.batches)
    efs = parse_ints(args.ef)
    nprobes = parse_ints(args.nprobe)

    names = BASELINE_NAMES if args.baselines == "all" else args.baselines.split(",")
    unknown = [n for n in names if n not in BASELINE_NAMES]
    if unknown:
        raise SystemExit(
            f"unknown baseline(s) {','.join(unknown)} — pick from "
            f"{','.join(BASELINE_NAMES)}"
        )

    gpu_index = load_gpu_index(args.gpu_index)
    gpu_index.validate()
    data = BenchData(gpu_index, args.source_index)
    k = min(args.k, data.n)
    pool, pool_meta = load_queries(args.queries, data, max(batches + [BATCH1_REPS]))
    if pool_meta["wrapped"]:
        print(
            f"note: {pool_meta['pool']} queries is fewer than the largest batch/rep "
            "count — reps wrap around the pool"
        )

    print(
        f"{data.n} vectors, dim={data.dim} (stride {data.stride}), "
        f"metric={data.metric}, normalized={data.normalized}, M={data.gpu.M}"
    )
    print(f"{pool_meta['pool']} queries from {pool_meta['source']}, k={k}")

    selected = [b for b in all_baselines() if b.name in names]
    for baseline in selected:
        baseline.build(data)
    for baseline in selected:
        if not baseline.available:
            print(f"skip {baseline.name}: {baseline.reason}")

    t0 = time.perf_counter()
    truth = ground_truth(data.vectors, pool, k, data.metric)
    print(
        f"ground truth: fp64 brute force over {pool_meta['pool']} queries in "
        f"{time.perf_counter() - t0:.1f}s"
    )

    rows = []
    print_header(k)
    for baseline in selected:
        if not baseline.available:
            continue
        for batch in batches:
            # a mid-run failure disables the adapter; stop feeding it
            if not baseline.available:
                break
            for knob_value in sweep_for(baseline, efs, nprobes):
                try:
                    row = run_config(baseline, pool, truth, k, batch, knob_value)
                except Exception as e:
                    # a failure here is the kernel stub or an oom, not a
                    # missing wheel; kill the baseline and keep the reason
                    baseline.disable(f"{type(e).__name__}: {e}")
                    print(f"skip {baseline.name} (from batch {batch}): {baseline.reason}")
                    break
                rows.append(row)
                print_row(row)

    for baseline in selected:
        descends = [
            r["extra"]["descend_ms"] / r["batch"]
            for r in rows
            if r["baseline"] == baseline.name and "descend_ms" in r.get("extra", {})
        ]
        if descends:
            # inside wall, never inside kernel — say so rather than let
            # a reader assume the gpu number is the whole story
            print(
                f"note: {baseline.name} host descend costs "
                f"{min(descends):.4f}-{max(descends):.4f} ms/query, included in wall"
            )

    out_path = args.out or os.path.join(
        BENCH_DIR, "results", time.strftime("%Y%m%d-%H%M%S") + ".json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu_index": os.path.abspath(args.gpu_index),
        "source_index": args.source_index and os.path.abspath(args.source_index),
        "k": k,
        "dataset": {
            "n": data.n,
            "dim": data.dim,
            "stride": data.stride,
            "metric": data.metric,
            "normalized": bool(data.normalized),
            "M": data.gpu.M,
        },
        "queries": pool_meta,
        "protocol": {
            "warmup_reps": WARMUP_REPS,
            "timed_reps": TIMED_REPS,
            "batch1_reps": BATCH1_REPS,
            "batches": batches,
            "ef": efs,
            "nprobe": nprobes,
            "statistic": "median over timed reps",
        },
        "environment": environment(),
        "baselines": {
            b.name: {"available": b.available, "reason": b.reason, "extra": b.extra}
            for b in selected
        },
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
