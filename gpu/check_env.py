#!/usr/bin/env python3
"""phase-0 gates for the gpu chapter, run by setup_pod.sh after the
build. nonzero exit on any FAIL. gate 5 failing is a provider problem,
not a code problem — do not start paid kernel work until it passes."""

import os
import re
import subprocess
import sys

GPU_DIR = os.path.dirname(os.path.abspath(__file__))
# nothing under gpu/ is pip-installed; the package imports from here
if GPU_DIR not in sys.path:
    sys.path.insert(0, GPU_DIR)

# tiny workload for the profiler/sanitizer probes — runs as a fresh
# interpreter, so it has to bootstrap sys.path itself
PROBE = (
    "import sys; sys.path.insert(0, %r); "
    "import numpy as np; "
    "from vecstore_gpu import _vecstore_gpu; "
    "a = np.arange(4096, dtype=np.float32); "
    "_vecstore_gpu.hello_add(a, a)"
) % GPU_DIR


def run(cmd, timeout):
    # missing binaries surface as FileNotFoundError, handled per gate
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def gate_nvidia_smi():
    try:
        proc = run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            timeout=30,
        )
    except FileNotFoundError:
        return "FAIL", "nvidia-smi not found — no driver visible in this container"
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip().splitlines()
        return "FAIL", msg[0] if msg else "nvidia-smi exited %d" % proc.returncode
    lines = proc.stdout.strip().splitlines()
    if not lines:
        return "FAIL", "nvidia-smi ran but reported no gpus"
    return "PASS", lines[0]


def gate_nvcc():
    try:
        proc = run(["nvcc", "--version"], timeout=30)
    except FileNotFoundError:
        return "FAIL", "nvcc not found — wrong image, use a *-devel cuda image"
    m = re.search(r"release (\d+)\.(\d+)", proc.stdout)
    if not m:
        return "FAIL", "could not parse nvcc --version output"
    if int(m.group(1)) < 12:
        return "FAIL", "cuda %s.%s — need 12.x" % (m.group(1), m.group(2))
    return "PASS", "cuda %s.%s" % (m.group(1), m.group(2))


def gate_import():
    try:
        import vecstore_gpu
    except Exception as e:
        return "FAIL", "import vecstore_gpu failed: %s" % e
    if not vecstore_gpu.is_available():
        return "FAIL", "extension not built — run cmake (setup_pod.sh does it)"
    return "PASS", "vecstore_gpu.is_available() == True"


def gate_hello():
    import numpy as np

    from vecstore_gpu import _vecstore_gpu

    rng = np.random.default_rng(0)
    a = rng.standard_normal(100_000).astype(np.float32)
    b = rng.standard_normal(100_000).astype(np.float32)
    try:
        out = _vecstore_gpu.hello_add(a, b)
    except Exception as e:
        return "FAIL", "hello_add raised: %s" % e
    # fp32 add is deterministic — device and numpy must agree bitwise
    if not np.array_equal(out, a + b):
        bad = int(np.count_nonzero(out != a + b))
        return "FAIL", "%d of 100000 elements differ from numpy" % bad
    return "PASS", "100000 floats, exact match with numpy"


def gate_ncu():
    cmd = [
        "ncu",
        "--metrics",
        "sm__cycles_elapsed.sum",
        "--target-processes",
        "all",
        sys.executable,
        "-c",
        PROBE,
    ]
    try:
        proc = run(cmd, timeout=300)
    except FileNotFoundError:
        return "WARN", "ncu not found — install the nsight-compute package to profile"
    except subprocess.TimeoutExpired:
        return "FAIL", "ncu timed out after 300s — retry once, then suspect the pod"
    out = proc.stdout + proc.stderr
    if "ERR_NVGPUCTRPERM" in out:
        return "FAIL", (
            "counters blocked (ERR_NVGPUCTRPERM) — provider problem: needs a "
            "privileged container, a different provider, or a full VM; do NOT "
            "start paid kernel work until this passes"
        )
    if proc.returncode != 0:
        return "FAIL", "ncu exited %d" % proc.returncode
    if "sm__cycles_elapsed.sum" not in out:
        return "FAIL", "ncu ran but collected no counters"
    return "PASS", "counter collection works"


def gate_sanitizer():
    # --error-exitcode makes detected errors visible in the return code;
    # by default the sanitizer passes the app's exit code through
    cmd = ["compute-sanitizer", "--error-exitcode", "9", sys.executable, "-c", PROBE]
    try:
        proc = run(cmd, timeout=300)
    except FileNotFoundError:
        return "WARN", "compute-sanitizer not found — part of the cuda toolkit"
    except subprocess.TimeoutExpired:
        return "FAIL", "compute-sanitizer timed out after 300s"
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        m = re.search(r"ERROR SUMMARY: \d+ errors?", out)
        return "FAIL", m.group(0) if m else "exited %d" % proc.returncode
    return "PASS", "hello_add runs clean"


GATES = [
    ("1 nvidia-smi", gate_nvidia_smi),
    ("2 nvcc >= 12", gate_nvcc),
    ("3 extension import", gate_import),
    ("4 hello_add == numpy", gate_hello),
    ("5 ncu counters", gate_ncu),
    ("6 compute-sanitizer", gate_sanitizer),
]


def main():
    width = max(len(name) for name, _ in GATES)
    print("phase-0 gates")
    failed = False
    ext_ok = True
    for name, fn in GATES:
        # 4-6 all exercise the extension; report why instead of a pile
        # of identical import tracebacks
        if not ext_ok and name[0] in "456":
            status, detail = "FAIL", "skipped — extension unavailable"
        else:
            try:
                status, detail = fn()
            except Exception as e:
                status, detail = "FAIL", "unexpected error: %s" % e
        if name[0] == "3" and status == "FAIL":
            ext_ok = False
        if status == "FAIL":
            failed = True
        print("  %-*s  %-4s  %s" % (width, name, status, detail), flush=True)
    if failed:
        print("\nFAIL — fix the gates above before starting paid kernel work")
        return 1
    print("\nall gates passed (WARNs are non-blocking)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
