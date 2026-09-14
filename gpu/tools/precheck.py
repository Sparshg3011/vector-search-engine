"""Pre-pod structural check of the native layer, for machines with no
nvcc (i.e. your mac).

Compiles the device side against stub cuda headers (tools/cudastub)
with the host compiler. That catches typos, undeclared names, wrong
argument counts at kernel launches and unbalanced braces - the mistakes
that would otherwise burn the first paid minutes on a pod. It does NOT
check cuda semantics: barriers, shared memory, warp behavior and fp16
arithmetic all pass through the stubs unjudged. The pod's nvcc is the
real compiler; this only makes sure you arrive with code worth compiling.

    python3 gpu/tools/precheck.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
STUB = os.path.join(HERE, "cudastub")


def blank_comments_and_strings(src):
    """Same length as src, with comment and string-literal bodies turned
    to spaces (newlines kept), so searching for launch syntax can never
    trip on a '<<<' inside a comment."""
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            while i < n and src[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and nxt == "*":
            out[i] = out[i + 1] = " "
            i += 2
            while i < n and not (src[i] == "*" and i + 1 < n and src[i + 1] == "/"):
                if src[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = out[i + 1] = " "
                i += 2
        elif c in "\"'":
            quote = c
            i += 1
            while i < n and src[i] != quote and src[i] != "\n":
                # an escape consumes the next character, quotes included
                step = 2 if src[i] == "\\" else 1
                for p in range(i, min(i + step, n)):
                    out[p] = " "
                i += step
            i += 1
        else:
            i += 1
    return "".join(out)


def strip_launch_configs(src, name):
    """kernel<<<grid, block>>>(args)  ->  kernel               (args)

    The launch configuration is replaced by spaces of the same length,
    so every line and column clang reports still points at the real
    file."""
    scan = blank_comments_and_strings(src)
    out = list(src)
    i = 0
    while True:
        j = scan.find("<<<", i)
        if j < 0:
            break
        k = scan.find(">>>", j + 3)
        if k < 0:
            line = src.count("\n", 0, j) + 1
            raise SystemExit(f"{name}:{line}: '<<<' with no matching '>>>'")
        for p in range(j, k + 3):
            if src[p] != "\n":
                out[p] = " "
        i = k + 3
    return "".join(out)


def find_cxx():
    for cxx in ("clang++", "g++"):
        if shutil.which(cxx):
            return cxx
    raise SystemExit("no host c++ compiler found (clang++ or g++)")


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def check_device(gpu, cxx, tmp):
    """launchers.cu plus every kernel it includes, as one translation
    unit - the same shape nvcc sees."""
    path = os.path.join(gpu, "src", "launchers.cu")
    with open(path) as f:
        src = f.read()
    body = f'#line 1 "{path}"\n' + strip_launch_configs(src, "launchers.cu")
    unit = os.path.join(tmp, "launchers_precheck.cpp")
    with open(unit, "w") as f:
        f.write(body)
    cmd = [
        cxx, "-std=c++17", "-fsyntax-only", "-Wall",
        "-Wno-unused-variable", "-Wno-unused-function",
        "-Wno-unused-but-set-variable", "-Wno-undefined-internal",
        "-Wno-unknown-warning-option",
        "-I", STUB,
        "-I", os.path.join(gpu, "src"),
        "-I", os.path.join(gpu, "kernels"),
        unit,
    ]
    return run(cmd)


def check_bindings(gpu, cxx):
    """bindings.cpp is host-only by design, so this is a real compile
    against real pybind11 headers - when pybind11 is installed."""
    try:
        import pybind11
    except ImportError:
        return None, "pip install pybind11 to check bindings.cpp too"
    import sysconfig

    cmd = [
        cxx, "-std=c++17", "-fsyntax-only", "-Wall",
        "-I", pybind11.get_include(),
        "-I", sysconfig.get_paths()["include"],
        "-I", os.path.join(gpu, "src"),
        os.path.join(gpu, "src", "bindings.cpp"),
    ]
    return run(cmd)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--gpu-dir",
        default=os.path.dirname(HERE),
        help="gpu/ tree to check (default: the one this script lives in)",
    )
    args = parser.parse_args()
    gpu = os.path.abspath(args.gpu_dir)
    cxx = find_cxx()

    failed = False
    tmp = tempfile.mkdtemp(prefix="vsg-precheck-")
    try:
        code, out = check_device(gpu, cxx, tmp)
        print(f"device side (launchers.cu + kernels)  {'PASS' if code == 0 else 'FAIL'}")
        if out:
            print(out)
        failed |= code != 0

        code, out = check_bindings(gpu, cxx)
        if code is None:
            print(f"bindings.cpp                          SKIP  ({out})")
        else:
            print(f"bindings.cpp                          {'PASS' if code == 0 else 'FAIL'}")
            if out:
                print(out)
            failed |= code != 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("structure only - cuda semantics are checked by nvcc on the pod, not here")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
