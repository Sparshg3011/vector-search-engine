#!/usr/bin/env bash
# deps -> build -> phase-0 gates, run from the repo root.
#
#   bash gpu/setup_pod.sh              both phases, on one box with a gpu and network
#   bash gpu/setup_pod.sh --deps-only  python packages only; needs network, no gpu
#   bash gpu/setup_pod.sh --build-only build + gates; needs a gpu, no network
#
# the split exists for clusters whose compute nodes are offline: run
# --deps-only on a login node, --build-only inside the gpu job. add
# --profiler-optional where ncu counters may be unavailable (no root) so
# a blocked profiler does not fail the run.
#
# needs a cuda 12 *devel* toolchain (nvcc) for the build phase. idempotent.
set -euo pipefail

deps=1
build=1
check_args=()
for arg in "$@"; do
  case "$arg" in
    --deps-only) build=0 ;;
    --build-only) deps=0 ;;
    --profiler-optional) check_args+=(--profiler-optional) ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# ubuntu 24.04 images refuse system-wide pip installs without this; pip
# inside a venv, and older pip, ignore it
export PIP_BREAK_SYSTEM_PACKAGES=1

have_python_headers() {
  python3 -c 'import os, sysconfig; raise SystemExit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)'
}

if [ "$deps" = 1 ]; then
  # cmake is deliberately NOT taken from apt: ubuntu 22.04 ships 3.22 and
  # CMakeLists.txt needs 3.24
  if command -v apt-get >/dev/null 2>&1; then
    missing=()
    command -v git >/dev/null 2>&1 || missing+=(git)
    command -v rsync >/dev/null 2>&1 || missing+=(rsync)
    have_python_headers || missing+=(python3-dev)
    if [ "${#missing[@]}" -gt 0 ]; then
      apt-get update
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
    fi
  fi

  # cmake 3.x: pybind11 2.13's fallback predates cmake 4's policy removals.
  # pybind11 comes from pip so the build phase never touches the network.
  python3 -m pip install -q "cmake>=3.24,<4" ninja "pybind11>=2.13" \
    numpy pytest matplotlib h5py
  echo "python packages installed for $(python3 --version)"
fi

[ "$build" = 1 ] || exit 0

have_python_headers || { echo "Python.h not found - install python3-dev (or load a python module)" >&2; exit 1; }

tool() {
  python3 -c "import $1, os; print(os.path.join($1.$2, '$1'))" 2>/dev/null \
    || { echo "$1 is not installed for $(command -v python3) - run: bash gpu/setup_pod.sh --deps-only" >&2; exit 1; }
}
CMAKE="$(tool cmake CMAKE_BIN_DIR)"
NINJA="$(tool ninja BIN_DIR)"
PYBIND11_DIR="$(python3 -m pybind11 --cmakedir 2>/dev/null)" \
  || { echo "pybind11 is not installed - run: bash gpu/setup_pod.sh --deps-only" >&2; exit 1; }

NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
if [ ! -x "$NVCC" ]; then
  echo "nvcc not found - use a cuda *devel* image, or load a cuda module" >&2
  exit 1
fi
echo "$("$CMAKE" --version | head -1) | nvcc $("$NVCC" --version | grep -o 'release [0-9.]*')"

host_compiler=()
if command -v g++ >/dev/null 2>&1; then
  # nvcc otherwise picks its own g++, which can differ from a loaded module
  host_compiler=(-DCMAKE_CXX_COMPILER="$(command -v g++)" -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++)")
fi

"$CMAKE" -S gpu -B gpu/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM="$NINJA" \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  "${host_compiler[@]}" \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  -DPYBIND11_FINDPYTHON=ON \
  -DPython_EXECUTABLE="$(command -v python3)"
"$CMAKE" --build gpu/build

# nonzero exit stops the script - gates must pass before any benchmark
# work starts
python3 gpu/check_env.py ${check_args[@]+"${check_args[@]}"}

echo
echo "all gates passed. next steps:"
echo "  python3 -m pytest gpu/tests -v"
echo "  python3 gpu/bench/run.py --help"
