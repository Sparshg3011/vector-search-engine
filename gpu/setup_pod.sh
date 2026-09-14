#!/usr/bin/env bash
# pod bootstrap: deps -> build -> phase-0 gates. run ON the pod from the
# repo root: bash gpu/setup_pod.sh
#
# needs a cuda 12 *devel* image (nvcc present) - runtime/base images have
# no compiler. idempotent, safe to re-run.
set -euo pipefail

# ubuntu 24.04 images refuse system-wide pip installs without this;
# older pip ignores it
export PIP_BREAK_SYSTEM_PACKAGES=1

have_python_headers() {
  python3 -c 'import os, sysconfig; raise SystemExit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)'
}

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
have_python_headers || { echo "Python.h not found - install python3-dev" >&2; exit 1; }

# cmake 3.x: pybind11 2.13 predates cmake 4's removal of old policies
python3 -m pip install -q "cmake>=3.24,<4" ninja numpy pytest matplotlib h5py

CMAKE="$(python3 -c 'import cmake, os; print(os.path.join(cmake.CMAKE_BIN_DIR, "cmake"))')"
NINJA="$(python3 -c 'import ninja, os; print(os.path.join(ninja.BIN_DIR, "ninja"))')"
NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
if [ ! -x "$NVCC" ]; then
  echo "nvcc not found - this image has no cuda compiler; pick a *-devel image" >&2
  exit 1
fi
echo "$("$CMAKE" --version | head -1) | $("$NVCC" --version | grep -o 'release [0-9.]*')"

"$CMAKE" -S gpu -B gpu/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM="$NINJA" \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DPYBIND11_FINDPYTHON=ON \
  -DPython_EXECUTABLE="$(command -v python3)"
"$CMAKE" --build gpu/build

# nonzero exit here stops the script - gates must pass before any paid
# benchmark work starts
python3 gpu/check_env.py

echo
echo "all gates passed. next steps:"
echo "  python3 -m pytest gpu/tests -v"
echo "  python3 gpu/bench/run.py --help"
