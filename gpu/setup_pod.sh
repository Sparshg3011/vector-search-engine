#!/usr/bin/env bash
# pod bootstrap: deps -> build -> phase-0 gates. run ON the pod from the
# repo root: bash gpu/setup_pod.sh  (or chmod +x gpu/setup_pod.sh once).
# works on runpod/brev cuda images (nvidia/cuda:12.x devel-class) and
# assumes nvcc is already present — if it isn't, the image is wrong:
# pick a *-devel image, not runtime/base. idempotent — safe to re-run.
set -euo pipefail

# system deps — most cuda pod images already ship these; only touch apt
# when it exists and something is actually missing
if command -v apt-get >/dev/null 2>&1; then
  missing=()
  command -v cmake >/dev/null 2>&1 || missing+=(cmake)
  command -v ninja >/dev/null 2>&1 || missing+=(ninja-build)
  command -v git >/dev/null 2>&1 || missing+=(git)
  command -v rsync >/dev/null 2>&1 || missing+=(rsync)
  if [ "${#missing[@]}" -gt 0 ]; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
  fi
fi

python3 -m pip install -q numpy pytest matplotlib h5py

cmake -S gpu -B gpu/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build gpu/build

# nonzero exit here stops the script (set -e) — gates must pass before
# any paid kernel work starts
python3 gpu/check_env.py

echo
echo "all gates passed. next steps:"
echo "  pytest gpu/tests -v"
echo "  python gpu/bench/run.py --help"
