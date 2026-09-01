#!/usr/bin/env bash
# push the repo mac -> pod, optionally run a command there. the whole
# push-to-result loop must stay under 2 minutes; rsync deltas make
# repeat pushes near-instant. chmod +x gpu/sync.sh to run it directly.
#
#   POD_SSH=root@1.2.3.4 gpu/sync.sh                       # sync only
#   POD_SSH=root@1.2.3.4 gpu/sync.sh 'python3 -m pytest gpu/tests -x -q'
#   gpu/sync.sh --with-data 'python3 gpu/bench/run.py'     # ship data/*.gpu.npz too
#
# env: POD_SSH (required, e.g. root@1.2.3.4), POD_PORT (default 22),
#      POD_DIR (default /root/vector-search-engine)
set -euo pipefail

: "${POD_SSH:?set POD_SSH, e.g. POD_SSH=root@1.2.3.4}"
POD_PORT="${POD_PORT:-22}"
POD_DIR="${POD_DIR:-/root/vector-search-engine}"

# repo root is one level up from this script
repo_root="$(cd "$(dirname "$0")/.." && pwd)"

filters=()
if [ "${1:-}" = "--with-data" ]; then
  shift
  # first match wins in rsync: let the benchmark payloads through, keep
  # the rest of data/ (raw hdf5, cpu-side npz) out
  filters+=(--include='data/' --include='data/*.gpu.npz' --exclude='data/*')
fi

filters+=(
  --exclude='.git' --exclude='.venv' --exclude='data/' --exclude='results/'
  --exclude='gpu/build' --exclude='gpu/bench/results' --exclude='__pycache__'
  --exclude='*.egg-info' --exclude='*.hdf5' --exclude='*.npz'
)

# --delete keeps the pod an exact mirror; excluded paths (gpu/build,
# bench results) are protected from deletion, so pod builds survive
rsync -az --delete -e "ssh -p $POD_PORT" "${filters[@]}" \
  "$repo_root/" "$POD_SSH:$POD_DIR/"

if [ "$#" -gt 0 ]; then
  ssh -p "$POD_PORT" "$POD_SSH" "cd '$POD_DIR' && $*"
else
  echo "synced. loop with: gpu/sync.sh 'python3 -m pytest gpu/tests -x -q'"
fi
