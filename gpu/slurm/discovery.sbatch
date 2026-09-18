#!/bin/bash
#SBATCH --job-name=vecstore-gpu
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

# Build, test and benchmark on a USC Discovery gpu node. Compute nodes are
# offline, so the network steps run on a login node first (gpu/SPEC.md,
# "Running on a Slurm cluster"). Submit from the repo root:
#
#   sbatch --account=<project_id> gpu/slurm/discovery.sbatch
#
# BENCH=0 stops after the tests. VSG_VENV overrides the venv location.

set -eo pipefail

module purge
module load gcc/13.3.0 cuda/12.6.3 python/3.11.9

venv="${VSG_VENV:-$HOME/venvs/vecstore}"
if [ ! -f "$venv/bin/activate" ]; then
  echo "no venv at $venv - create it on a login node first (gpu/SPEC.md)" >&2
  exit 1
fi
source "$venv/bin/activate"
set -u

cd "$SLURM_SUBMIT_DIR"
echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# one binary for every generation the job may land on (A100 8.0, A40
# 8.6, L40S 8.9), from a fresh build dir: cmake caches the gpu it was
# first configured on, so a stale dir would silently target the wrong one
export CUDAARCHS="${CUDAARCHS:-80;86;89}"
rm -rf gpu/build
bash gpu/setup_pod.sh --build-only --profiler-optional
python3 -m pytest gpu/tests -v

if [ "${BENCH:-1}" = 1 ]; then
  for f in data/sift.gpu.npz data/sift-128-euclidean-vecstore.npz data/sift-128-euclidean-queries.npy; do
    if [ ! -f "$f" ]; then
      echo "missing $f - upload or fetch it on a login node, or submit with BENCH=0" >&2
      exit 1
    fi
  done
  out="gpu/bench/results/discovery-${SLURM_JOB_ID}.json"
  python3 gpu/bench/run.py \
    --gpu-index data/sift.gpu.npz \
    --source-index data/sift-128-euclidean-vecstore.npz \
    --queries data/sift-128-euclidean-queries.npy \
    --baselines cpu_hnsw,gpu_flat_hand,gpu_flat_cublas,gpu_hnsw \
    --batches 1,128,2048 \
    --ef 50,200 \
    --out "$out"
  python3 gpu/bench/plot.py "$out"
fi
