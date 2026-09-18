#!/bin/bash
#SBATCH --job-name=vecstore-sanitize
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%x-%j.out

# compute-sanitizer over the kernel tests: memcheck for out-of-bounds and
# misaligned accesses, racecheck for shared-memory hazards. Uses the
# extension discovery.sbatch built; submit from the repo root:
#
#   sbatch --account=<project_id> gpu/slurm/sanitize.sbatch

set -eo pipefail

module purge
module load gcc/13.3.0 cuda/12.6.3 python/3.11.9
source "${VSG_VENV:-$HOME/venvs/vecstore}/bin/activate"
set -u

cd "$SLURM_SUBMIT_DIR"
echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# an unbuilt extension would make every gpu test skip and the run pass
python3 -c "import sys; sys.path.insert(0, 'gpu'); import vecstore_gpu; sys.exit(0 if vecstore_gpu.is_available() else 1)" \
  || { echo "extension not built - run discovery.sbatch first" >&2; exit 1; }

tests="gpu/tests/test_k1_distances.py gpu/tests/test_k2_topk.py gpu/tests/test_k3_hnsw.py"
status=0
for tool in memcheck racecheck; do
  echo "=== $tool ==="
  if compute-sanitizer --tool "$tool" --error-exitcode 9 python3 -m pytest $tests -q; then
    echo "$tool: clean"
  else
    echo "$tool: FAILED"
    status=1
  fi
done
exit $status
