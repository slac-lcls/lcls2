#!/usr/bin/env bash
#SBATCH --account=lcls:data@ampere
#SBATCH --qos=normal
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --mem=700G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@120
set -euo pipefail
source_dir=${1:?source}
output=${2:?output}
node_root="${SLURM_TMPDIR:-/lscratch/$USER}/sim10-100k-bds-${SLURM_JOB_ID}"
mkdir -m 700 "$node_root"
export TMPDIR="$node_root/tmp" CUPY_CACHE_DIR="$node_root/cupy-cache"
mkdir "$TMPDIR" "$CUPY_CACHE_DIR"
child=''
cleanup() {
    rc=$?
    trap - EXIT TERM INT
    if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" || true; fi
    rm -rf -- "$node_root"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
bash "$source_dir/../jf_raw_cache_20261109/runtime.sh" /sdf/scratch/users/m/monarin/gpu-validation/kvikio-2610-validation-20261015-r1/venv/bin/python "$source_dir/campaign.py" "$output" &
child=$!
wait "$child"
child=''
