#!/usr/bin/env bash
#SBATCH --account=lcls:data@ampere
#SBATCH --qos=normal
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --mem=700G
#SBATCH --time=00:15:00
set -euo pipefail
source_dir=${1:?source}
output=${2:?output}
local_root="${SLURM_TMPDIR:-/lscratch/$USER}/jf-legacy-control-${SLURM_JOB_ID}"
mkdir -m 700 "$local_root"
export TMPDIR="$local_root/tmp" CUPY_CACHE_DIR="$local_root/cupy-cache"
mkdir "$TMPDIR" "$CUPY_CACHE_DIR"
trap 'rm -rf -- "$local_root"' EXIT
bash "$source_dir/../jf_raw_cache_20261109/runtime.sh" /sdf/scratch/users/m/monarin/gpu-validation/kvikio-2610-validation-20261015-r1/venv/bin/python "$source_dir/run.py" "$output"
