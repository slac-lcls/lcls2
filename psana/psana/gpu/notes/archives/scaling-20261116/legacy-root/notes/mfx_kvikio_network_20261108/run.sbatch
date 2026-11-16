#!/usr/bin/env bash
#SBATCH --account=lcls:data@ampere
#SBATCH --qos=normal
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:25:00
set -euo pipefail
source_dir=${1:?absolute source directory}
output=${2:?new scratch output directory}
umask 0007
node_root=${SLURM_TMPDIR:-/lscratch/$USER}
local_root="$node_root/mfx-kvikio-${SLURM_JOB_ID}"
mkdir -p "$node_root"
mkdir -m 700 "$local_root"
export TMPDIR="$local_root/tmp" CUPY_CACHE_DIR="$local_root/cupy-cache"
mkdir "$TMPDIR" "$CUPY_CACHE_DIR"
runtime="$source_dir/../mfx_expand10_stage2_20261105/runtime.sh"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
sha256sum "$source_dir"/*.py "$source_dir/run.sbatch" "$runtime"
srun --ntasks=1 --cpus-per-task=16 --cpu-bind=cores bash "$runtime" "$source_dir/campaign.py" "$output"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
