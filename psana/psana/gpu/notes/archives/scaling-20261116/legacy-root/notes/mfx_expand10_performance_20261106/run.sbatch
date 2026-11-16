#!/usr/bin/env bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
set -euo pipefail
source_dir=${1:?absolute source directory}
output=${2:?new project scratch result directory}
mode=${3:?smoke, throughput or profile}
shift 3
umask 0007
node_root=${SLURM_TMPDIR:-/lscratch/$USER}
local_root="$node_root/mfx-performance-${SLURM_JOB_ID}"
mkdir -p "$node_root"
mkdir -m 700 "$local_root"
mkdir -m 770 "$output"
export TMPDIR="$local_root/tmp"
export CUPY_CACHE_DIR="$local_root/cupy-cache"
mkdir "$TMPDIR" "$CUPY_CACHE_DIR"
runtime="$source_dir/../mfx_expand10_stage2_20261105/runtime.sh"
if [[ "$mode" == smoke ]]; then
    bash "$runtime" -m unittest discover -s "$source_dir" -p test_preparation.py -v
    bash "$runtime" "$source_dir/device_checks.py" "$output/device-checks.json"
    pilot=/sdf/scratch/users/m/monarin/gpu-validation/mfx-expand10/20261105-stage2-r1/pilot
    for streams in 1 2; do
        bash "$runtime" "$source_dir/runner.py" --root "$pilot" --output "$output/throughput-$streams" \
            --streams "$streams" --warmup 100 --bins 100
    done
    bash "$runtime" "$source_dir/runner.py" --root "$pilot" --output "$output/profile-2" \
        --mode profile --streams 2 --warmup 100 --bins 100
else
    bash "$runtime" "$source_dir/runner.py" --output "$output/case" --mode "$mode" "$@"
fi
# Keep this job's node-local cache until allocation ends; results are on shared scratch.
