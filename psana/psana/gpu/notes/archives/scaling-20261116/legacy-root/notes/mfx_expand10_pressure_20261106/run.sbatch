#!/usr/bin/env bash
#SBATCH --account=lcls:data@ampere
#SBATCH --qos=normal
#SBATCH --partition=ampere
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:05:00
set -euo pipefail
source_dir=${1:?absolute source directory}
output=${2:?new project scratch output directory}
umask 0007
node_root=${SLURM_TMPDIR:-/lscratch/$USER}
local_root="$node_root/mfx-pressure-${SLURM_JOB_ID}"
mkdir -p "$node_root"
mkdir -m 700 "$local_root"
mkdir -m 770 "$output"
export TMPDIR="$local_root/tmp"
export CUPY_CACHE_DIR="$local_root/cupy-cache"
mkdir "$TMPDIR" "$CUPY_CACHE_DIR"
runtime="$source_dir/../mfx_expand10_stage2_20261105/runtime.sh"
bash "$runtime" -m unittest discover -s "$source_dir" -p test_diagnose.py -v
bash "$runtime" -m unittest discover -s "$source_dir/../mfx_expand10_performance_20261106" -p test_preparation.py -v
bash "$runtime" "$source_dir/diagnose.py" --output "$output/one-4g-8m" --streams 1
bash "$runtime" "$source_dir/diagnose.py" --output "$output/two-4g-8m" --streams 2
bash "$runtime" "$source_dir/diagnose.py" --output "$output/two-8g-8m" --streams 2 --budget-gib 8
bash "$runtime" "$source_dir/diagnose.py" --output "$output/two-4g-512k" --streams 2 --target-bytes 524288
