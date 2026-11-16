#!/usr/bin/env bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
set -euo pipefail
stage_notes=${1:?absolute notes directory}
fixture_root=${2:?scratch fixture xtc directory}
result_root=${3:?new scratch report directory}
batch_count=${4:-3}
export CUPY_CACHE_DIR=$(dirname "$result_root")/cupy-cache
export TMPDIR=$(dirname "$result_root")/tmp
mkdir -p "$CUPY_CACHE_DIR" "$TMPDIR"
exec bash "$stage_notes/runtime.sh" "$stage_notes/gpu_accept.py" --root "$fixture_root" --output "$result_root" --batch-size "$batch_count"
