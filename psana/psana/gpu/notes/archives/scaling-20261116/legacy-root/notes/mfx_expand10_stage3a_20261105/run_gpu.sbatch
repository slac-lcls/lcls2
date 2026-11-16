#!/usr/bin/env bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
set -euo pipefail
stage_notes=${1:?absolute notes directory}
fixture_root=${2:?existing fixture}
result_root=${3:?new result directory}
batch_count=${4:-7}
extra_args=()
if [[ ${5:-} == save ]]; then extra_args+=(--save-cube); fi
export CUPY_CACHE_DIR=$(dirname "$result_root")/cupy-cache
export TMPDIR=$(dirname "$result_root")/tmp
mkdir -p "$CUPY_CACHE_DIR" "$TMPDIR"
exec bash "$stage_notes/../mfx_expand10_stage2_20261105/runtime.sh" "$stage_notes/acceptance.py" --root "$fixture_root" --output "$result_root" --batch-size "$batch_count" "${extra_args[@]}"
