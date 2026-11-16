#!/usr/bin/env bash
#SBATCH --account=lcls:data@ampere
#SBATCH --qos=normal
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --exclusive
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --job-name=sim10-kvikio-bounce
#SBATCH --signal=B:TERM@120
set -euo pipefail
source_dir=${1:?source}
attempt=${2:?unique shared scratch attempt}
export SIM10_PR_SOURCE="$attempt/pr-source"
export SIM10_NATIVE_RECEIPT="$attempt/native-compatibility.json"
export TMPDIR="$attempt/tmp" CUPY_CACHE_DIR="$attempt/cupy-cache"
mkdir -p "$TMPDIR" "$CUPY_CACHE_DIR"
child=''
cleanup() {
    rc=$?
    trap - EXIT TERM INT
    if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" || true; fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
bash "$source_dir/../jf_raw_cache_20261109/runtime.sh" /sdf/scratch/users/m/monarin/gpu-validation/kvikio-2610-validation-20261015-r1/venv/bin/python "$source_dir/campaign.py" "$attempt/results" &
child=$!
wait "$child"
child=''
