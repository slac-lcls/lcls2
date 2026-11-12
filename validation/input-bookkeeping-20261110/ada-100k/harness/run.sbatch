#!/usr/bin/env bash
#SBATCH --account=lcls:prjdat21
#SBATCH --qos=normal
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclusive
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=sim10-ada-100k
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
bash "$source_dir/../jf_raw_cache_20261109/runtime.sh" /sdf/scratch/users/m/monarin/gpu-validation/kvikio-2610-validation-20261015-r1/venv/bin/python "$source_dir/audit.py" "$attempt/results" > "$attempt/audit.json"
