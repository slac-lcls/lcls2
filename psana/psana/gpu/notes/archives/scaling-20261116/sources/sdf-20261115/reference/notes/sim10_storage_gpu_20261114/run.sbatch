#!/usr/bin/env bash
#SBATCH --account=lcls:data@ampere
#SBATCH --qos=normal
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --job-name=sim10-storage-gpu
#SBATCH --signal=B:TERM@120
set -euo pipefail
attempt=${1:?unique prepared shared-scratch attempt}
reviewed_manifest=${2:?independently reviewed manifest SHA256}
export SIM10_EXPECTED_NODE=${3:?reviewed node name}
source_dir="$attempt/source/notes/sim10_storage_gpu_20261114"
export TMPDIR="$attempt/tmp" CUPY_CACHE_DIR="$attempt/cupy-cache"
mkdir -p "$TMPDIR" "$CUPY_CACHE_DIR"
child=''
cleanup() {
    status=$?
    trap - EXIT TERM INT
    if [[ -n "$child" ]]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
bash "$source_dir/../jf_raw_cache_20261109/runtime.sh" \
    /sdf/scratch/users/m/monarin/gpu-validation/kvikio-2610-validation-20261015-r1/venv/bin/python \
    "$source_dir/launch.py" "$attempt" "$reviewed_manifest" &
child=$!
wait "$child"
child=''
