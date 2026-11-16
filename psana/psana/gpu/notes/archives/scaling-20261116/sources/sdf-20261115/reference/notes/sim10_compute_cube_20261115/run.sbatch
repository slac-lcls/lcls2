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
#SBATCH --job-name=sim10-compute-cube
#SBATCH --signal=B:TERM@120
set -euo pipefail
attempt=${1:?prepared shared-scratch attempt}
review=${2:?independent implementation review JSON}
review_sha256=${3:?review receipt SHA256}
source_dir="$attempt/source/notes/sim10_compute_cube_20261115"
export TMPDIR="$attempt/tmp" CUPY_CACHE_DIR="$attempt/cupy-cache"
mkdir -p "$TMPDIR" "$CUPY_CACHE_DIR"
child=''
stop_child() {
    status=$?
    trap - EXIT TERM INT
    if [[ -n "$child" ]]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" || true
    fi
    exit "$status"
}
trap stop_child EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
bash "$source_dir/../jf_raw_cache_20261109/runtime.sh" \
    /sdf/scratch/users/m/monarin/gpu-validation/kvikio-2610-validation-20261015-r1/venv/bin/python \
    "$source_dir/launch.py" "$attempt" "$review" "$review_sha256" &
child=$!
wait "$child"
child=''
