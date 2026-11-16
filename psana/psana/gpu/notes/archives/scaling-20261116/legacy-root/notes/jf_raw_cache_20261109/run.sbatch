#!/usr/bin/env bash
#SBATCH --account=lcls:data@ampere
#SBATCH --qos=normal
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --mem=700G
#SBATCH --time=01:30:00
#SBATCH --signal=B:TERM@180
set -euo pipefail
source_dir=${1:?source}
output=${2:?new output}
umask 0007
local_root="${SLURM_TMPDIR:-/lscratch/$USER}/jf-raw-cache-${SLURM_JOB_ID}"
mkdir -m 700 "$local_root"
export TMPDIR="$local_root/tmp" CUPY_CACHE_DIR="$local_root/cupy-cache" JF_LOCAL_INPUT="$local_root/xtc"
mkdir "$TMPDIR" "$CUPY_CACHE_DIR"
cleanup() {
    result=$?
    trap - EXIT TERM INT
    if [[ -n "${bench_pid:-}" ]]; then
        kill -TERM "$bench_pid" 2>/dev/null || true
        wait "$bench_pid" || true
    fi
    rm -rf -- "$local_root"
    if [[ -d "$output" ]]; then
        /sdf/group/lcls/ds/ana/sw/conda2/inst/envs/xpp_drp_gpu_311/bin/python - "$local_root" "$output" "$result" <<'PYEND'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);out=Path(sys.argv[2]);(out/'root-cleanup.json').write_text(json.dumps(dict(root=str(root),removed=not root.exists(),exit_code=int(sys.argv[3])))+'\n')
PYEND
    fi
    exit "$result"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export PS_SMD_N_EVENTS=1000
sha256sum "$source_dir"/*.py "$source_dir"/*.sh "$source_dir"/*.sbatch
bash "$source_dir/runtime.sh" /sdf/scratch/users/m/monarin/gpu-validation/kvikio-2610-validation-20261015-r1/venv/bin/python "$source_dir/campaign.py" "$output" &
bench_pid=$!
wait "$bench_pid"
unset bench_pid
