#!/usr/bin/env bash
# Single-node MPI/GPU transport smoke check, not a numerical or performance test.
# Activate the build's Python environment before running this launcher.
# Default: one SMD0 + one EB + two BDs on two A100s.
# Options are forwarded unchanged to gpu_multi_rank_smoke.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/gpu_multi_rank_smoke.py"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
PSANA_GPU_TEST_PREFIX="${PSANA_GPU_TEST_PREFIX:-${REPO_ROOT}/install_psana}"

if [[ ! "$N_GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]] || (( N_GPUS_PER_NODE < 2 )); then
    echo "Need N_GPUS_PER_NODE >= 2" >&2
    exit 1
fi
if [[ "${PS_EB_NODES:-1}" != 1 || "${PS_SRV_NODES:-0}" != 0 ]]; then
    echo "Supported topology: PS_EB_NODES=1, PS_SRV_NODES=0" >&2
    exit 1
fi
N_TOTAL=$((2 + N_GPUS_PER_NODE))

# Use a selected local install, without resetting the caller's conda environment.
# Missing/broken setup is fatal; do not fall back to an unrelated psana install.
source "${PSANA_GPU_TEST_PREFIX}/activate.sh"
PYTHON="$(command -v python)"
PS_PARALLEL=none CUDA_VISIBLE_DEVICES= "$PYTHON" - "$PSANA_GPU_TEST_PREFIX" <<'PY'
from pathlib import Path
import sys
import psana

prefix = Path(sys.argv[1]).resolve()
module = Path(psana.__file__).resolve()
if prefix not in module.parents:
    raise SystemExit(f"Wrong psana install: {module}; expected under {prefix}")
print(f"Python: {sys.executable}", flush=True)
print(f"psana: {module}", flush=True)
PY

export PS_EB_NODES=1 PS_SRV_NODES=0 PS_EB_NODE_LOCAL=0 PS_PARALLEL=mpi
export SLURM_GPUS_ON_NODE="$N_GPUS_PER_NODE"
export OMPI_MCA_btl='^smcuda'
export TMPDIR="${TMPDIR:-/tmp}"

echo "Topology: 1 SMD0 + 1 EB + ${N_GPUS_PER_NODE} BDs, one node"
echo "Validation: event delivery and BD/GPU participation (not pixel correctness)"

# Mask CPU roles before Python imports. For this single-EB topology, the
# BD-local index is world rank - 2, matching MPIDataSource's own GPU pinning.
# Pass executable/script/arguments positionally, never interpolate user argv.
RANK_WRAPPER='
    rank=${SLURM_PROCID:?srun must supply SLURM_PROCID}
    if (( rank >= 2 )); then
        export CUDA_VISIBLE_DEVICES=$((rank - 2))
    else
        export CUDA_VISIBLE_DEVICES=""
    fi
    exec "$@"
'
SRUN_ARGS=(
    --nodes=1 --ntasks="$N_TOTAL" --ntasks-per-node="$N_TOTAL"
    --cpus-per-task=2 --gpu-bind=none --mpi=pmix --export=ALL
    --kill-on-bad-exit=1 --time="${PSANA_GPU_TEST_TIME:-00:15:00}"
)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Using allocation ${SLURM_JOB_ID}; it must provide ${N_TOTAL} tasks and ${N_GPUS_PER_NODE} GPUs"
else
    SRUN_ARGS+=(-p ampere -A lcls --gres="gpu:a100:${N_GPUS_PER_NODE}")
fi

# No output filter or pipe: retain diagnostics and return srun's failure status.
exec srun "${SRUN_ARGS[@]}" bash -c "$RANK_WRAPPER" gpu-smoke "$PYTHON" "$TEST_SCRIPT" "$@"
