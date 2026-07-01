#!/bin/bash
# =============================================================================
# run_multi_gpu_test.sh
# Multi-GPU MPI correctness test for psana2 GPU BD ranks.
#
# Uses the same srun --mpi=pmix --export=ALL approach as
# run_mpi_perf_compare.sh: one Slurm task per MPI rank, SLURM_PROCID for
# per-rank GPU pinning, and setup_env.sh forwarded via --export=ALL.
# No mpirun, no --oversubscribe, no wrapper scripts.
#
# Topology (PS_EB_NODES=1, 4 total ranks):
#   rank 0  — SMD0
#   rank 1  — EB
#   rank 2  — BD: SLURM_PROCID=2 → CUDA_VISIBLE_DEVICES=0 (A100 #0)
#   rank 3  — BD: SLURM_PROCID=3 → CUDA_VISIBLE_DEVICES=1 (A100 #1)
#
# Usage (from ~/lcls2 on a login node):
#   sh psana/psana/gpu/scripts/run_multi_gpu_test.sh
#   sh psana/psana/gpu/scripts/run_multi_gpu_test.sh --max-events 20
#
# Optional environment variables:
#   PSANA_GPU_TEST_SMD_GLOB  — override the default MFX SMD glob pattern
#   N_GPUS_PER_NODE          — number of A100s to request (default 2)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/../../tests/test_gpu_multi_rank.py"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
PS_EB_NODES_VAL="${PS_EB_NODES:-1}"
N_BD=$(( N_GPUS_PER_NODE ))          # 1 BD rank per GPU
N_TOTAL=$(( 1 + PS_EB_NODES_VAL + N_BD ))

PYTHON="/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/ps_20241122/bin/python3"

# Forward CLI args to the test script.
PYTHON_ARGS="$*"

echo "Topology: ${N_TOTAL} ranks  (smd0=1, eb=${PS_EB_NODES_VAL}, bd=${N_BD})"
echo "Hardware: 1 node × ${N_GPUS_PER_NODE} A100(s)"
echo ""

# Source setup_env.sh so PYTHONPATH and LD_LIBRARY_PATH are set in this shell;
# --export=ALL forwards them to every srun task automatically.
source "${REPO_ROOT}/setup_env.sh" 2>/dev/null || true

export PS_EB_NODES="${PS_EB_NODES_VAL}"
export PS_SRV_NODES=0
export SLURM_GPUS_ON_NODE="${N_GPUS_PER_NODE}"
export OMPI_MCA_btl='^smcuda'
unset  PS_TEST_GPU_STREAM_IDS
[ -n "${PSANA_GPU_TEST_SMD_GLOB}" ] && export PSANA_GPU_TEST_SMD_GLOB
mkdir -p "/lscratch/${USER:-nobody}/tmp" 2>/dev/null || true
export TMPDIR="${TMPDIR:-/tmp}"

# Two srun invocations depending on context:
#
#   Inside a Slurm job (SLURM_JOB_ID is set, e.g. run via sbatch or from
#   within an salloc session):
#     Use a plain step srun — no --partition, --account, --gres, or --time
#     flags.  The existing job allocation already owns the GPUs; requesting
#     them again causes "Invalid generic resource (gres) specification".
#     The parent job must have been submitted with --gres=gpu:a100:2 (or more)
#     and --ntasks >= N_TOTAL.
#
#   Outside a Slurm job (login node, e.g. sdfiana026):
#     Request a fresh allocation with the full set of flags.

_RANK_WRAPPER='
    R=${SLURM_PROCID:-0}
    PS_EB='"${PS_EB_NODES_VAL}"'
    N_GPUS='"${N_GPUS_PER_NODE}"'
    if [ "${R}" -gt "${PS_EB}" ]; then
        BD_IDX=$(( R - PS_EB - 1 ))
        export CUDA_VISIBLE_DEVICES=$(( BD_IDX % N_GPUS ))
    else
        export CUDA_VISIBLE_DEVICES=""
    fi
    exec '"\"${PYTHON}\" \"${TEST_SCRIPT}\""' '"${PYTHON_ARGS}"'
'

_FILTER='grep -v "shmem: mmap\|create_and_attach\|unable to create shared\|coordinating structure\|UCX  WARN\|A requested component\|was not found\|unable to be opened\|not installed\|shared libraries\|that the component\|unable to be found\|PMIx stopped\|Framework:\|Component:\|Host:\|^---"'

if [ -n "${SLURM_JOB_ID}" ]; then
    # ── Already inside a Slurm allocation ──────────────────────────────────
    echo "Running inside Slurm job ${SLURM_JOB_ID} — using existing allocation"
    srun \
        --ntasks="${N_TOTAL}" \
        --ntasks-per-node="${N_TOTAL}" \
        --cpus-per-task=2 \
        --gpu-bind=none \
        --mpi=pmix \
        --export=ALL \
        bash -c "${_RANK_WRAPPER}" \
    2>&1 | eval "${_FILTER}"
else
    # ── Login node — request a fresh allocation ────────────────────────────
    srun \
        -p ampere \
        -A lcls \
        -N 1 \
        --ntasks="${N_TOTAL}" \
        --ntasks-per-node="${N_TOTAL}" \
        --cpus-per-task=2 \
        --gres="gpu:a100:${N_GPUS_PER_NODE}" \
        --gpu-bind=none \
        --mpi=pmix \
        --export=ALL \
        -t 00:05:00 \
        bash -c "${_RANK_WRAPPER}" \
    2>&1 | eval "${_FILTER}"
fi
