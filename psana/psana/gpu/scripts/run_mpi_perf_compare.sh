#!/bin/bash
# =============================================================================
# run_mpi_perf_compare.sh
# Multi-node CPU vs GPU MPI performance benchmark
#
# Runs both CPU (numpy) and GPU (CUDA) calibration paths in the same MPI
# topology and reports aggregate throughput, per-rank breakdown, and speedup.
#
# Uses a proper multi-task Slurm allocation (one Slurm task per MPI rank)
# so that each rank has its own CPU core and events are distributed evenly
# across BD ranks — unlike the --oversubscribe single-task approach where
# one rank can starve others.
#
# MPI is wired via srun --mpi=pmi2 with the conda Python path and
# psana2 env vars exported from the calling shell via --export=ALL.
#
# Default topology  (N_NODES=1, N_GPUS_PER_NODE=2, PS_EB_NODES=1):
#   4 total ranks: smd0(0) + eb(1) + bd(2) + bd(3)
#   --ntasks=4, --ntasks-per-node=4, --gres=gpu:a100:2
#   CPU path: 2 BD ranks run numpy calibration
#   GPU path: 2 BD ranks run CUDA calibration (1 GPU each)
#
# For 2 nodes × 2 GPUs (6 BD ranks, 8 total ranks):
#   N_NODES=2 N_GPUS_PER_NODE=2 sh run_mpi_perf_compare.sh
#
# Usage (from ~/lcls2 on a login node):
#   sh psana/psana/gpu/scripts/run_mpi_perf_compare.sh
#   sh psana/psana/gpu/scripts/run_mpi_perf_compare.sh --n-events 100
#   sh psana/psana/gpu/scripts/run_mpi_perf_compare.sh --scaling
#   sh psana/psana/gpu/scripts/run_mpi_perf_compare.sh --skip-cpu
#
# Environment variables:
#   N_NODES           Number of nodes (default 1)
#   N_GPUS_PER_NODE   A100s per node  (default 2)
#   N_BD_PER_GPU      BD ranks per GPU (default 1; set >1 to share a GPU)
#   PS_EB_NODES       EB ranks        (default 1)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BENCHMARK="${SCRIPT_DIR}/../gpu_mpi_perf_compare.py"

# ── Topology ─────────────────────────────────────────────────────────────────
N_NODES="${N_NODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
N_BD_PER_GPU="${N_BD_PER_GPU:-1}"
PS_EB_NODES_VAL="${PS_EB_NODES:-1}"
N_BD=$(( N_NODES * N_GPUS_PER_NODE * N_BD_PER_GPU ))
N_TOTAL=$(( 1 + PS_EB_NODES_VAL + N_BD ))
N_TASKS_PER_NODE=$(( N_TOTAL / N_NODES ))

PYTHON_ARGS="$*"

# Full path to conda Python — accessible from all nodes without sourcing.
PYTHON="/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/ps_20241122/bin/python3"

echo "Topology: ${N_TOTAL} ranks  (smd0=1, eb=${PS_EB_NODES_VAL}, bd=${N_BD}, ${N_BD_PER_GPU}/GPU)"
echo "Hardware: ${N_NODES} node(s) × ${N_GPUS_PER_NODE} A100(s) × ${N_BD_PER_GPU} BD/GPU = ${N_BD} BD ranks"
echo "Slurm:    --ntasks=${N_TOTAL} --ntasks-per-node=${N_TASKS_PER_NODE} --mpi=pmix"
echo ""

# ── Set env vars in calling shell before srun so --export=ALL forwards them ──
source "${REPO_ROOT}/setup_env.sh" 2>/dev/null || true

export PS_EB_NODES="${PS_EB_NODES_VAL}"
export PS_SRV_NODES=0
export SLURM_GPUS_ON_NODE="${N_GPUS_PER_NODE}"
export OMPI_MCA_btl='^smcuda'
unset  PS_TEST_GPU_STREAM_IDS
mkdir -p "/lscratch/${USER:-nobody}/tmp"
export TMPDIR="/lscratch/${USER:-nobody}/tmp"

# ── Submit: one Slurm task per MPI rank, srun --mpi=pmi2 wires them together ─
srun \
    -p ampere \
    -A lcls \
    -N "${N_NODES}" \
    --ntasks="${N_TOTAL}" \
    --ntasks-per-node="${N_TASKS_PER_NODE}" \
    --cpus-per-task=2 \
    --gres="gpu:a100:${N_GPUS_PER_NODE}" \
    --gpu-bind=none \
    --mpi=pmix \
    --export=ALL \
    -t 00:30:00 \
    bash -c '
        # srun sets SLURM_PROCID; mpirun inside srun sets OMPI_COMM_WORLD_RANK.
        R=${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-0}}
        PS_EB='"${PS_EB_NODES_VAL}"'
        N_GPUS='"${N_GPUS_PER_NODE}"'
        if [ "${R}" -gt "${PS_EB}" ]; then
            BD_IDX=$(( R - PS_EB - 1 ))
            export CUDA_VISIBLE_DEVICES=$(( BD_IDX % N_GPUS ))
        else
            # smd0 and EB: disable GPU so they do not compete with BD ranks
            export CUDA_VISIBLE_DEVICES=""
        fi
        exec '"\"${PYTHON}\" \"${BENCHMARK}\" ${PYTHON_ARGS}"'
    ' \
2>&1 | grep -v "shmem: mmap\|create_and_attach\|unable to create shared\|coordinating structure\|UCX  WARN\|A requested component\|was not found\|unable to be opened\|not installed\|shared libraries\|that the component\|unable to be found\|PMIx stopped\|Framework:\|Component:\|Host:\|^---"
