#!/bin/bash
# =============================================================================
# run_mpi_performance_benchmark.sh
# Multi-GPU multi-node psana2 MPI performance benchmark
#
# Launches gpu_mpi_benchmark.py with a configurable MPI topology.
# Measures aggregate GPU throughput across all BD ranks and reports
# scaling efficiency vs the single-GPU baseline.
#
# Default topology  (adjustable via env vars below):
#   N_NODES=1, N_GPUS_PER_NODE=2, PS_EB_NODES=1
#   → 4 total ranks: smd0(0) + eb(1) + bd(2) + bd(3)
#   → 2 GPU BD ranks, one per A100 on one node
#
# For 2 nodes × 2 GPUs each (8 total ranks, 6 BD GPUs):
#   N_NODES=2 N_GPUS_PER_NODE=2 PS_EB_NODES=1 sh run_mpi_performance_benchmark.sh
#
# Usage (from ~/lcls2 on a login node):
#   sh psana/psana/gpu/scripts/run_mpi_performance_benchmark.sh
#   sh psana/psana/gpu/scripts/run_mpi_performance_benchmark.sh --n-events 100
#   sh psana/psana/gpu/scripts/run_mpi_performance_benchmark.sh --scaling
#
#   # Pass single-GPU baseline for scaling efficiency calculation:
#   SINGLE_GPU_MS=0.35 sh run_mpi_performance_benchmark.sh
#
# Environment variables:
#   N_NODES               Number of nodes (default 1)
#   N_GPUS_PER_NODE       A100s per node (default 2)
#   PS_EB_NODES           EB ranks (default 1)
#   SINGLE_GPU_MS         Single-GPU amortised ms/event from
#                         run_performance_benchmark.sh (optional)
#   PSANA_GPU_TEST_SMD_GLOB  Override default MFX SMD glob (unused; exp/run/dir used)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BENCHMARK="${SCRIPT_DIR}/../gpu_mpi_benchmark.py"

# --- Topology config -------------------------------------------------------
N_NODES="${N_NODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
PS_EB_NODES_VAL="${PS_EB_NODES:-1}"

# Total MPI ranks = 1 smd0 + PS_EB_NODES eb + N_NODES×N_GPUS_PER_NODE bd
N_BD=$(( N_NODES * N_GPUS_PER_NODE ))
N_TOTAL=$(( 1 + PS_EB_NODES_VAL + N_BD ))

# Single-GPU baseline for scaling efficiency (from run_performance_benchmark.sh)
SGL_MS="${SINGLE_GPU_MS:-0.0}"

# --- Forward CLI args to the Python benchmark ------------------------------
PYTHON_ARGS="$*"
if [ -n "${SGL_MS}" ] && [ "${SGL_MS}" != "0.0" ]; then
    PYTHON_ARGS="--single-gpu-baseline-ms ${SGL_MS} ${PYTHON_ARGS}"
fi

# --- Conda env paths -------------------------------------------------------
CONDA_ENV="/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/ps_20241122"
MPIRUN="${CONDA_ENV}/bin/mpirun"
PYTHON="${CONDA_ENV}/bin/python3"

# Compute PYTHONPATH for the lcls2 install tree (same logic as setup_env.sh).
PYVER=$("${PYTHON}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
INSTALL_PYTHONPATH="${REPO_ROOT}/install/lib/python${PYVER}/site-packages"

# Discover CuPy nvrtc/runtime lib dirs (same logic as setup_env.sh) so that
# CuPy RawKernel JIT compilation works on the GPU node.
CUDA_LIBS=$("${PYTHON}" -c "
import importlib.util, os
dirs = []
for pkg in ('nvidia.cuda_nvrtc', 'nvidia.cuda_runtime'):
    spec = importlib.util.find_spec(pkg)
    if spec and spec.submodule_search_locations:
        d = os.path.join(list(spec.submodule_search_locations)[0], 'lib')
        if os.path.isdir(d):
            dirs.append(d)
print(':'.join(dirs))
" 2>/dev/null)

echo "Topology: ${N_TOTAL} ranks  (smd0=1, eb=${PS_EB_NODES_VAL}, bd=${N_BD})"
echo "Hardware: ${N_NODES} node(s) × ${N_GPUS_PER_NODE} A100(s) = ${N_BD} GPUs"
echo "PYTHONPATH: ${INSTALL_PYTHONPATH}"
echo ""

srun \
    -p ampere \
    -A lcls \
    -N "${N_NODES}" \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --gres="gpu:a100:${N_GPUS_PER_NODE}" \
    --gpu-bind=none \
    -t 00:30:00 \
    bash -c "
set -e
export OMPI_MCA_btl='^smcuda'
export PS_EB_NODES=${PS_EB_NODES_VAL}
export PS_SRV_NODES=0
export SLURM_GPUS_ON_NODE=${N_GPUS_PER_NODE}
unset PS_TEST_GPU_STREAM_IDS

export PYTHONPATH='${INSTALL_PYTHONPATH}'
if [ -n '${CUDA_LIBS}' ]; then
    export LD_LIBRARY_PATH='${CUDA_LIBS}'\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}
fi

mkdir -p /lscratch/\${USER:-nobody}/tmp && export TMPDIR=/lscratch/\${USER:-nobody}/tmp

'${MPIRUN}' -n ${N_TOTAL} --oversubscribe --bind-to none \
    '${PYTHON}' '${BENCHMARK}' ${PYTHON_ARGS}
" 2>&1 | grep -v "shmem: mmap\|create_and_attach\|unable to create shared\|coordinating structure"
