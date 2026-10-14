#!/bin/bash
# =============================================================================
# run_performance_benchmark.sh
# GPU vs CPU calibration performance benchmark for psana2
#
# Runs on a GPU node (sdfampereNNN).  Submits a GPU allocation via srun and
# executes the Python benchmark script inside it.
#
# Usage (from ~/lcls2):
#   sh psana/psana/gpu/scripts/run_performance_benchmark.sh
#   sh psana/psana/gpu/scripts/run_performance_benchmark.sh --n-events 100 --batch-size 10
#
# All arguments are forwarded to the Python benchmark script.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# Forward any CLI arguments to the Python script
PYTHON_ARGS="$*"

srun -p ampere -A lcls -N1 --ntasks=1 --gres=gpu:a100:1 -t 00:30:00 \
    bash -c "
set -e
export OMPI_MCA_btl=^smcuda
unset PS_TEST_GPU_STREAM_IDS

cd '${REPO_ROOT}'
source setup_env.sh 2>/dev/null

python3 '${SCRIPT_DIR}/../gpu_performance_benchmark.py' ${PYTHON_ARGS}
" 2>&1 | grep -v "shmem: mmap\|create_and_attach\|unable to create shared\|coordinating structure"
