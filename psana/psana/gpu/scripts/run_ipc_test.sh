#!/bin/bash
# Run the CUDA IPC shared calibconst test.
# Uses 2 MPI ranks on the same GPU — no EB/smd0, just the IPC mechanism.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TEST="${SCRIPT_DIR}/../test_ipc_sharing.py"
CONDA_ENV="/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/ps_20241122"

srun -p ampere -A lcls -N1 --ntasks=1 --cpus-per-task=4 \
     --gres=gpu:a100:1 --gpu-bind=none -t 00:05:00 bash -c "
export CUDA_VISIBLE_DEVICES=0   # both ranks use the same GPU
export OMPI_MCA_btl='^smcuda'
cd '${REPO_ROOT}' && source setup_env.sh 2>/dev/null
# 2 ranks: both are BD workers on the same GPU (no EB rank needed now)
'${CONDA_ENV}/bin/mpirun' -n 2 --oversubscribe --bind-to none \
    '${CONDA_ENV}/bin/python3' '${TEST}' 2>&1
" 2>&1 | grep -v "shmem: mmap\|create_and_attach\|unable to create\|coordinating\|UCX"
