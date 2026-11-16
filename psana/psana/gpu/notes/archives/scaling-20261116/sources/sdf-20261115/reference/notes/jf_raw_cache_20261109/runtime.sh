#!/usr/bin/env bash
set -euo pipefail
prefix=/sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-integration-20261026-r1/B/install
venv=/sdf/scratch/users/m/monarin/gpu-validation/kvikio-2610-validation-20261015-r1/venv
base=/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/xpp_drp_gpu_311
export PATH="$venv/bin:$base/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib:$base/lib"
export PYTHONPATH="$prefix/lib/python3.11/site-packages"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_PATH=/sdf/group/lcls/ds/tools/cuda12.8
export KVIKIO_COMPAT_MODE=ON KVIKIO_NTHREADS=8
export PS_PARALLEL=mpi PS_FAKESTEP_FLAG=0 PS_SMD_N_EVENTS=1000 PS_SMD_CHUNKSIZE=268435456
export PS_EB_NODES=1 PS_SRV_NODES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$@"
