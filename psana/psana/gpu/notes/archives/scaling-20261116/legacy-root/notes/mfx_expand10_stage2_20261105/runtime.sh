#!/usr/bin/env bash
set -euo pipefail
stage_prefix=/sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-integration-20261026-r1/B/install
stage_env=/sdf/scratch/users/m/monarin/gpu-validation/kvikio-2610-validation-20261015-r1/venv
stage_base=/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/xpp_drp_gpu_311
export PATH="$stage_env/bin:$stage_base/bin:$PATH"
export LD_LIBRARY_PATH="$stage_prefix/lib:$stage_base/lib"
export PYTHONPATH="$stage_prefix/lib/python3.11/site-packages"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_PATH=/sdf/group/lcls/ds/tools/cuda12.8
export KVIKIO_COMPAT_MODE=ON
export KVIKIO_NTHREADS=4
export PS_PARALLEL=none
export PS_FAKESTEP_FLAG=0
export PS_SMD_N_EVENTS=20
export PS_SMD_CHUNKSIZE=268435456
export PS_SRV_NODES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
exec "$stage_env/bin/python" -u "$@"
