#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-integration-20261026-r1/B/install/lib/python3.11/site-packages
export LD_LIBRARY_PATH=/sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-integration-20261026-r1/B/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PS_FAKESTEP_FLAG=0
export PS_PARALLEL=none
exec /sdf/group/lcls/ds/ana/sw/conda2/inst/envs/xpp_drp_gpu_311/bin/python "$@"
