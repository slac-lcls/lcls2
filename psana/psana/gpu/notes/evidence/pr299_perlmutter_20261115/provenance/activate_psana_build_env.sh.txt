#!/usr/bin/env bash

env_prefix="${1:-$HOME/.conda-envs/psana-build}"

module load conda/Miniforge3-24.11.3-0 gcc-native/13.2 >/dev/null 2>&1
source /global/common/software/nersc/pe/conda/25.2.0/Miniforge3-24.11.3-0/etc/profile.d/conda.sh
conda activate "$env_prefix"

# Keep CMake pointed at headers and libraries installed in this env.
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export MPICH_GPU_SUPPORT_ENABLED=0
export LCLS_CALIB_HTTP="https://pswww.slac.stanford.edu/calib_ws/"

cat <<EOF
Activated psana build environment:
  CONDA_PREFIX=${CONDA_PREFIX}
  CC=${CC}
  CXX=${CXX}
  MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED}
  LCLS_CALIB_HTTP=${LCLS_CALIB_HTTP}

Hint:
  XTC data on NERSC are typically under:
  /pscratch/sd/p/psdatmgr/psdm/<instr>/<expt>/...
EOF
