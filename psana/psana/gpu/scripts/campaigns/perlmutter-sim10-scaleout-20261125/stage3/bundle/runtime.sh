#!/bin/bash
set -eo pipefail
attempt=$1
shift
source /opt/nersc/pe/etc/lmod/lmod_setup.sh
source /etc/profile.d/zz-cray-pe.sh
module load gcc-native/13.2 cudatoolkit/12.9
source /global/common/software/nersc/pe/conda/25.2.0/Miniforge3-24.11.3-0/etc/profile.d/conda.sh
conda activate /pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env
unset PYTHONPATH LD_PRELOAD SIT_PSDM_OFFSITE
case "${SIM10_RUNTIME:?Select P or C}" in
 P) sim10_prefix=/pscratch/sd/m/monarin/agent-work/lcls2/gpu-master-sync-20261119-attempt02/baseline/install; sim10_source=/pscratch/sd/m/monarin/agent-work/lcls2/gpu-master-sync-20261119-attempt02/baseline/source ;;
 C) sim10_prefix=/pscratch/sd/m/monarin/agent-work/lcls2/gpu-master-sync-20261119-attempt02/candidate/install; sim10_source=/pscratch/sd/m/monarin/agent-work/lcls2/gpu-master-sync-20261119-attempt02/candidate/source ;;
 *) exit 2 ;;
esac
source "$sim10_prefix/activate.sh"
export LD_LIBRARY_PATH="/opt/cray/libfabric/2.3.1/lib64:${LD_LIBRARY_PATH:-}"
export MPICH_GPU_SUPPORT_ENABLED=0 LCLS_CALIB_HTTP=https://pswww.slac.stanford.edu/calib_ws/
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="$attempt/tmp" CUPY_CACHE_DIR="$attempt/cupy-cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export KVIKIO_COMPAT_MODE=ON KVIKIO_NTHREADS=4 KVIKIO_TASK_SIZE=1048576
export PS_PARALLEL=mpi PS_FAKESTEP_FLAG=0 PS_EB_NODES=1 PS_SRV_NODES=0 PS_SMD_N_EVENTS=1000 PS_SMD_CHUNKSIZE=268435456
export KVIKIO_BOUNCE_BUFFER_SIZE=16777216 PS_JF_SHARE_DERIVED=0 PS_GEO_SHARE=0
[[ "$SIM10_RUNTIME" == C ]] || exit 2
export SIM10_INPUTS="$attempt/inputs.json"
export SIM10_PR_SOURCE="$sim10_source"
export SIM10_NATIVE_RECEIPT="$attempt/bundle/dependency-pins.json"
exec "$@"
