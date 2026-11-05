#!/bin/bash
set -eo pipefail
campaign_root=$1
variant=$2
shift 2
source /opt/nersc/pe/etc/lmod/lmod_setup.sh
source /etc/profile.d/zz-cray-pe.sh
module load conda/Miniforge3-24.11.3-0 gcc-native/13.2 cudatoolkit/12.9
source /global/common/software/nersc/pe/conda/25.2.0/Miniforge3-24.11.3-0/etc/profile.d/conda.sh
conda activate /pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env
unset PYTHONPATH LD_PRELOAD SIT_PSDM_OFFSITE
export CC="$(command -v gcc)" CXX="$(command -v g++)"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export MPICH_GPU_SUPPORT_ENABLED=0 LCLS_CALIB_HTTP=https://pswww.slac.stanford.edu/ws
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="$campaign_root/tmp" CUPY_CACHE_DIR="$campaign_root/cupy-cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export KVIKIO_COMPAT_MODE=ON KVIKIO_NTHREADS=8 KVIKIO_TASK_SIZE=1048576
export PS_PARALLEL=mpi PS_EB_NODES=1 PS_SRV_NODES=0 PS_SMD_N_EVENTS=1000
export CAMPAIGN_ROOT="$campaign_root" CAMPAIGN_VARIANT="$variant"
if [[ "$variant" != none ]]; then
    source "$campaign_root/install-$variant/activate.sh"
    export EXPECTED_INSTALL="$campaign_root/install-$variant"
fi
export BENCH_INSTALL="${EXPECTED_INSTALL:-}"
export PSANA_GPU_SHARE_CONSTANTS=1 BENCH_ALIGNMENT_SHIFT=0
if [[ "$variant" != none ]]; then
 export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$campaign_root/scripts:$campaign_root/harness/jf_scaling:$campaign_root/harness/feespec_bulk_benchmark:$campaign_root/harness/stage1_regression"
fi
exec "$@"
