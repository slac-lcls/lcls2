#!/bin/bash
set -euo pipefail
export MPICH_GPU_SUPPORT_ENABLED=0 OMP_NUM_THREADS=1 MPICH_OFI_CXI_COUNTER_REPORT=0
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_MEM_BIND
cd -- "$(dirname -- "$0")"
exec /pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python campaign.py run "$@"
