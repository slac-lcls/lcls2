#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=/pscratch/sd/m/monarin/agent-work/lcls2/pr331-endrun-combined-20261124-attempt01
[[ $(hostname) == nid* ]]
trap 'rc=$?; echo "$rc" > "$stage/allocation-exit-code.txt"' EXIT
scontrol show job -d "$SLURM_JOB_ID" > "$stage/allocation.txt"
module load gcc-native/13.2 cudatoolkit/12.9
unset PYTHONPATH PYTHONHOME LD_PRELOAD SIT_PSDM_OFFSITE
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPICH_GPU_SUPPORT_ENABLED=0
export LCLS_CALIB_HTTP=https://pswww.slac.stanford.edu/calib_ws/
export TMPDIR="$stage/tmp"
cd "$stage"
/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python -B "$stage/service.py" > "$stage/workload.log" 2>&1
