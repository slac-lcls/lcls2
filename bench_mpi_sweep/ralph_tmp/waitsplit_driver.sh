#!/bin/bash
cd /sdf/scratch/users/a/ajshack/lcls2
set +u; source setup_env.sh; set -u
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc
TS=$(date +%H%M%S)
D=bench_mpi_sweep/ralph_tmp

echo "=== 1 BD wait-split (3 procs) ==="
mpirun --bind-to none --oversubscribe -n 3 \
  python psana/psana/gpu/bench_calib.py --wait-split \
  -e mfx101572426 -r 47 -n 100 --warmup 10 --dir $FFB \
  > $D/waitsplit_1bd_${TS}.log 2>&1
echo "1BD done"

echo "=== 32 BD wait-split (34 procs) ==="
mpirun --bind-to none --oversubscribe -n 34 \
  python psana/psana/gpu/bench_calib.py --wait-split \
  -e mfx101572426 -r 47 -n 200 --warmup 10 --dir $FFB \
  > $D/waitsplit_32bd_${TS}.log 2>&1
echo "32BD done TS=$TS"
