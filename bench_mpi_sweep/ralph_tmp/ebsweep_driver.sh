#!/bin/bash
# iter 11: re-measure PS_EB_NODES=1/2/4 at fixed 32 BD on the seg-h2d fast path
# with --wait-split, to test whether eb_wait (51.5 ms/event @32BD, iter 10)
# shrinks with more EB ranks. Two interleaved sweeps bracket FFB variance.
# Layout: total procs = 1 smd0 + E EB + 32 BD.
cd /sdf/scratch/users/a/ajshack/lcls2
set +u; source setup_env.sh; set -u
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc
TS=$(date +%H%M%S)
D=bench_mpi_sweep/ralph_tmp

run_one () {
  local E=$1 pass=$2
  local NP=$((1 + E + 32))
  echo "=== PS_EB_NODES=$E  ($NP procs, 32 BD)  pass $pass ==="
  PS_EB_NODES=$E mpirun --bind-to none --oversubscribe -n $NP \
    python psana/psana/gpu/bench_calib.py --wait-split \
    -e mfx101572426 -r 47 -n 200 --warmup 10 --dir $FFB \
    > $D/ebsweep_eb${E}_p${pass}_${TS}.log 2>&1
  echo "  eb=$E pass=$pass done"
}

for pass in 1 2; do
  for E in 1 2 4; do
    run_one $E $pass
  done
done
echo "ALL DONE TS=$TS"
