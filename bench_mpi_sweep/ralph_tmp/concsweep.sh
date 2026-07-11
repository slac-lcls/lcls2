#!/bin/bash
# BD-ranks-per-node concurrency sweep, fixed 1 node / FFB / r47 GPU path.
# Decides latency-bound (agg GB/s climbs toward 7.9 with more ranks) vs
# per-rank serialization limit (plateaus well below 7.9).
# 64 BD = 66 procs <= 112 allocated CPUs -> real cores, no oversubscription.
set +u
cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh
set -u
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc
OUT=bench_mpi_sweep/ralph_tmp
STAMP=$1
NEV=100
# bracket: run 32 first and last to gauge FFB minute-to-minute variance
for BD in 8 16 32 48 64 32; do
  NPROC=$((BD+2))
  LOG=$OUT/conc_${BD}bd_${STAMP}_$(printf '%02d' $BD).log
  echo "### BD=$BD NPROC=$NPROC $(date +%H:%M:%S)" | tee -a $OUT/conc_summary_${STAMP}.log
  mpirun --bind-to none --oversubscribe -n $NPROC \
      python psana/psana/gpu/bench_calib.py \
      -e mfx101572426 -r 47 -n $NEV --warmup 10 --dir $FFB > $LOG 2>&1
  grep -E "aggregate rate|BD ranks|H->D|kernel" $LOG | tee -a $OUT/conc_summary_${STAMP}.log
  echo "" | tee -a $OUT/conc_summary_${STAMP}.log
done
echo "SWEEP_DONE $(date +%H:%M:%S)" | tee -a $OUT/conc_summary_${STAMP}.log
