#!/bin/bash
# --profile-read decomposition of det.raw.raw at 1 and 32 BD, r47/FFB, GPU path.
# Settles storage-I/O vs in-process CPU for the 119 ms/event read bucket (iter 3).
set +u
cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh
set -u
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc
OUT=bench_mpi_sweep/ralph_tmp
STAMP=$1
for BD in 1 32; do
  NPROC=$((BD+2))
  NEV=$([ $BD -eq 1 ] && echo 300 || echo 150)
  LOG=$OUT/profread_${BD}bd_${STAMP}.log
  echo "### BD=$BD NPROC=$NPROC $(date +%H:%M:%S)"
  mpirun --bind-to none --oversubscribe -n $NPROC \
      python psana/psana/gpu/bench_calib.py \
      -e mfx101572426 -r 47 -n $NEV --warmup 10 --profile-read --dir $FFB > $LOG 2>&1
  echo "--- BD=$BD result ---"
  grep -E "rate|wait|read1|seg:|stack|copy|read2|H->D|kernel|per-rank|aggregate|BD ranks" $LOG
  echo ""
done
echo "PROFREAD_DONE $(date +%H:%M:%S)"
