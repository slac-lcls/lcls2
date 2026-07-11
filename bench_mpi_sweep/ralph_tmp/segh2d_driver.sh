#!/bin/bash
# iter 8 A/B: baseline (copy=False, one big H2D) vs --seg-h2d (per-seg H2D, no
# host stack). Interleaved back-to-back to control for FFB minute-to-minute
# variance (same method as iter 7's copy=False A/B).
cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh >/dev/null 2>&1  # no `set -u`: conda hooks reference unset vars
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc
TS=$(date +%H%M%S)
OUT=bench_mpi_sweep/ralph_tmp
run() {  # $1=label $2=nprocs $3=nevents $4=extra_flags
  local label=$1 np=$2 nev=$3 extra=$4
  local log=$OUT/segh2d_${label}_${TS}.log
  echo "=== $label (np=$np n=$nev flags='$extra') -> $log ==="
  mpirun --bind-to none --oversubscribe -n $np \
    python psana/psana/gpu/bench_calib.py -e mfx101572426 -r 47 \
    -n $nev --warmup 10 --dir $FFB $extra > $log 2>&1
  grep -E "aggregate rate|H->D|kernel|per-rank rate|rate:" $log | head -6
}

# 1 BD (3 procs): baseline / seg-h2d interleaved, 2 brackets
run 1bd_base_a 3 300 ""
run 1bd_seg_a  3 300 "--seg-h2d"
run 1bd_base_b 3 300 ""
run 1bd_seg_b  3 300 "--seg-h2d"

# 32 BD (34 procs): baseline / seg-h2d interleaved, 2 brackets
run 32bd_base_a 34 150 ""
run 32bd_seg_a  34 150 "--seg-h2d"
run 32bd_base_b 34 150 ""
run 32bd_seg_b  34 150 "--seg-h2d"

echo "=== DONE $TS ==="
