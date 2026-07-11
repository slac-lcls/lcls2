#!/bin/bash
# iter 13: concurrency-headroom probe for read-side prefetch.
# Palindrome bracket at 32 BD to control FFB window drift:
#   baseline(seg-h2d) , probe , probe , baseline
# baseline = seg-h2d fast path, no background reader
# probe    = seg-h2d + one background os.pread thread per BD rank
set +u
cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh
set -u
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc
TS=$(date +%H%M%S)
OUT=bench_mpi_sweep/ralph_tmp
N=200; W=10; NPROC=34   # 1 smd0 + 1 EB + 32 BD

run() {  # $1 label  $2 flag
  local tag=$1 flag=$2
  local log=$OUT/probe_${tag}_${TS}.log
  echo "=== $(date +%H:%M:%S) $tag ($flag) -> $log ===" | tee -a $OUT/probe_driver_${TS}.log
  mpirun --bind-to none --oversubscribe -n $NPROC \
     python psana/psana/gpu/bench_calib.py \
     -e mfx101572426 -r 47 -n $N --warmup $W $flag --dir $FFB \
     > "$log" 2>&1
  grep -E "aggregate rate|reader-probe" "$log" | sed 's/^/    /' | tee -a $OUT/probe_driver_${TS}.log
}

# palindrome: seg, probe, probe, seg  -> window drift symmetric, effect separable
run a_seg   "--seg-h2d"
run b_probe "--reader-probe"
run c_probe "--reader-probe"
run d_seg   "--seg-h2d"
echo "=== $(date +%H:%M:%S) DONE TS=$TS ===" | tee -a $OUT/probe_driver_${TS}.log
