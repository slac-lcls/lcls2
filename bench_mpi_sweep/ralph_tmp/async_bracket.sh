#!/bin/bash
# iter 12: A/B async-prefetch vs seg-h2d at 32 BD on FFB, palindrome-bracketed
# to control the FFB minute-to-minute window. Aggregate Hz is the headline.
source setup_env.sh >/dev/null 2>&1
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc
STAMP=$(date +%H%M%S)
OUT=bench_mpi_sweep/ralph_tmp
N=200; W=10; NPROC=34   # 1 smd0 + 1 EB + 32 BD

run() {  # $1 label  $2 flag
  local tag=$1 flag=$2
  local log=$OUT/async_${tag}_${STAMP}.log
  echo "=== $tag ($flag) -> $log ==="
  mpirun --bind-to none --oversubscribe -n $NPROC \
     python psana/psana/gpu/bench_calib.py \
     -e mfx101572426 -r 47 -n $N --warmup $W $flag --dir $FFB \
     > "$log" 2>&1
  grep -E "aggregate rate|per-rank rate|BD ranks:" "$log" | sed 's/^/    /'
}

# palindrome: seg, async, async, seg  -> window drift symmetric, effect separable
run a_seg   "--seg-h2d"
run b_async "--async-prefetch"
run c_async "--async-prefetch"
run d_seg   "--seg-h2d"
echo "STAMP=$STAMP"
