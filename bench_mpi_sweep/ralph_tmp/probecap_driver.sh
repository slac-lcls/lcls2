#!/bin/bash
# iter 14: settle iter-13's over-read confound. iter 13's --reader-probe halved
# the 32-BD GPU-feed loop (-50%) but the reader ran ~1115 MB/s/rank = 1.75x a
# FAITHFUL prefetch load (33.5 MB/event x ~19 Hz/rank = ~640 MB/s/rank). Cap the
# reader at 640 MB/s/rank (PS_READER_PROBE_MBPS) and re-run the SAME palindrome
# bracket (seg / capped-probe / capped-probe / seg) to see whether a faithful-rate
# reader still contends materially. One variable = the rate cap.
cd /sdf/scratch/users/a/ajshack/lcls2
set +u; source setup_env.sh >/dev/null 2>&1; set -u
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc
TS=$(date +%H%M%S)
OUT=bench_mpi_sweep/ralph_tmp
export PS_READER_PROBE_MBPS=640

run() {  # $1=label $2=flags
  local label=$1 extra=$2
  local log=$OUT/probecap_${label}_${TS}.log
  echo "=== $(date +%H:%M:%S) $label (flags='$extra' cap=${PS_READER_PROBE_MBPS}) -> $log ==="
  mpirun --bind-to none --oversubscribe -n 34 -x PS_READER_PROBE_MBPS \
    python psana/psana/gpu/bench_calib.py -e mfx101572426 -r 47 \
    -n 200 --warmup 10 --dir $FFB $extra > $log 2>&1
  grep -E "aggregate rate|reader-probe:.*aggregate" $log | tail -3
}

run a_seg   "--seg-h2d"
run b_probe "--reader-probe"
run c_probe "--reader-probe"
run d_seg   "--seg-h2d"
echo "=== DONE $TS ==="
