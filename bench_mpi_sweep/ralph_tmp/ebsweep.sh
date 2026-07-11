#!/bin/bash
# iter 11: re-measure PS_EB_NODES=1/2/4 at fixed 32 BD on the seg-h2d fast path
# with --wait-split, to test whether the 51.5 ms eb_wait (iter 10, 31% of wait)
# shrinks with more EB ranks. Palindrome bracket (1,2,4,4,2,1) to separate the
# EB-count effect from FFB minute-to-minute window drift.
set +u
cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh
set -u
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc
TS=$1
OUT=bench_mpi_sweep/ralph_tmp
# 32 BD fixed; NPROC = 1 (smd0) + PS_EB_NODES + 32
for TAG in a_eb1 b_eb2 c_eb4 d_eb4 e_eb2 f_eb1; do
    EB=${TAG##*eb}
    NPROC=$((1 + EB + 32))
    LOG=$OUT/ebsweep_${TAG}_${TS}.log
    echo "=== $(date +%H:%M:%S) run $TAG: PS_EB_NODES=$EB NPROC=$NPROC ===" | tee -a $OUT/ebsweep_driver_${TS}.log
    PS_EB_NODES=$EB mpirun --bind-to none --oversubscribe -x PS_EB_NODES -n $NPROC \
        python psana/psana/gpu/bench_calib.py \
        -e mfx101572426 -r 47 -n 200 --warmup 10 --dir $FFB --seg-h2d --wait-split \
        > $LOG 2>&1
    AGG=$(grep -A2 "\[aggregate\]" $LOG | grep "aggregate rate" | head -1)
    EBW=$(grep "eb_wait:" $LOG | tail -1)
    BDR=$(grep "bd_read:" $LOG | tail -1)
    echo "  $TAG -> $AGG | $EBW | $BDR" | tee -a $OUT/ebsweep_driver_${TS}.log
done
echo "=== $(date +%H:%M:%S) DONE ===" | tee -a $OUT/ebsweep_driver_${TS}.log
