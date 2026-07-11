#!/bin/bash
# Iter 7: interleaved copy=True vs copy=False before/after, bracketed for FFB
# minute-to-minute variance. Same binary, single toggled variable (--copy-false).
set -u
cd /sdf/scratch/users/a/ajshack/lcls2
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc
JOB=$(squeue -u ajshack -n ralph-gpu -h -t RUNNING -o %A)
TS=$1
OUT=bench_mpi_sweep/ralph_tmp

run() {  # label nprocs nevents flags
  local label=$1 nprocs=$2 nev=$3 flags=$4
  local log=$OUT/cf_${label}_${TS}.log
  echo "=== $label (nprocs=$nprocs, n=$nev, flags='$flags') @ $(date +%H:%M:%S) ==="
  srun --jobid=$JOB --overlap -n1 bash -c "
    set +u; source /sdf/scratch/users/a/ajshack/lcls2/setup_env.sh >/dev/null 2>&1; set -u
    mpirun --bind-to none --oversubscribe -n $nprocs \
      python psana/psana/gpu/bench_calib.py -e mfx101572426 -r 47 \
      -n $nev --warmup 10 --dir $FFB $flags" >"$log" 2>&1
  local hz=$(grep 'aggregate rate' "$log" | tail -1 | awk '{print $3}')
  echo "    -> aggregate ${hz} Hz  ($log)"
}

echo "#### 1 BD (3 procs), interleaved x2 ####"
run 1bd_copyTrue_a  3 500 ""
run 1bd_copyFalse_a 3 500 "--copy-false"
run 1bd_copyTrue_b  3 500 ""
run 1bd_copyFalse_b 3 500 "--copy-false"

echo "#### 32 BD (34 procs), interleaved x2 ####"
run 32bd_copyTrue_a  34 200 ""
run 32bd_copyFalse_a 34 200 "--copy-false"
run 32bd_copyTrue_b  34 200 ""
run 32bd_copyFalse_b 34 200 "--copy-false"

echo "#### DONE @ $(date +%H:%M:%S) ####"
