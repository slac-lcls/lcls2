#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:data
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=33
#SBATCH --time=00:40:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_mn2x32_r47

# Iteration 16: re-measure the multi-node plateau on the r47 reference AFTER
# copy=False (+30%) + seg-h2d (+32%) landed. Historical scaling was poor:
# single-node 32-BD ~210 Hz -> 2-node 64-BD ~295 Hz = only 1.40x for 2x nodes.
# Both were measured BEFORE the two per-rank wins, on r387 (now purged).
#
# Design: run single-node (34 ranks, node0 only) and two-node (66 ranks)
# configs WITHIN THE SAME allocation, bracketed single/multi/single, so the
# window/cache regime (the dominant run-to-run confound, iters 13-14) is
# controlled and the scaling ratio is a clean same-window number.
# --wait-split (bit-exact additive instrumentation) reports eb_wait/bd_read/
# dgram so we also see whether a single EB serving 64 BD becomes the ceiling.

set -x
cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh
export PS_EB_NODES=1

OUTDIR=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
N0=${NODES[0]}
N1=${NODES[1]}
echo "=== NODES: N0=${N0} N1=${N1} ==="

BENCH="python psana/psana/gpu/bench_calib.py -e mfx101572426 -r 47 -n 200 --warmup 10 --wait-split --dir ${FFB}"

# --- A: single-node 32 BD (34 ranks on N0), bracket start ---
T0=$(date +%s)
mpirun --bind-to none -x PS_EB_NODES --oversubscribe -H ${N0}:34 -n 34 ${BENCH} > ${OUTDIR}/sweep_sn32_r47_a.log 2>&1
RCA=$?; TA=$(date +%s)
echo "=== CONFIG MARKER: CFG=sn32_a exit=${RCA} wall=$((TA-T0)) ==="

# --- B: two-node 64 BD (66 ranks, 33 per node) ---
mpirun --bind-to none -x PS_EB_NODES -H ${N0}:33,${N1}:33 -n 66 ${BENCH} > ${OUTDIR}/sweep_mn2x32_r47.log 2>&1
RCB=$?; TB=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn2x32 exit=${RCB} wall=$((TB-TA)) ==="

# --- C: single-node 32 BD (34 ranks on N0), bracket end ---
mpirun --bind-to none -x PS_EB_NODES --oversubscribe -H ${N0}:34 -n 34 ${BENCH} > ${OUTDIR}/sweep_sn32_r47_c.log 2>&1
RCC=$?; TC=$(date +%s)
echo "=== CONFIG MARKER: CFG=sn32_c exit=${RCC} wall=$((TC-TB)) ==="

echo "=== DONE: sn32_a=${RCA} mn2x32=${RCB} sn32_c=${RCC} total_wall=$((TC-T0)) ==="
