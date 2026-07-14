#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:default
#SBATCH --qos=preemptable
#SBATCH --nodes=4
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=33
#SBATCH --time=00:25:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_eb_knee_4n_rev

# Iteration 27: REVERSED-ORDER 4-node confirm to DISCHARGE the iter-26 3-vs-4
# confound. iter-26 (forward order 4->2->3->5, job 31592091) placed the predicted
# knee=4 LAST (815.8 Hz) and k=3 FIRST-of-its-window best (914.1 Hz), breaking the
# knee~=node_count law -- but k=4 ran COLD-first and k=3 ran warmer/third, so the
# +12% k=3 lead is warming-inflated and the exact 3-vs-4 boundary was confounded.
#
# This bracket REVERSES the phase order to 5->3->2->4 so that k=4 now runs
# WARMEST-LAST and k=3 runs COLDER/earlier -- the opposite warming bias from
# iter-26. The logic: warming can only LIFT a later phase, so
#   - if k=4 STILL loses to the colder-run k=3 even with the warming advantage,
#     then knee=3 at 4 nodes is ROBUST and the saturation-near-3 verdict is LOCKED;
#   - if warm-last k=4 now overtakes colder k=3, the knee is 4 after all and the
#     node_count law survives weakly. Either way the 3-vs-4 confound is discharged.
#
# One variable: EBs per node (PS_EB_PER_NODE in {5,3,2,4}) at fixed 4 nodes,
# node-local colocation. Same topology as iter-26 (132 ranks, rank0=smd0):
#   k=2 ->  8 EB, 123 topo BD
#   k=3 -> 12 EB, 119 topo BD
#   k=4 -> 16 EB, 115 topo BD
#   k=5 -> 20 EB, 111 topo BD
# --wait-split REQUIRED: verdict is eb_wait (falls with more EB) vs BD-slot loss.

set -x
cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh
export PS_EB_NODES=1

OUTDIR=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
N0=${NODES[0]}; N1=${NODES[1]}; N2=${NODES[2]}; N3=${NODES[3]}
echo "=== NODES: N0=${N0} N1=${N1} N2=${N2} N3=${N3} ==="

BENCH="python psana/psana/gpu/bench_calib.py -e mfx101572426 -r 47 -n 100 --warmup 10 --wait-split --dir ${FFB}"
HOSTS="${N0}:33,${N1}:33,${N2}:33,${N3}:33"

T0=$(date +%s)

# --- A: 5 EB per node (upper bound; COLD first phase this time) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=5 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${HOSTS} -n 132 ${BENCH} > ${OUTDIR}/epn4nkr_5eb.log 2>&1
RCA=$?; TA=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn4_5eb exit=${RCA} wall=$((TA-T0)) ==="

# --- B: 3 EB per node (the incumbent knee; runs colder/earlier than k=4 here) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=3 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${HOSTS} -n 132 ${BENCH} > ${OUTDIR}/epn4nkr_3eb.log 2>&1
RCB=$?; TB=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn4_3eb exit=${RCB} wall=$((TB-TA)) ==="

# --- C: 2 EB per node (lower bound) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=2 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${HOSTS} -n 132 ${BENCH} > ${OUTDIR}/epn4nkr_2eb.log 2>&1
RCC=$?; TC=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn4_2eb exit=${RCC} wall=$((TC-TB)) ==="

# --- D: 4 EB per node (predicted knee; WARMEST LAST phase -- best shot for k=4) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=4 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${HOSTS} -n 132 ${BENCH} > ${OUTDIR}/epn4nkr_4eb.log 2>&1
RCD=$?; TD=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn4_4eb exit=${RCD} wall=$((TD-TC)) ==="

echo "=== DONE: mn4_5eb=${RCA} mn4_3eb=${RCB} mn4_2eb=${RCC} mn4_4eb=${RCD} total_wall=$((TD-T0)) ==="
