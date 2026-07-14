#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:default
#SBATCH --qos=preemptable
#SBATCH --nodes=4
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=33
#SBATCH --time=00:25:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_eb_knee_4n

# Iteration 26: TEST the emergent law knee ~= node_count at the 4-node point.
# Two measured knees so far: 2 EB/node @ 2 nodes (iter-23), 3 EB/node @ 3 nodes
# (iter-24/25, exactly 3 -- k=4,5 lost). The two-point law knee~=node_count
# predicts k=4 wins at 4 nodes. This bracket is the third point that turns the
# line into a law -- or breaks it.
#
# One variable: EBs per node (PS_EB_PER_NODE in {2,3,4,5}) at fixed 4 nodes,
# node-local colocation. Topology at 4 nodes, 132 ranks (rank0=smd0):
#   k=2 ->  8 EB, 123 BD   (below predicted knee)
#   k=3 -> 12 EB, 119 BD   (the 3-node knee; below if rule holds)
#   k=4 -> 16 EB, 115 BD   (predicted 4-node knee / hero)
#   k=5 -> 20 EB, 111 BD   (above predicted knee)
#
# Order front-loads the hypothesized winner k=4 as the COLD first phase, then
# runs its two nearest threats k=3 (rule says knee stays 3) and k=5 (rule says
# knee climbs faster) LATER so they get the FFB-window warming advantage. If k=4
# wins even cold-first over warmer k=3/k=5, the knee=4 verdict -- and the
# node_count law -- is conservative and strong. k=2 (lower bound) runs second.
# Same cold-first design as the iter-25 3-node knee bracket. --wait-split
# REQUIRED: verdict is eb_wait (falls with more EB) vs BD-slot loss past knee.

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

# --- A: 4 EB per node (predicted knee / hero; COLD first phase, in-window anchor) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=4 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${HOSTS} -n 132 ${BENCH} > ${OUTDIR}/epn4nk_4eb.log 2>&1
RCA=$?; TA=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn4_4eb exit=${RCA} wall=$((TA-T0)) ==="

# --- B: 2 EB per node (lower bound below the predicted knee) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=2 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${HOSTS} -n 132 ${BENCH} > ${OUTDIR}/epn4nk_2eb.log 2>&1
RCB=$?; TB=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn4_2eb exit=${RCB} wall=$((TB-TA)) ==="

# --- C: 3 EB per node (the 3-node knee; does it stay at 3 or climb?) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=3 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${HOSTS} -n 132 ${BENCH} > ${OUTDIR}/epn4nk_3eb.log 2>&1
RCC=$?; TC=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn4_3eb exit=${RCC} wall=$((TC-TB)) ==="

# --- D: 5 EB per node (upper bound; warmest last phase) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=5 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${HOSTS} -n 132 ${BENCH} > ${OUTDIR}/epn4nk_5eb.log 2>&1
RCD=$?; TD=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn4_5eb exit=${RCD} wall=$((TD-TC)) ==="

echo "=== DONE: mn4_4eb=${RCA} mn4_2eb=${RCB} mn4_3eb=${RCC} mn4_5eb=${RCD} total_wall=$((TD-T0)) ==="
