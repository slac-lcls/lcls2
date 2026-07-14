#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:default
#SBATCH --qos=preemptable
#SBATCH --nodes=3
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=33
#SBATCH --time=00:25:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_eb_per_node_3n_rev

# Iteration 24 confirm: REVERSED-ORDER control for the 3-node PS_EB_PER_NODE knee.
# The forward bracket (job 31588126) found 3 EB/node wins at 3 nodes (553.3 Hz vs
# 411.5/419.5 for 2/1 EB) -- a reversal of the 2-node knee (=2). BUT 3eb was the
# LAST of three sequential phases, so an FFB-window-warming artifact cannot be
# ruled out. This job runs the SAME three configs in REVERSED order -- 3eb FIRST
# (coldest window), then 2eb, then 1eb. If 3eb still wins when it is the cold
# first phase, the knee-shift-to-3 is real, not warming. If 3eb's lead collapses
# cold, the reversal was an artifact and =2 holds.
#
# One variable: EBs per node in {3,2,1} at fixed 3 nodes, node-local colocation.
# Topology at 3 nodes, 99 ranks (rank0=smd0), 3k EBs:
#   k=3 -> 89 BD   k=2 -> ~91-92 BD   k=1 -> 95 BD
# --wait-split REQUIRED (verdict is eb_wait vs BD-slot tradeoff, not aggregate).

set -x
cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh
export PS_EB_NODES=1

OUTDIR=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
N0=${NODES[0]}; N1=${NODES[1]}; N2=${NODES[2]}
echo "=== NODES: N0=${N0} N1=${N1} N2=${N2} ==="

BENCH="python psana/psana/gpu/bench_calib.py -e mfx101572426 -r 47 -n 100 --warmup 10 --wait-split --dir ${FFB}"

T0=$(date +%s)

# --- A: 3-node, 3 EB per node (forward-bracket winner, now the COLD first phase) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=3 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${N0}:33,${N1}:33,${N2}:33 -n 99 ${BENCH} > ${OUTDIR}/epn3nrev_3eb.log 2>&1
RCA=$?; TA=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn3_3eb exit=${RCA} wall=$((TA-T0)) ==="

# --- B: 3-node, 2 EB per node (the 2-node knee) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=2 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${N0}:33,${N1}:33,${N2}:33 -n 99 ${BENCH} > ${OUTDIR}/epn3nrev_2eb.log 2>&1
RCB=$?; TB=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn3_2eb exit=${RCB} wall=$((TB-TA)) ==="

# --- C: 3-node, 1 EB per node (node-local baseline) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=1 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${N0}:33,${N1}:33,${N2}:33 -n 99 ${BENCH} > ${OUTDIR}/epn3nrev_1eb.log 2>&1
RCC=$?; TC=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn3_1eb exit=${RCC} wall=$((TC-TB)) ==="

echo "=== DONE: mn3_3eb=${RCA} mn3_2eb=${RCB} mn3_1eb=${RCC} total_wall=$((TC-T0)) ==="
