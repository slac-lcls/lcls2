#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:default
#SBATCH --qos=preemptable
#SBATCH --nodes=3
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=33
#SBATCH --time=00:25:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_eb_per_node_3n

# Iteration 24: does the PS_EB_PER_NODE=2 knee COMPOUND at 3 nodes, and is 2
# still the knee there? iter-22/23 measured 2 EB/node beats 1 by +23% aggregate
# at 2 nodes (eb_wait -18%), with 3/4 EB regressing (the 2-node knee is 2).
# node-local placement (iter-20) widened its lead +9%@2n -> +31%@3n, so the
# open question is whether the intra-node EB fan-out lever also widens with node
# count, and whether more cross-node EB coordination at 3 nodes shifts the knee.
#
# One variable: EBs per node (PS_EB_PER_NODE in {1,2,3}) at fixed 3 nodes,
# node-local colocation. Topology at 3 nodes, 99 ranks (rank0=smd0), 3k EBs:
#   k=1 -> 95 BD (node-local one-EB-per-node, the 2-node lead was +23% over this)
#   k=2 -> 92 BD (the 2-node knee / hero)
#   k=3 -> 89 BD (did the knee shift up at 3 nodes?)
#
# Order front-loads the COMPOUNDING verdict: 2eb (hero/in-window anchor -- window
# state varies, so it MUST be measured in the same allocation) then 1eb (the
# node-local baseline the 2-node +23% was over). If preempted after two phases,
# the compounding question (does 2eb widen its lead over 1eb from 2n to 3n) is
# still answered. 3eb (knee-still-2 check) runs last. --wait-split REQUIRED: the
# verdict is eb_wait vs BD-slot tradeoff, not aggregate alone.

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

# --- A: 3-node, 2 EB per node (the 2-node knee/hero; in-window anchor) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=2 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${N0}:33,${N1}:33,${N2}:33 -n 99 ${BENCH} > ${OUTDIR}/epn3n_2eb.log 2>&1
RCA=$?; TA=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn3_2eb exit=${RCA} wall=$((TA-T0)) ==="

# --- B: 3-node, 1 EB per node (node-local baseline; compounding comparison) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=1 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${N0}:33,${N1}:33,${N2}:33 -n 99 ${BENCH} > ${OUTDIR}/epn3n_1eb.log 2>&1
RCB=$?; TB=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn3_1eb exit=${RCB} wall=$((TB-TA)) ==="

# --- C: 3-node, 3 EB per node (did the knee shift up at 3 nodes?) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=3 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${N0}:33,${N1}:33,${N2}:33 -n 99 ${BENCH} > ${OUTDIR}/epn3n_3eb.log 2>&1
RCC=$?; TC=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn3_3eb exit=${RCC} wall=$((TC-TB)) ==="

echo "=== DONE: mn3_2eb=${RCA} mn3_1eb=${RCB} mn3_3eb=${RCC} total_wall=$((TC-T0)) ==="
