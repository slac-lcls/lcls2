#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:default
#SBATCH --qos=preemptable
#SBATCH --nodes=3
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=33
#SBATCH --time=00:25:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_eb_knee_3n

# Iteration 25: PIN the 3-node PS_EB_PER_NODE knee. iter-24 (both orders) found
# 3 EB/node wins at 3 nodes over {1,2} (+23.8% cold-first), so the 3-node knee
# is bounded only as >=3 -- k=4,5 were never tested. The emergent rule from the
# two measured knees (2 @ 2n, >=3 @ 3n) is knee ~= node_count; if k=3 is the
# TRUE 3-node optimum this rule holds, but if k=4 or k=5 keeps winning the knee
# grows FASTER than node_count and the rule is wrong.
#
# One variable: EBs per node (PS_EB_PER_NODE in {3,4,5}) at fixed 3 nodes,
# node-local colocation. Topology at 3 nodes, 99 ranks (rank0=smd0):
#   k=3 -> 9 EB, 89 BD (iter-24 hero)
#   k=4 -> 12 EB, 86 BD
#   k=5 -> 15 EB, 83 BD
#
# Order front-loads the hypothesized winner k=3 as the COLD first phase. Warming
# drifts FFB window state to favor LATER phases, so if k=3 wins even cold-first
# (over warmer k=4/k=5) the knee=3 verdict is conservative and strong. If a
# later higher-k wins, that could be warming and would need a reversed confirm.
# --wait-split REQUIRED: verdict is eb_wait (falls with more EB) vs BD-slot loss.

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

# --- A: 3 EB per node (iter-24 hero; COLD first phase / in-window anchor) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=3 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${N0}:33,${N1}:33,${N2}:33 -n 99 ${BENCH} > ${OUTDIR}/epn3nk_3eb.log 2>&1
RCA=$?; TA=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn3_3eb exit=${RCA} wall=$((TA-T0)) ==="

# --- B: 4 EB per node (does the knee keep climbing past 3?) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=4 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${N0}:33,${N1}:33,${N2}:33 -n 99 ${BENCH} > ${OUTDIR}/epn3nk_4eb.log 2>&1
RCB=$?; TB=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn3_4eb exit=${RCB} wall=$((TB-TA)) ==="

# --- C: 5 EB per node (upper bound of the knee sweep) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=5 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${N0}:33,${N1}:33,${N2}:33 -n 99 ${BENCH} > ${OUTDIR}/epn3nk_5eb.log 2>&1
RCC=$?; TC=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn3_5eb exit=${RCC} wall=$((TC-TB)) ==="

echo "=== DONE: mn3_3eb=${RCA} mn3_4eb=${RCB} mn3_5eb=${RCC} total_wall=$((TC-T0)) ==="
