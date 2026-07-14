#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:default
#SBATCH --qos=preemptable
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=33
#SBATCH --time=00:20:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_eb_per_node_2n

# Iteration 22: 2-node bracket for the intra-node EB fan-out lever
# (PS_EB_PER_NODE). One variable: number of EBs per node under node-local
# colocation. iter-20 measured colocate (1 EB/node) at 3 nodes still leaves
# eb_wait 26 ms / 77% per-node efficiency; the open question (iter-17 vs 20) is
# whether that 26 ms is intra-node BDs-per-EB count (fixable -> 2 EB/node lowers
# eb_wait) or cross-node smd0/EB coordination (not fixable by more EBs/node).
#
# 2 nodes is the least preemption-fragile multi-node scale (iter-21). Front-load:
# sn_a control (quick, single node) -> mn2_2eb (the new science) -> mn2_1eb, so
# the two multi-node phases land inside the ~6 min preemption window (iter-20).
# --wait-split is REQUIRED here: the verdict is eb_wait, not just aggregate.
#
# Read: if mn2_2eb eb_wait < mn2_1eb eb_wait AND aggregate/per-rank rise (a 2nd
# EB costs a BD slot, so per-rank eb_wait must fall enough to beat the lost BD),
# the intra-node lever is confirmed -> graduate PS_EB_PER_NODE. If eb_wait is
# FLAT, the 26 ms is cross-node coordination -> closes the code-side multi-node
# levers (loop near LOOP DONE pending storage-side facility levers).

set -x
cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh
export PS_EB_NODES=1

OUTDIR=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
N0=${NODES[0]}; N1=${NODES[1]}
echo "=== NODES: N0=${N0} N1=${N1} ==="

BENCH="python psana/psana/gpu/bench_calib.py -e mfx101572426 -r 47 -n 100 --warmup 10 --wait-split --dir ${FFB}"

# --- A: single-node control (32 BD), in-window per-node baseline ---
T0=$(date +%s)
mpirun --bind-to none --oversubscribe -x PS_EB_NODES -H ${N0}:34 -n 34 ${BENCH} > ${OUTDIR}/epn2n_sn_a.log 2>&1
RCA=$?; TA=$(date +%s)
echo "=== CONFIG MARKER: CFG=sn_a exit=${RCA} wall=$((TA-T0)) ==="

# --- B: 2-node, 2 EB per node (PS_EB_PER_NODE=2) -- new science, run early ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=2 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${N0}:33,${N1}:33 -n 66 ${BENCH} > ${OUTDIR}/epn2n_2eb.log 2>&1
RCB=$?; TB=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn2_2eb exit=${RCB} wall=$((TB-TA)) ==="

# --- C: 2-node, 1 EB per node (node-local baseline, PS_EB_PER_NODE unset=1) ---
env PS_EB_NODE_LOCAL=1 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -H ${N0}:33,${N1}:33 -n 66 ${BENCH} > ${OUTDIR}/epn2n_1eb.log 2>&1
RCC=$?; TC=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn2_1eb exit=${RCC} wall=$((TC-TB)) ==="

echo "=== DONE: sn_a=${RCA} mn2_2eb=${RCB} mn2_1eb=${RCC} total_wall=$((TC-T0)) ==="
