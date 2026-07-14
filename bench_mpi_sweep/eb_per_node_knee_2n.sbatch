#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:default
#SBATCH --qos=preemptable
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=33
#SBATCH --time=00:20:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_eb_per_node_knee_2n

# Iteration 23: KNEE test for the intra-node EB fan-out lever (PS_EB_PER_NODE).
# iter-22 measured PS_EB_PER_NODE=2 beats 1 at 2 nodes (+23% aggregate, eb_wait
# -18%). The graduation-decision question is whether 2 is the sweet spot or 3/4
# keeps helping. One variable: EBs per node at fixed 2 nodes, node-local
# colocation. More EBs cost BD slots (BD = 65 - 2k for k EB/node at 2x33), so
# each EB serving fewer BDs must lower eb_wait enough to beat the lost workers.
#
# Topology at 2 nodes, 66 ranks (rank0=smd0):
#   k=2 -> 61 BD (iter-22 anchor: 504.9 Hz, eb_wait 17.4 ms)
#   k=3 -> 59 BD
#   k=4 -> 57 BD
#
# Order front-loads the core knee comparison: 3eb (new science) then 2eb (the
# in-window anchor -- REQUIRED, window state varies) then 4eb. If preempted after
# two phases, the 2-vs-3 knee question is still answered. --wait-split REQUIRED:
# the verdict is eb_wait vs BD-slot tradeoff, not aggregate alone.

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

T0=$(date +%s)

# --- A: 2-node, 3 EB per node (the new knee point) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=3 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${N0}:33,${N1}:33 -n 66 ${BENCH} > ${OUTDIR}/epnknee_3eb.log 2>&1
RCA=$?; TA=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn2_3eb exit=${RCA} wall=$((TA-T0)) ==="

# --- B: 2-node, 2 EB per node (in-window anchor vs iter-22) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=2 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${N0}:33,${N1}:33 -n 66 ${BENCH} > ${OUTDIR}/epnknee_2eb.log 2>&1
RCB=$?; TB=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn2_2eb exit=${RCB} wall=$((TB-TA)) ==="

# --- C: 2-node, 4 EB per node (does it keep falling past the knee?) ---
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=4 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${N0}:33,${N1}:33 -n 66 ${BENCH} > ${OUTDIR}/epnknee_4eb.log 2>&1
RCC=$?; TC=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn2_4eb exit=${RCC} wall=$((TC-TB)) ==="

echo "=== DONE: mn2_3eb=${RCA} mn2_2eb=${RCB} mn2_4eb=${RCC} total_wall=$((TC-T0)) ==="
