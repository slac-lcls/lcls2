#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:data
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=33
#SBATCH --time=00:40:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_eb_node_local

# Iteration 18: test PS_EB_NODE_LOCAL (node.py's colocate_non_marching mode) as
# the structural lever for the multi-node plateau. iters 16/17 localized the
# 2x-node scaling loss (1.60-1.66x) to smd0/EB cross-node COORDINATION latency:
# eb_wait ~2x and the wait-residual ~2x while bd_read/dgram/smdparse stay flat.
# Default modulo placement (color = bd_main_rank % PS_EB_NODES) puts the single
# EB on node 0; ALL of node 1's BD ranks reach across the network to it every
# batch. PS_EB_NODE_LOCAL splits bd_main by COMM_TYPE_SHARED so each physical
# node gets its OWN EB serving only its local BD ranks -> the cross-node hop for
# node 1's BDs is removed. HYPOTHESIS: eb_wait + residual drop, aggregate rises.
#
# One variable: PS_EB_NODE_LOCAL off (default) vs on, at 2-node/64-BD, bracketed
# default/colocate/default so the window/cache regime is controlled. Colocate
# yields 63 BD ranks (2 EBs) vs default 64 BD (1 EB) -- a ~1.5% BD-count penalty
# that a real placement win must clear. bench_calib now reads the true role from
# mpi_ds.nodetype so the non-contiguous colocate EBs (world ranks 1 and 33) are
# correctly excluded from the benchmark body and the aggregate.

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

BENCH="python psana/psana/gpu/bench_calib.py -e mfx101572426 -r 47 -n 150 --warmup 10 --wait-split --dir ${FFB}"

# --- A: 2-node default modulo placement (single EB on N0), bracket start ---
T0=$(date +%s)
mpirun --bind-to none -x PS_EB_NODES -H ${N0}:33,${N1}:33 -n 66 ${BENCH} > ${OUTDIR}/enl_default_a.log 2>&1
RCA=$?; TA=$(date +%s)
echo "=== CONFIG MARKER: CFG=default_a exit=${RCA} wall=$((TA-T0)) ==="

# --- B: 2-node node-local EBs (one EB per node) ---
env PS_EB_NODE_LOCAL=1 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -H ${N0}:33,${N1}:33 -n 66 ${BENCH} > ${OUTDIR}/enl_colocate.log 2>&1
RCB=$?; TB=$(date +%s)
echo "=== CONFIG MARKER: CFG=colocate exit=${RCB} wall=$((TB-TA)) ==="

# --- C: 2-node default modulo placement, bracket end ---
mpirun --bind-to none -x PS_EB_NODES -H ${N0}:33,${N1}:33 -n 66 ${BENCH} > ${OUTDIR}/enl_default_c.log 2>&1
RCC=$?; TC=$(date +%s)
echo "=== CONFIG MARKER: CFG=default_c exit=${RCC} wall=$((TC-TB)) ==="

echo "=== DONE: default_a=${RCA} colocate=${RCB} default_c=${RCC} total_wall=$((TC-T0)) ==="
