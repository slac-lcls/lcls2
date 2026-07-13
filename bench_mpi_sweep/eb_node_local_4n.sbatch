#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:data
#SBATCH --nodes=4
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=33
#SBATCH --time=00:50:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_eb_node_local_4n

# Iteration 19: does node-local EB placement (PS_EB_NODE_LOCAL) COMPOUND at 4
# nodes? iter-18 confirmed it at 2 nodes: colocate 328.4 Hz vs 301.0 default
# mean (+9.1% aggregate, +10.7% per-rank), residual 76.2->65.2 ms (-14%),
# bd_read 125.4->111.1 (-11%), implying ~1.81x node scaling (~90% per-node eff)
# vs default 1.66x. iter-18 could only pin scaling against a CROSS-WINDOW
# single-node anchor (181.7 Hz from iter-16); this run adds an IN-WINDOW
# single-node control so the 4-node per-node efficiency is measured against the
# same cache/window regime, not a stale anchor.
#
# One variable: PS_EB_NODE_LOCAL off (default modulo, single EB on N0) vs on
# (one EB per node) at 4-node/130-BD. Bracketed by single-node/32-BD controls
# (sn_a ... sn_c) so both the window regime AND the per-node baseline are pinned
# in the same allocation. HYPOTHESIS: if the cross-node-serve removal compounds,
# 4-node colocate keeps ~90% per-node efficiency (4x anchor) while 4-node
# default degrades further below its 2-node 83%; colocate residual/bd_read stay
# depressed vs default. If colocate's per-node efficiency has fallen back toward
# default, the win does NOT compound and the residual lever is intra-EB
# serialization (one EB rank/node saturating), not the cross-node hop.
#
# Colocate yields 127 BD (4 EBs) vs default 130 BD (1 EB) -- a ~2.3% BD-count
# penalty a real placement win must clear. bench_calib reads the true role from
# mpi_ds.nodetype so the 4 non-contiguous colocate EBs are correctly excluded.

set -x
cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh
export PS_EB_NODES=1

OUTDIR=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
N0=${NODES[0]}; N1=${NODES[1]}; N2=${NODES[2]}; N3=${NODES[3]}
echo "=== NODES: N0=${N0} N1=${N1} N2=${N2} N3=${N3} ==="

BENCH="python psana/psana/gpu/bench_calib.py -e mfx101572426 -r 47 -n 150 --warmup 10 --wait-split --dir ${FFB}"

# --- A: single-node control (32 BD), bracket start ---
T0=$(date +%s)
mpirun --bind-to none -x PS_EB_NODES -H ${N0}:34 -n 34 ${BENCH} > ${OUTDIR}/enl4_sn_a.log 2>&1
RCA=$?; TA=$(date +%s)
echo "=== CONFIG MARKER: CFG=sn_a exit=${RCA} wall=$((TA-T0)) ==="

# --- B: 4-node default modulo placement (single EB on N0) ---
mpirun --bind-to none -x PS_EB_NODES -H ${N0}:33,${N1}:33,${N2}:33,${N3}:33 -n 132 ${BENCH} > ${OUTDIR}/enl4_default.log 2>&1
RCB=$?; TB=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn4_default exit=${RCB} wall=$((TB-TA)) ==="

# --- C: 4-node node-local EBs (one EB per node) ---
env PS_EB_NODE_LOCAL=1 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -H ${N0}:33,${N1}:33,${N2}:33,${N3}:33 -n 132 ${BENCH} > ${OUTDIR}/enl4_colocate.log 2>&1
RCC=$?; TC=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn4_colocate exit=${RCC} wall=$((TC-TB)) ==="

# --- D: single-node control (32 BD), bracket end ---
mpirun --bind-to none -x PS_EB_NODES -H ${N0}:34 -n 34 ${BENCH} > ${OUTDIR}/enl4_sn_c.log 2>&1
RCD=$?; TD=$(date +%s)
echo "=== CONFIG MARKER: CFG=sn_c exit=${RCD} wall=$((TD-TC)) ==="

echo "=== DONE: sn_a=${RCA} mn4_default=${RCB} mn4_colocate=${RCC} sn_c=${RCD} total_wall=$((TD-T0)) ==="
