#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:default
#SBATCH --qos=preemptable
#SBATCH --nodes=3
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=33
#SBATCH --time=00:25:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_eb_node_local_3n_v2

# Iteration 20: clean 3-node compounding bracket for node-local EB placement
# (PS_EB_NODE_LOCAL). Supersedes jobs 31573442/31573443 (iter-19-addendum), which
# each failed the same TWO ways:
#   1. sn control never mapped: it asked `-H N0:34 -n 34` on a 33-slot/node
#      allocation with no --oversubscribe -> exit 213 "Out of resource". So the
#      in-window single-node normalization iter-19 needed never existed.
#   2. colocate (phase 3 of 4) was preempted ~6 min in -> "PRTE lost
#      communication with a remote daemon", no aggregate. The default numbers
#      that DID land (3n 274.9 Hz, 4n 294.4 Hz) came from that degraded,
#      uncontrolled window and are not a valid scaling curve.
#
# Two fixes, no science change (default-vs-colocate stays the single variable):
#   - sn control gets --oversubscribe so it actually runs.
#   - front-load the fragile phases: sn_a (control) -> colocate (the phase that
#     keeps dying) -> default. Trailing sn_c dropped. -n cut 150->120. Target
#     ~4.5 min total so all three land before the ~6 min preemption window seen
#     on both prior jobs.
#
# One variable: PS_EB_NODE_LOCAL off (single EB on N0) vs on (one EB/node) at
# 3-node/97-BD, both against the sn_a/32-BD control in the same allocation for
# per-node efficiency. iter-18 confirmed the win at 2 nodes (colocate 328.4 vs
# 301.0 default mean, +9.1%, ~90% per-node eff vs 83%). Compounding verdict:
# colocate holds ~90% per-node eff (3x sn) while default degrades below its
# 2-node 83% -> node-local EB compounds; make PS_EB_NODE_LOCAL the multi-node
# default. Colocate holds near default -> does not compound; lever moves to
# intra-EB serialization (2nd EB/node or fatter serve batch).

set -x
cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh
export PS_EB_NODES=1

OUTDIR=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
N0=${NODES[0]}; N1=${NODES[1]}; N2=${NODES[2]}
echo "=== NODES: N0=${N0} N1=${N1} N2=${N2} ==="

BENCH="python psana/psana/gpu/bench_calib.py -e mfx101572426 -r 47 -n 120 --warmup 10 --wait-split --dir ${FFB}"

# --- A: single-node control (32 BD), in-window per-node baseline ---
T0=$(date +%s)
mpirun --bind-to none --oversubscribe -x PS_EB_NODES -H ${N0}:34 -n 34 ${BENCH} > ${OUTDIR}/enl3v2_sn_a.log 2>&1
RCA=$?; TA=$(date +%s)
echo "=== CONFIG MARKER: CFG=sn_a exit=${RCA} wall=$((TA-T0)) ==="

# --- B: 3-node node-local EBs (one EB per node) -- fragile phase, run early ---
env PS_EB_NODE_LOCAL=1 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -H ${N0}:33,${N1}:33,${N2}:33 -n 99 ${BENCH} > ${OUTDIR}/enl3v2_colocate.log 2>&1
RCB=$?; TB=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn3_colocate exit=${RCB} wall=$((TB-TA)) ==="

# --- C: 3-node default modulo placement (single EB on N0) ---
mpirun --bind-to none -x PS_EB_NODES -H ${N0}:33,${N1}:33,${N2}:33 -n 99 ${BENCH} > ${OUTDIR}/enl3v2_default.log 2>&1
RCC=$?; TC=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn3_default exit=${RCC} wall=$((TC-TB)) ==="

echo "=== DONE: sn_a=${RCA} mn3_colocate=${RCB} mn3_default=${RCC} total_wall=$((TC-T0)) ==="
