#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:default
#SBATCH --qos=preemptable
#SBATCH --nodes=4
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=33
#SBATCH --time=00:15:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_eb_3eb_cold

# Cold-window re-measurement of the iter-26 winner: PS_EB_PER_NODE=3 at 4 nodes,
# node-local colocation. In the iter-26 bracket this config ran as the THIRD
# phase (914.1 Hz, warmed FFB window); here it runs as the sole first phase so
# the number is a clean cold-window point for the k=3 placement.

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
env PS_EB_NODE_LOCAL=1 PS_EB_PER_NODE=3 mpirun --bind-to none -x PS_EB_NODES -x PS_EB_NODE_LOCAL -x PS_EB_PER_NODE -H ${HOSTS} -n 132 ${BENCH} > ${OUTDIR}/epn4nk_3eb_cold.log 2>&1
RC=$?; T1=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn4_3eb_cold exit=${RC} wall=$((T1-T0)) ==="
