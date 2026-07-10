#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:data
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:55:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_io_matrix

set -x

cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh   # just for a consistent python; no psana used

OUTDIR=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep
READER=${OUTDIR}/io_reader.py
PLAN=${OUTDIR}/plan.tsv

NODES=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))
echo "nodes: ${NODES[@]}"

run_cfg() {
    local CFG=$1 NNODES=$2 LOG=$3
    echo "=== starting config ${CFG} nodes=${NNODES} ==="
    local T0=$(date +%s)
    srun -N ${NNODES} -n ${NNODES} --cpus-per-task=32 \
        python ${READER} --plan ${PLAN} --config ${CFG} > ${LOG} 2>&1
    local RC=$?
    local T1=$(date +%s)
    echo "=== CONFIG MARKER: CFG=${CFG} nodes=${NNODES} exit_code=${RC} wall_seconds=$((T1 - T0)) ==="
}

# B1: single-node reader scaling (all on first node via -N1)
run_cfg B1_1   1 ${OUTDIR}/io_B1_1.log
run_cfg B1_8   1 ${OUTDIR}/io_B1_8.log
run_cfg B1_32  1 ${OUTDIR}/io_B1_32.log

# B2: 32 readers across 2 then 4 nodes
run_cfg B2_2n  2 ${OUTDIR}/io_B2_2n.log
run_cfg B2_4n  4 ${OUTDIR}/io_B2_4n.log

# B3: random-pattern penalty, one node
run_cfg B3_rand 1 ${OUTDIR}/io_B3_rand.log

# B4: concurrent file-set isolation on two distinct nodes
echo "=== starting config B4 (concurrent r387 + r0400) ==="
T0=$(date +%s)
srun -w ${NODES[0]} -N1 -n1 --cpus-per-task=32 \
    python ${READER} --plan ${PLAN} --config B4_r387 --node-index 0 \
    > ${OUTDIR}/io_B4_r387.log 2>&1 &
P1=$!
srun -w ${NODES[1]} -N1 -n1 --cpus-per-task=32 \
    python ${READER} --plan ${PLAN} --config B4_r400 --node-index 0 \
    > ${OUTDIR}/io_B4_r400.log 2>&1 &
P2=$!
wait ${P1}; RC1=$?
wait ${P2}; RC2=$?
T1=$(date +%s)
echo "=== CONFIG MARKER: CFG=B4 exit_codes=${RC1},${RC2} wall_seconds=$((T1 - T0)) ==="

echo "=== io matrix complete ==="
