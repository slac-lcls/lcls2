#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:data
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_io_2node

set -x
cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh

OUTDIR=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep
READER=${OUTDIR}/io_reader.py
PLAN=${OUTDIR}/plan.tsv
NODES=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))
echo "nodes: ${NODES[@]}"

# B2a: 32 readers split 16+16 across both nodes (SLURM_PROCID = node index)
echo "=== starting config B2_2n ==="
T0=$(date +%s)
srun -N2 -n2 --cpus-per-task=32 \
    python ${READER} --plan ${PLAN} --config B2_2n \
    > ${OUTDIR}/io_B2_2n.log 2>&1
RC=$?
T1=$(date +%s)
echo "=== CONFIG MARKER: CFG=B2_2n nodes=2 exit_code=${RC} wall_seconds=$((T1 - T0)) ==="

# B4: concurrent file-set isolation, one node per file set
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
echo "=== CONFIG MARKER: CFG=B4 nodes=2 exit_codes=${RC1},${RC2} wall_seconds=$((T1 - T0)) ==="
echo "=== io 2-node complete ==="
