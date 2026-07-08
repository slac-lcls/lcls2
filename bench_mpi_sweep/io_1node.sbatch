#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_io_1node

set -x
cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh   # consistent python; no psana used

OUTDIR=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep
READER=${OUTDIR}/io_reader.py
PLAN=${OUTDIR}/plan.tsv
echo "nodes: ${SLURM_JOB_NODELIST}"

for CFG in B1_1 B1_8 B1_32 B3_rand; do
    echo "=== starting config ${CFG} ==="
    T0=$(date +%s)
    python ${READER} --plan ${PLAN} --config ${CFG} --node-index 0 \
        > ${OUTDIR}/io_${CFG}.log 2>&1
    RC=$?
    T1=$(date +%s)
    echo "=== CONFIG MARKER: CFG=${CFG} nodes=1 exit_code=${RC} wall_seconds=$((T1 - T0)) ==="
done
echo "=== io 1-node complete ==="
