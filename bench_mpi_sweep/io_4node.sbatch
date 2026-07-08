#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:data
#SBATCH --nodes=4
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:15:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_io_4node

set -x
cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh

OUTDIR=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep
READER=${OUTDIR}/io_reader.py
PLAN=${OUTDIR}/plan.tsv
echo "nodes: ${SLURM_JOB_NODELIST}"

echo "=== starting config B2_4n ==="
T0=$(date +%s)
srun -N4 -n4 --cpus-per-task=32 \
    python ${READER} --plan ${PLAN} --config B2_4n \
    > ${OUTDIR}/io_B2_4n.log 2>&1
RC=$?
T1=$(date +%s)
echo "=== CONFIG MARKER: CFG=B2_4n nodes=4 exit_code=${RC} wall_seconds=$((T1 - T0)) ==="
echo "=== io 4-node complete ==="
