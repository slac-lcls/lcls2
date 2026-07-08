#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:data
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --exclude=sdfampere029,sdfampere040
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=io_b5

# B5: 64 readers (32 per node) over 2 nodes, fresh r400 windows.
# Answers the 4-node job's question inside a schedulable allocation:
# if the FFB cap is truly aggregate ~12 GB/s, doubling per-node
# concurrency changes nothing; if it rises, concurrency was the limit.
# Excluded nodes: the only two that have any of today's windows in
# page cache.

set -x
cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh   # just for a consistent python; no psana used

OUTDIR=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep

T0=$(date +%s)
srun -N2 -n2 --cpus-per-task=32 \
    python ${OUTDIR}/io_reader.py --plan ${OUTDIR}/plan.tsv --config B5_2x32 \
    > ${OUTDIR}/io_B5_2x32.log 2>&1
RC=$?
T1=$(date +%s)
echo "=== CONFIG MARKER: CFG=B5_2x32 exit_code=${RC} wall_seconds=$((T1 - T0)) ==="
