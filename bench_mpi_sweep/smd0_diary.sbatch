#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:data
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=34
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_smd0_diary

set -x

cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh
export PS_EB_NODES=1

OUTDIR=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep
BENCH=/sdf/scratch/users/a/ajshack/lcls2/psana/psana/gpu/bench_calib.py
XTCDIR=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101210926/xtc

echo "nodelist: ${SLURM_JOB_NODELIST}"
T0=$(date +%s)
mpirun --bind-to none -x PS_EB_NODES -n 34 python ${BENCH} \
    -e mfx101210926 -r 387 -n 200 --warmup 10 --dir ${XTCDIR} --smd0-debug \
    > ${OUTDIR}/sweep_smd0diag.log 2>&1
RC=$?
T1=$(date +%s)
echo "=== CONFIG MARKER: SMD0DIAG NP=34 exit_code=${RC} wall_seconds=$((T1 - T0)) ==="
