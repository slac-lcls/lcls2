#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:data
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=2
#SBATCH --ntasks=38
#SBATCH --ntasks-per-node=19
#SBATCH --time=00:40:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_eb_sweep

set -x

cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh

OUTDIR=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep
BENCH=/sdf/scratch/users/a/ajshack/lcls2/psana/psana/gpu/bench_calib.py
XTCDIR=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101210926/xtc

echo "which mpirun: $(which mpirun)"
echo "nodelist: ${SLURM_JOB_NODELIST}"
srun --overlap -N ${SLURM_NNODES} --ntasks-per-node=1 hostname

for EB in 1 2 4; do
    NP=$((33 + EB))
    export PS_EB_NODES=${EB}

    echo "=== starting config EB=${EB} NP=${NP} ==="
    T0=$(date +%s)

    mpirun --bind-to none -x PS_EB_NODES -n ${NP} python ${BENCH} \
        -e mfx101210926 -r 387 -n 200 --warmup 10 --dir ${XTCDIR} \
        > ${OUTDIR}/sweep_eb${EB}.log 2>&1
    RC=$?

    T1=$(date +%s)
    WALL=$((T1 - T0))
    echo "=== CONFIG MARKER: EB=${EB} NP=${NP} exit_code=${RC} wall_seconds=${WALL} ==="
done

echo "=== eb sweep complete ==="
