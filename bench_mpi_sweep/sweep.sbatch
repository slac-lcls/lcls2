#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:data
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=34
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_mpi_sweep

set -x

cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh
export PS_EB_NODES=1

OUTDIR=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep
BENCH=/sdf/scratch/users/a/ajshack/lcls2/psana/psana/gpu/bench_calib.py

echo "which mpirun: $(which mpirun)"
echo "which python: $(which python)"
echo "python version: $(python --version 2>&1)"

for NBD in 1 2 4 8 16 32; do
    NP=$((NBD + 2))

    echo "=== starting config NBD=${NBD} NP=${NP} ==="

    # NIC sampler: append timestamp + /proc/net/dev every 2s
    NICLOG=${OUTDIR}/nic_bd${NBD}.ffb.log
    : > "${NICLOG}"
    (
        while true; do
            date +%s >> "${NICLOG}"
            cat /proc/net/dev >> "${NICLOG}"
            sleep 2
        done
    ) &
    SAMPLER_PID=$!

    T0=$(date +%s)

    if [ -x "$(command -v mpirun)" ]; then
        mpirun -n ${NP} --bind-to none python ${BENCH} \
            -e mfx101210926 -r 387 -n 200 --warmup 10 --dir /sdf/data/lcls/drpsrcf/ffb/mfx/mfx101210926/xtc \
            > ${OUTDIR}/sweep_bd${NBD}.ffb.log 2>&1
        RC=$?
    else
        srun --mpi=pmix -n ${NP} --overlap --cpu-bind=none python ${BENCH} \
            -e mfx101210926 -r 387 -n 200 --warmup 10 --dir /sdf/data/lcls/drpsrcf/ffb/mfx/mfx101210926/xtc \
            > ${OUTDIR}/sweep_bd${NBD}.ffb.log 2>&1
        RC=$?
    fi

    T1=$(date +%s)
    WALL=$((T1 - T0))

    kill "${SAMPLER_PID}" 2>/dev/null
    wait "${SAMPLER_PID}" 2>/dev/null

    echo "=== CONFIG MARKER: NBD=${NBD} NP=${NP} exit_code=${RC} wall_seconds=${WALL} ==="
done

echo "=== sweep complete ==="
