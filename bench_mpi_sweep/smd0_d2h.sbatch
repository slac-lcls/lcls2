#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:data
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=2
#SBATCH --ntasks=38
#SBATCH --ntasks-per-node=19
#SBATCH --time=00:40:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_smd0_d2h

set -x

cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh
export PS_EB_NODES=1

OUTDIR=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep
BENCH=/sdf/scratch/users/a/ajshack/lcls2/psana/psana/gpu/bench_calib.py
XTCDIR=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101210926/xtc
NP=34

echo "which mpirun: $(which mpirun)"
echo "nodelist: ${SLURM_JOB_NODELIST}"
srun --overlap -N ${SLURM_NNODES} --ntasks-per-node=1 hostname

# --- CONFIG SET 1: PS_SMD_N_EVENTS sweep (NBD=32) ---
for SMDN in 1000 4000 16000; do
    export PS_SMD_N_EVENTS=${SMDN}
    echo "=== starting config SMDN=${SMDN} NP=${NP} ==="
    T0=$(date +%s)
    mpirun --bind-to none -x PS_EB_NODES -x PS_SMD_N_EVENTS -n ${NP} python ${BENCH} \
        -e mfx101210926 -r 387 -n 200 --warmup 10 --dir ${XTCDIR} \
        > ${OUTDIR}/sweep_smd${SMDN}.log 2>&1
    RC=$?
    T1=$(date +%s)
    echo "=== CONFIG MARKER: SMDN=${SMDN} NP=${NP} exit_code=${RC} wall_seconds=$((T1 - T0)) ==="
done
unset PS_SMD_N_EVENTS

# --- CONFIG SET 2: D2H on/off at NBD=32, default PS_SMD_N_EVENTS ---
for D2H in off on; do
    EXTRA=""
    [ "${D2H}" = "on" ] && EXTRA="--d2h"
    echo "=== starting config D2H=${D2H} NP=${NP} ==="
    T0=$(date +%s)
    mpirun --bind-to none -x PS_EB_NODES -n ${NP} python ${BENCH} \
        -e mfx101210926 -r 387 -n 200 --warmup 10 --dir ${XTCDIR} ${EXTRA} \
        > ${OUTDIR}/sweep_d2h_${D2H}.log 2>&1
    RC=$?
    T1=$(date +%s)
    echo "=== CONFIG MARKER: D2H=${D2H} NP=${NP} exit_code=${RC} wall_seconds=$((T1 - T0)) ==="
done

echo "=== smd0/d2h sweep complete ==="
