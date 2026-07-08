#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:data
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=33
#SBATCH --time=00:30:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_mn2x32

# psana analog of raw-IO config B5: 64 BD workers packed 32-ish per node
# on 2 nodes (66 ranks: smd0 + 1 EB + 64 BD). Raw readers at this layout
# sustain 15.2 GB/s; if psana approaches that (~450 Hz), the multi-node
# shortfall was under-concurrency per node, not the serving chain.

set -x
cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh
export PS_EB_NODES=1

OUTDIR=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101210926/xtc

T0=$(date +%s)
mpirun --bind-to none -x PS_EB_NODES -n 66 \
    python psana/psana/gpu/bench_calib.py -e mfx101210926 -r 387 \
    -n 100 --warmup 10 --dir ${FFB} > ${OUTDIR}/sweep_mn2x32.log 2>&1
RC=$?
T1=$(date +%s)
echo "=== CONFIG MARKER: CFG=mn2x32 exit_code=${RC} wall_seconds=$((T1 - T0)) ==="
