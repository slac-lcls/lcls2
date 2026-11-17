#!/bin/bash
#SBATCH --account=lcls_g
#SBATCH --qos=premium
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=runtime-capture
#SBATCH --output=/pscratch/sd/m/monarin/psana2-gpu/pr-regression-runtimes-20261117/capture-%j.log
set -euo pipefail
module load gcc-native/13.2 cudatoolkit/12.9
root=/pscratch/sd/m/monarin/psana2-gpu/pr-regression-runtimes-20261117
export TMPDIR=/pscratch/sd/m/monarin/codex-tmp PYTHONDONTWRITEBYTECODE=1 MPICH_GPU_SUPPORT_ENABLED=0
unset PYTHONPATH PYTHONHOME LD_PRELOAD
cd "$root/frozen-01"
sha256sum --check SHA256SUMS
srun --mpi=cray_shasta --ntasks=1 --cpus-per-task=4 --gpus-per-task=1 --gpu-bind=single:1 \
  /pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python -B -s prepare_runtime.py capture \
  --receipt "$root/build-01/build-complete.json" \
  --receipt-sha256 1a9b3ccf684865c5973b2d3e59ac6894f38643de104f33d1e39d4ed2b81308e7 \
  --output "$root/runtime-01.json"
