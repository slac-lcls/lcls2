#!/bin/bash
#SBATCH --account=lcls_g
#SBATCH --qos=premium
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=run51-prepare
#SBATCH --output=/pscratch/sd/m/monarin/psana2-gpu/pr-regression-inputs-20261117/prepare-%j.log
set -euo pipefail
root=/pscratch/sd/m/monarin/psana2-gpu/pr-regression-inputs-20261117
export TMPDIR=/pscratch/sd/m/monarin/codex-tmp
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME LD_PRELOAD
cd "$root/frozen"
sha256sum --check SHA256SUMS
exec /pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python -B -s prepare_inputs.py prepare \
  --plan "$root/plan-01.json" \
  --plan-sha256 dd791f0a19e13828e5b7e9ae0cee613c4f136c2e4c7e52dc64e15bd2a65522c3 \
  --destination /pscratch/sd/m/monarin/psana2-gpu/datasets/mfx100848724-r0051-pr-regression-20261117
