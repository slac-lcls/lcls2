#!/bin/bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --exclude=sdfampere013
#SBATCH --job-name=canonical-gather-test
#SBATCH --output=validation/batched-gather-20260918/correctness-%j.log
cd /sdf/home/m/monarin/lcls2_worktree/psana2-gpu-batched-locators
source setup_env.sh >/dev/null
set -euo pipefail
source install_psana/activate.sh
export PS_PARALLEL=mpi OMPI_MCA_btl='^smcuda' TMPDIR=/tmp
export CUPY_CACHE_DIR="$PWD/validation/batched-gather-20260918/cupy-cache"
export LCLS_CALIB_HTTP=https://pswww.slac.stanford.edu/ws
unset SIT_PSDM_OFFSITE
hostname
nvidia-smi --query-gpu=uuid,name,driver_version --format=csv
python -m pytest -q psana/psana/tests/gpu/unit psana/psana/tests/gpu/integration -m 'not slow'
