#!/bin/bash
#SBATCH --account=lcls:data@ampere
#SBATCH --partition=ampere
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --gres-flags=disable-binding
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=prreg-sdf-site
set -euo pipefail
attempt=${1:?task attempt directory}
venv=/sdf/scratch/users/m/monarin/gpu-validation/kvikio-2610-validation-20261015-r1/venv
base=/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/xpp_drp_gpu_311
export PATH="$venv/bin:$base/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="$base/lib"
export CUDA_HOME=/sdf/group/lcls/ds/tools/cuda12.8 CUDA_PATH=/sdf/group/lcls/ds/tools/cuda12.8
export CONDA_PREFIX="$base" LCLS_CALIB_HTTP=https://pswww.slac.stanford.edu/calib_ws/
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
export TMPDIR="$attempt/tmp" CUPY_CACHE_DIR="$attempt/cupy-cache"
mkdir -p "$TMPDIR" "$CUPY_CACHE_DIR"
"$venv/bin/python" -B -s "$attempt/source/launch_probe.py" "$attempt/job-$SLURM_JOB_ID"
