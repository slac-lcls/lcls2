#!/bin/bash
#SBATCH --account=lcls_g
#SBATCH --qos=premium
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@45
#SBATCH --job-name=pr299-pilot
#SBATCH --output=/pscratch/sd/m/monarin/agent-work/lcls2/pr299-perlmutter-screen-20261115TKJ0du6/logs/slurm-%j.out
#SBATCH --error=/pscratch/sd/m/monarin/agent-work/lcls2/pr299-perlmutter-screen-20261115TKJ0du6/logs/slurm-%j.err
TASK_ROOT=/pscratch/sd/m/monarin/agent-work/lcls2/pr299-perlmutter-screen-20261115TKJ0du6
export TMPDIR="$TASK_ROOT/tmp" XDG_CACHE_HOME="$TASK_ROOT/cache" PIP_CACHE_DIR="$TASK_ROOT/cache/pip"
source /global/u2/m/monarin/psana-nersc/activate_psana_build_env.sh /global/homes/m/monarin/.conda-envs/psana-build
set -euo pipefail
test "$(realpath "$CONDA_PREFIX")" = "$(realpath /global/homes/m/monarin/.conda-envs/psana-build)"
python -c 'import sys; assert sys.prefix == "/global/homes/m/monarin/.conda-envs/psana-build"'
unset PYTHONPATH SIT_PSDM_OFFSITE
module load cudatoolkit/12.9
export CUDA_PATH="$CUDA_HOME"
export SCREEN_RUNTIME_LIBS="$CUDA_PATH/lib64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$SCREEN_RUNTIME_LIBS"
export PYTHONDONTWRITEBYTECODE=1 MPICH_GPU_SUPPORT_ENABLED=0
export LCLS_CALIB_HTTP=https://pswww.slac.stanford.edu/calib_ws/
cd "$TASK_ROOT"
exec python -B "$TASK_ROOT/harness/perlmutter_screen.py" "${1:?pilot or campaign}"
