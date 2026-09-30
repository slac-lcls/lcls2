#!/bin/bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=450G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --exclude=sdfampere013,sdfampere024,sdfampere036
#SBATCH --job-name=four-way-gpu
#SBATCH --output=validation/four-way-20260918/job-%j.log
cd /sdf/home/m/monarin/lcls2_worktree/psana2-gpu-batched-locators
source setup_env.sh >/dev/null
set -euo pipefail
ulimit -c 0
source install_psana/activate.sh
export PS_PARALLEL=mpi PS_EB_NODES=1 PS_SRV_NODES=0 PS_SMD_N_EVENTS=1000
export OMPI_MCA_btl='^smcuda' TMPDIR=/tmp PYTHONUNBUFFERED=1
export CUPY_CACHE_DIR="$PWD/validation/four-way-20260918/cupy-cache"
export KVIKIO_NTHREADS=8 KVIKIO_TASK_SIZE=1048576 KVIKIO_COMPAT_MODE=ON
export LCLS_CALIB_HTTP=https://pswww.slac.stanford.edu/ws
unset SIT_PSDM_OFFSITE PSANA_GPU_MEM_DEBUG
hostname
python -c 'import psana, cupy, psana.gpu.gpudgram.parser as p; print(psana.__file__, p.__file__, cupy.__version__, cupy.cuda.runtime.runtimeGetVersion())'
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
lscpu
numactl --hardware
nvidia-smi topo -m
findmnt -T /lscratch -o TARGET,SOURCE,FSTYPE,OPTIONS
python -u validation/four-way-20260918/run_warm_ab.py --skip-trace "$@"
