#!/bin/bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=450G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --exclude=sdfampere013
#SBATCH --job-name=batched-locators
#SBATCH --output=validation/batched-locators-20260917/job-%j.log
cd /sdf/home/m/monarin/lcls2_worktree/psana2-gpu-batched-locators
source setup_env.sh >/dev/null
set -euo pipefail
ulimit -c 0
source install_psana/activate.sh
export PS_PARALLEL=mpi PS_EB_NODES=1 PS_SRV_NODES=0 PS_SMD_N_EVENTS=1000
export OMPI_MCA_btl='^smcuda' TMPDIR=/tmp PYTHONUNBUFFERED=1
export CUPY_CACHE_DIR="$PWD/validation/batched-locators-20260917/cupy-cache"
export KVIKIO_NTHREADS=8 KVIKIO_TASK_SIZE=1048576 KVIKIO_COMPAT_MODE=ON
export LCLS_CALIB_HTTP=https://pswww.slac.stanford.edu/ws
unset SIT_PSDM_OFFSITE PSANA_GPU_MEM_DEBUG
hostname
python -c 'import psana, cupy, psana.gpu.gpudgram.parser as p; print(psana.__file__, p.__file__, cupy.__version__, cupy.cuda.runtime.runtimeGetVersion())'
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
pytest -q psana/psana/tests/gpu/unit psana/psana/tests/gpu/integration/test_gpudgram_device.py psana/psana/tests/gpu/integration/test_batched_locators.py psana/psana/tests/gpu/integration/test_pixel_exact.py -m 'not slow' > "validation/batched-locators-20260917/gpu-tests-$SLURM_JOB_ID.log" 2>&1
python -u validation/batched-locators-20260917/run_warm_ab.py "$@"
