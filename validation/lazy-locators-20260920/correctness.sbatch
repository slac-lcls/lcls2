#!/bin/bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --exclude=sdfampere013,sdfampere024
#SBATCH --job-name=lazy-locator-test
#SBATCH --output=validation/lazy-locators-20260920/correctness-%j.log
cd /sdf/home/m/monarin/lcls2_worktree/psana2-gpu-batched-locators
source setup_env.sh >/dev/null
set -euo pipefail
source install_psana/activate.sh
export PYTHONPATH="$PWD/validation/lazy-locators-20260920/install/lib/python3.9/site-packages"
export PS_PARALLEL=mpi OMPI_MCA_btl='^smcuda' TMPDIR=/tmp
export CUPY_CACHE_DIR="$PWD/validation/lazy-locators-20260920/cupy-cache"
export LCLS_CALIB_HTTP=https://pswww.slac.stanford.edu/ws
unset SIT_PSDM_OFFSITE
hostname
nvidia-smi --query-gpu=uuid,name,driver_version --format=csv
python - <<'PY'
from pathlib import Path
import cupy as cp
from psana.gpu.gpudgram import parser
source = Path('psana/psana/gpu/gpudgram/parser.py')
assert Path(parser.__file__).read_bytes() == source.read_bytes()
assert 'validation/lazy-locators-20260920/install/' in parser.__file__
assert cp.cuda.runtime.getDeviceCount() > 0
print('Parser:', parser.__file__)
print('CuPy:', cp.__version__, 'CUDA:', cp.cuda.runtime.runtimeGetVersion())
PY
python -m pytest -q psana/psana/tests/gpu/unit psana/psana/tests/gpu/integration -m 'not slow'
