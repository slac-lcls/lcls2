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
#SBATCH --job-name=prreg-sdf-cleanup
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
source_dir=${2:?frozen regression source directory}
"$venv/bin/python" -B -s - "$attempt/runtime.json" <<'PYRUNTIME'
import json,os,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(python=sys.executable,pythonpath='',
    scope='minimal environment for disposable cleanup workers only; not a psana runtime qualification',
    environment={k:os.environ[k] for k in ('PATH','LD_LIBRARY_PATH','CUDA_HOME','CUDA_PATH','CONDA_PREFIX','LCLS_CALIB_HTTP')})))
PYRUNTIME
runtime_sha=$(sha256sum "$attempt/runtime.json" | cut -d ' ' -f 1)
"$venv/bin/python" -B -s "$source_dir/qualify_cleanup.py" --site sdf \
    --sdf-mpiexec "$base/bin/mpiexec" --runtime "$attempt/runtime.json" \
    --runtime-sha256 "$runtime_sha" --output "$attempt/job-$SLURM_JOB_ID"
