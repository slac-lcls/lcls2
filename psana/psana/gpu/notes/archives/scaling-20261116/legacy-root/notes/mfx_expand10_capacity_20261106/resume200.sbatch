#!/usr/bin/env bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
set -euo pipefail
capacity_dir=${1:?absolute source directory}
output=${2:?new shared scratch result directory}
node_root=${SLURM_TMPDIR:-/lscratch/$USER}
local_root="$node_root/mfx-capacity-${SLURM_JOB_ID}"
mkdir -p "$node_root"
mkdir -m 700 "$local_root" "$output"
export TMPDIR="$local_root/tmp"
export CUPY_CACHE_DIR="$local_root/cupy-cache"
mkdir "$TMPDIR" "$CUPY_CACHE_DIR"
runtime="$capacity_dir/../mfx_expand10_stage2_20261105/runtime.sh"
export CAPACITY_SOURCE="$capacity_dir"
export CAPACITY_LOCAL="$local_root"
export CAPACITY_OUTPUT="$output"
bash "$runtime" - <<'PY'
import os,shutil
from pathlib import Path
local=Path(os.environ['CAPACITY_LOCAL']);out=Path(os.environ['CAPACITY_OUTPUT'])
assert shutil.disk_usage(local).free > 40*(1<<30), 'Need 40 GiB node-local space'
assert shutil.disk_usage(out).free > 2*(1<<30), 'Need 2 GiB shared scratch space'
import hashlib,json
launcher=Path(os.environ['CAPACITY_SOURCE'])/'resume200.sbatch'
(out/'launcher.json').write_text(json.dumps({'path':str(launcher),'sha256':hashlib.sha256(launcher.read_bytes()).hexdigest(),'job':os.environ['SLURM_JOB_ID']})+'\n')
print('Storage preflight passed',local,out,flush=True)
PY
bash "$runtime" -m unittest discover -s "$capacity_dir" -p test_dense_cube.py -v
bash "$runtime" - <<'PY'
import json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['CAPACITY_SOURCE'])
from test_dense_cube import device_tests
report=device_tests(Path(os.environ['CAPACITY_LOCAL'])/'controlled-tests')
(Path(os.environ['CAPACITY_OUTPUT'])/'controlled-tests.json').write_text(json.dumps(report,indent=2)+'\n')
print(report,flush=True)
PY
bash "$runtime" - <<'PY'
import json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['CAPACITY_SOURCE'])
from validate import prepare
root=Path('/sdf/scratch/users/m/monarin/gpu-validation/mfx-expand10/20261105-stage2-r1/pilot')
r=prepare(root,Path(os.environ['CAPACITY_LOCAL'])/'seed-cache')
(Path(os.environ['CAPACITY_OUTPUT'])/'seed-cache-manifest.json').write_text(json.dumps(r,indent=2)+'\n')
print('CPU seed cache ready:',r['seed_cache_bytes'],'bytes',flush=True)
PY
for bins in 200; do
    bash "$runtime" "$capacity_dir/runner.py" --bins "$bins" --output "$output/bins-$bins" \
        --local "$local_root/bins-$bins" --seed-cache "$local_root/seed-cache"
done
# Remove only this job's newly created local workspace after all cases pass.
bash "$runtime" - <<'PY'
import json,os,shutil
from pathlib import Path
local=Path(os.environ['CAPACITY_LOCAL']);out=Path(os.environ['CAPACITY_OUTPUT'])
assert local.name=='mfx-capacity-'+os.environ['SLURM_JOB_ID']
for b in (200,):assert json.loads((out/f'bins-{b}/report.json').read_text())['status']=='pass'
shutil.rmtree(local)
(out/'completion.json').write_text(json.dumps({'status':'pass','local_workspace_removed':str(local),'bins':[200]})+'\n')
PY
