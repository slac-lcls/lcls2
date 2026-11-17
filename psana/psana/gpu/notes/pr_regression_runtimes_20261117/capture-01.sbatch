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
printf '%s\n' \
  '1c9ee29b2602ae53659443eca2ca29bfbe436f7a7a2d8030c62d65cc9086bbf3  prepare_runtime.py' \
  '5e34f2f916f93b8f6bfa579695abab151df1e980b5ee5857da42949035b8e9a2  native_contract.py' | sha256sum --check -
record_prefix="$root/capture-$SLURM_JOB_ID"
record_exit() {
  capture_status=$?
  sacct -j "$SLURM_JOB_ID" --format=JobID,State,ExitCode,Elapsed,NodeList,AllocTRES,ReqTRES,Account,QOS,Timelimit -P > "$record_prefix-accounting.txt" || true
  exit "$capture_status"
}
trap record_exit EXIT
scontrol show job -d "$SLURM_JOB_ID" > "$record_prefix-allocation.txt"
python3 - <<'PYENV' > "$record_prefix-slurm.json"
import json,os
print(json.dumps({k:v for k,v in os.environ.items() if k.startswith('SLURM_')},indent=2))
PYENV
nvidia-smi --query-gpu=uuid,name,pci.bus_id --format=csv,noheader > "$record_prefix-gpus.csv"
srun --mpi=cray_shasta --ntasks=1 --cpus-per-task=4 --gpus-per-task=1 --gpu-bind=single:1 \
  /pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python -B -s prepare_runtime.py capture \
  --receipt "$root/build-01/build-complete.json" \
  --receipt-sha256 1a9b3ccf684865c5973b2d3e59ac6894f38643de104f33d1e39d4ed2b81308e7 \
  --output "$root/runtime-01.json"
