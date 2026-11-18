#!/bin/bash
#SBATCH --account=lcls_g
#SBATCH --qos=premium
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gres-flags=disable-binding
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=cleanup-qualification
#SBATCH --output=/pscratch/sd/m/monarin/psana2-gpu/pr-regression-cleanup-20261118/job-%j.log
set -euo pipefail
root=/pscratch/sd/m/monarin/psana2-gpu/pr-regression-cleanup-20261118
record_exit() {
  result=$?
  timeout 30s sacct -j "$SLURM_JOB_ID" --format=JobIDRaw,State,ExitCode,Elapsed,NodeList,AllocTRES,ReqTRES,Account,QOS,Timelimit -P > "$root/job-$SLURM_JOB_ID-exit-accounting.txt" || true
  exit "$result"
}
trap record_exit EXIT
module load gcc-native/13.2 cudatoolkit/12.9
export TMPDIR=/pscratch/sd/m/monarin/codex-tmp PYTHONDONTWRITEBYTECODE=1 MPICH_GPU_SUPPORT_ENABLED=0
unset PYTHONPATH PYTHONHOME LD_PRELOAD
cd "$root/frozen-02"
sha256sum --check SHA256SUMS
/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python -B -s qualify_cleanup.py \
 --runtime /pscratch/sd/m/monarin/psana2-gpu/pr-regression-runtimes-20261117/runtime-01.json \
 --runtime-sha256 6082ff8edb359bd8596cddb6e7bb070b09ddb641ec8800a26a1ace36283a5b35 \
 --output "$root/cleanup-$SLURM_JOB_ID"
