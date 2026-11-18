#!/bin/bash
#SBATCH --account=lcls_g
#SBATCH --qos=premium
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gres-flags=disable-binding
#SBATCH --mem=128G
#SBATCH --time=01:15:00
#SBATCH --job-name=native-smoke
#SBATCH --output=/pscratch/sd/m/monarin/psana2-gpu/pr-regression-smoke-20261118/job-%j.log
set -euo pipefail
root=/pscratch/sd/m/monarin/psana2-gpu/pr-regression-smoke-20261118
record_exit() {
  result=$?
  timeout 30s sacct -j "$SLURM_JOB_ID" --format=JobIDRaw,State,ExitCode,Elapsed,NodeList,AllocTRES,ReqTRES,Account,QOS,Timelimit -P > "$root/job-$SLURM_JOB_ID-exit-accounting.txt" || true
  exit "$result"
}
trap record_exit EXIT
module load gcc-native/13.2 cudatoolkit/12.9
export TMPDIR=/pscratch/sd/m/monarin/codex-tmp PYTHONDONTWRITEBYTECODE=1 MPICH_GPU_SUPPORT_ENABLED=0
unset PYTHONPATH PYTHONHOME LD_PRELOAD
cd "$root/frozen-01"
sha256sum --check SHA256SUMS
/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python -B -s run_smoke.py \
 --inputs "$root/inputs-01.json" --inputs-sha256 8c8ff73085849969c8f162a7a1f5b13ff03e6419dd6dd7a93f811414daf0fbdd \
 --output "$root/smoke-$SLURM_JOB_ID"
