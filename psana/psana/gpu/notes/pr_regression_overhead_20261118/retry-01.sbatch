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
#SBATCH --time=03:00:00
#SBATCH --job-name=observer-retry
#SBATCH --output=/pscratch/sd/m/monarin/psana2-gpu/pr-regression-overhead-20261118/job-%j.log
set -euo pipefail
reviewed_commit=${1:?Pass the reviewed full commit SHA to sbatch}
root=/pscratch/sd/m/monarin/psana2-gpu/pr-regression-overhead-20261118
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
/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python -B -s overhead.py --reviewed-commit "$reviewed_commit" \
 --inputs /pscratch/sd/m/monarin/psana2-gpu/pr-regression-accounting-20261118/inputs-01.json \
 --inputs-sha256 8c8ff73085849969c8f162a7a1f5b13ff03e6419dd6dd7a93f811414daf0fbdd \
 --retry-parent "$root/full-61542523" \
 --retry-parent-sha256 232aceab5ce6e9087361c055728bb5eeaf33181c5bf8c506b590bcf101bdd662 \
 --output "$root/retry-$SLURM_JOB_ID"
