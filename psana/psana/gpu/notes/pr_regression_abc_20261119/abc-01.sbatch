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
#SBATCH --time=06:00:00
#SBATCH --job-name=observer-abc
#SBATCH --output=/pscratch/sd/m/monarin/psana2-gpu/pr-regression-abc-20261119/job-%j.log
set -euo pipefail
reviewed_commit=${1:?Pass the reviewed full commit SHA to sbatch}
root=/pscratch/sd/m/monarin/psana2-gpu/pr-regression-abc-20261119
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
task_python=/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python
inputs=/pscratch/sd/m/monarin/psana2-gpu/pr-regression-accounting-20261118/inputs-01.json
inputs_sha=8c8ff73085849969c8f162a7a1f5b13ff03e6419dd6dd7a93f811414daf0fbdd
"$task_python" -B -s overhead.py --abc --pilot --reviewed-commit "$reviewed_commit" \
 --inputs "$inputs" --inputs-sha256 "$inputs_sha" --output "$root/preflight-$SLURM_JOB_ID"
pilot_sha=$(sha256sum "$root/preflight-$SLURM_JOB_ID/complete.json" | cut -d ' ' -f1)
"$task_python" -B -s overhead.py --abc --reviewed-commit "$reviewed_commit" \
 --inputs "$inputs" --inputs-sha256 "$inputs_sha" \
 --pilot-output "$root/preflight-$SLURM_JOB_ID" --pilot-sha256 "$pilot_sha" \
 --output "$root/main-$SLURM_JOB_ID"
