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
#SBATCH --time=00:03:00
#SBATCH --job-name=cleanup-name-diagnostic
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
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=1 --overlap --exact --gpus-per-node=0 --cpu-bind=none --job-name=prreg-12345678901234567890 sleep 20 &
first_pid=$!
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=1 --overlap --exact --gpus-per-node=0 --cpu-bind=none --job-name=prreg-12345678901234567890-sentinel sleep 20 &
second_pid=$!
sleep 5
squeue --steps --jobs="$SLURM_JOB_ID" --noheader --format='%i|%j' > "$root/diagnostic-$SLURM_JOB_ID-default.txt"
squeue --steps --jobs="$SLURM_JOB_ID" --noheader --format='%i|%.128j' > "$root/diagnostic-$SLURM_JOB_ID-wide.txt"
scontrol show step "$SLURM_JOB_ID" > "$root/diagnostic-$SLURM_JOB_ID-scontrol.txt"
wait "$first_pid"
wait "$second_pid"
