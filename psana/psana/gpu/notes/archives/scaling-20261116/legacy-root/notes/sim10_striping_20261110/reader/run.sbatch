#!/bin/bash
#SBATCH --account=lcls_g
#SBATCH --constraint=gpu
#SBATCH --qos=premium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=sim10-stripe-reader
set -euo pipefail
set -o noclobber
attempt=$(realpath -e "${1:?attempt}")
mode=${2:?pilot or sweep}
campaign=/pscratch/sd/m/monarin/psana2-gpu/sim10-striping-20261110
case "$attempt" in "$campaign"/reader-*) ;; *) exit 20;; esac
[[ "$(pwd -P)" == "$attempt" ]]
[[ "$mode" == pilot || "$mode" == sweep ]]
mkdir "$attempt/job-started"
trap 'result=$?; /usr/sbin/lctl get_param "llite.*.stats" > "$attempt/lustre-counters-after.txt" 2> "$attempt/lustre-counters-after-error.txt" || true; exit "$result"' EXIT
# Submission pins --chdir, --output and --error to this fresh attempt.
scontrol show job "$SLURM_JOB_ID" > "$attempt/slurm-job.txt"
lscpu > "$attempt/cpu.txt"
uname -a > "$attempt/kernel.txt"
findmnt -T "$attempt" > "$attempt/mount.txt"
/usr/bin/lfs --version > "$attempt/lustre-version.txt"
# Read-only counter availability probe; no resets. Failure is recorded explicitly.
if ! /usr/sbin/lctl get_param 'llite.*.stats' > "$attempt/lustre-counters-before.txt" 2> "$attempt/lustre-counters-error.txt"; then
    echo 'Lustre byte counters unavailable; remote bytes remain unverified.'
fi
python_bin=/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python
srun --ntasks=1 --cpus-per-task=128 --cpu-bind=none "$python_bin" -u "$attempt/bundle/campaign.py" --attempt "$attempt" --mode "$mode"
