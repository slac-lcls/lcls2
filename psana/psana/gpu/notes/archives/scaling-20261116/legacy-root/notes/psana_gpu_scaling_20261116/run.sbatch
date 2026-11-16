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
#SBATCH --time=00:45:00
#SBATCH --job-name=pr298-gpu-scale
set -euo pipefail
attempt=${1:?prepared attempt}
bundle="$attempt/bundle"
( set -o noclobber; date -u > "$attempt/job-started.txt" )
mkdir -p "$attempt/tmp" "$attempt/cupy-cache"
TZ=UTC scontrol show job "$SLURM_JOB_ID" > "$attempt/slurm-job.txt"
lscpu > "$attempt/cpu.txt"
/usr/bin/python3 "$bundle/clean_exec.py" "$attempt" python "$bundle/campaign.py" "$attempt"
