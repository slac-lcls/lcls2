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
#SBATCH --job-name=pr298-stripe-pilot
set -euo pipefail
bundle=${1:?bundle}
attempt=${2:?attempt}
mode=${3:?pilot or screen}
( set -o noclobber; date -u > "$attempt/job-started.txt" )
mkdir -p "$attempt/tmp" "$attempt/cupy-cache"
scontrol show job "$SLURM_JOB_ID" > "$attempt/slurm-job.txt"
lscpu > "$attempt/cpu.txt"
/usr/bin/python3 "$bundle/clean_exec.py" "$attempt" python "$bundle/campaign.py" "$attempt" "$mode"
