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
#SBATCH --time=01:10:00
#SBATCH --job-name=sim10-baseline-bootstrap
set -euo pipefail
attempt=${1:?prepared attempt}
[[ "$attempt" == /pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-20261115/bootstrap02r2 ]]
(set -o noclobber; date -u > "$attempt/job-started.txt")
TZ=UTC scontrol show job "$SLURM_JOB_ID" > "$attempt/slurm-job.txt"
lscpu > "$attempt/cpu.txt"
/usr/bin/python3 "$attempt/bundle/clean_exec.py" "$attempt" python "$attempt/bundle/campaign.py" "$attempt"
