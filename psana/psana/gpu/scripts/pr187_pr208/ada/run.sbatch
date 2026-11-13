#!/bin/bash
#SBATCH --account=lcls:prjdat21
#SBATCH --qos=normal
#SBATCH --partition=ada
#SBATCH --nodelist=sdfada016
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=700G
#SBATCH --gres=gpu:l40s:10
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --job-name=pr187-208-ada
#SBATCH --output=/sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/logs/driver-%j.log
set -eo pipefail
ulimit -c 0
source /sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/screen/env.sh
python -B /sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/metadata/wait_gpu_idle.py /sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/logs/idle-gate-${SLURM_JOB_ID}.json
python -B /sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/metadata/ampere_inventory.py > /sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/logs/inventory-${SLURM_JOB_ID}.json
python -B /sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/metadata/check_local_storage.py /sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/logs/inventory-${SLURM_JOB_ID}.json /sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/screen/reference.json /sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/logs/storage-gate-${SLURM_JOB_ID}.json
python -B /sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/metadata/network_monitor.py /sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/screen/job-${SLURM_JOB_ID} &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
python -B /sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/screen/campaign.py --root /sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/screen --source /sdf/data/lcls/drpsrcf/ffb/users/monarin/jf-feespec-bulk-38995226/xtc > /sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/logs/screen-${SLURM_JOB_ID}.log 2>&1
