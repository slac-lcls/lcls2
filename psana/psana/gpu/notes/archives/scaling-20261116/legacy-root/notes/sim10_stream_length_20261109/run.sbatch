#!/bin/bash
#SBATCH --account=lcls_g
#SBATCH --constraint=gpu
#SBATCH --qos=premium
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --job-name=sim10-stream-length
set -euo pipefail
source_dir=${1:?source}
attempt=${2:?unique attempt directory}
mkdir -p "$attempt/tmp" "$attempt/cupy-cache"
/usr/bin/python3 "$source_dir/clean_exec.py" "$attempt" python "$source_dir/campaign.py" "$attempt/results"
