#!/bin/bash
#SBATCH -A lcls_g
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -t 00:20:00
#SBATCH -J pr187-ipc-prepare
#SBATCH -o /pscratch/sd/m/monarin/psana2-gpu/pr187-ipc-startup-perlmutter-20261027/attempt01/logs/prepare-%j.log
set -euo pipefail
root=/pscratch/sd/m/monarin/psana2-gpu/pr187-ipc-startup-perlmutter-20261027/attempt01
/usr/bin/python3 "$root/scripts/clean_exec.py" "$root" A env PS_PARALLEL=none CUDA_VISIBLE_DEVICES= python -B "$root/scripts/prepare.py"
