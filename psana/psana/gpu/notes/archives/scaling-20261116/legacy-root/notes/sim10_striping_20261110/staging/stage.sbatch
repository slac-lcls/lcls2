#!/bin/bash
#SBATCH --account=lcls_g
#SBATCH --constraint=gpu
#SBATCH --qos=premium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --signal=TERM@120
#SBATCH --job-name=sim10-stripe-copies
#SBATCH --chdir=/pscratch/sd/m/monarin/psana2-gpu/sim10-striping-20261110/attempt01
#SBATCH --output=/pscratch/sd/m/monarin/psana2-gpu/sim10-striping-20261110/attempt01/slurm-%j.out
#SBATCH --error=/pscratch/sd/m/monarin/psana2-gpu/sim10-striping-20261110/attempt01/slurm-%j.err
set -euo pipefail
set -o noclobber
bundle=$(realpath -e "${1:?immutable source bundle}")
attempt=$(realpath -e "${2:?attempt directory}")
campaign=/pscratch/sd/m/monarin/psana2-gpu/sim10-striping-20261110
case "$attempt" in "$campaign"/*) ;; *) exit 20 ;; esac
[[ "$bundle" == "$attempt/bundle" && "$(pwd -P)" == "$attempt" ]]
# Submit from attempt with --chdir=attempt --output=attempt/slurm-%j.out
# --error=attempt/slurm-%j.err. Fail before evidence writes if reused.
[[ ! -e "$attempt/datasets" ]]
mkdir "$attempt/job-started"
source_data=/pscratch/sd/m/monarin/psana2-gpu/datasets/mfx101555026-r0010-sim10-100k-jf
python_bin=/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python
scontrol show job "$SLURM_JOB_ID" > "$attempt/slurm-job.txt"
uname -a > "$attempt/kernel.txt"
lscpu > "$attempt/cpu.txt"
/usr/bin/lfs quota -u monarin /pscratch > "$attempt/quota-at-start.txt"
df -B1 /pscratch > "$attempt/capacity-at-start.txt"
# A single native task with five bounded file workers; no MPI or GPU operations.
srun --ntasks=1 --cpus-per-task=16 --cpu-bind=cores "$python_bin" -u "$bundle/stage.py" \
  --source "$source_data" --manifest "$bundle/transfer-manifest.json" \
  --output "$attempt/datasets"
