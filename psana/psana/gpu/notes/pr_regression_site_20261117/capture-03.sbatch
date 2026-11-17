#!/bin/bash
#SBATCH --account=lcls_g
#SBATCH --qos=premium
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --job-name=site-placement
#SBATCH --output=/pscratch/sd/m/monarin/psana2-gpu/pr-regression-site-20261117/capture-%j.log
set -euo pipefail
root=/pscratch/sd/m/monarin/psana2-gpu/pr-regression-site-20261117
record_exit() {
  capture_status=$?
  sacct -j "$SLURM_JOB_ID" --format=JobID,State,ExitCode,Elapsed,NodeList,AllocTRES,ReqTRES,Account,QOS,Timelimit -P > "$root/capture-$SLURM_JOB_ID-accounting.txt" || true
  exit "$capture_status"
}
trap record_exit EXIT
module load gcc-native/13.2 cudatoolkit/12.9
export TMPDIR=/pscratch/sd/m/monarin/codex-tmp PYTHONDONTWRITEBYTECODE=1 MPICH_GPU_SUPPORT_ENABLED=0
unset PYTHONPATH PYTHONHOME LD_PRELOAD
cd "$root/frozen-03"
sha256sum --check SHA256SUMS
scontrol show job -d "$SLURM_JOB_ID" > "$root/capture-$SLURM_JOB_ID-allocation.txt"
ls -l /sys/class/net > "$root/capture-$SLURM_JOB_ID-net.txt"
/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python -B -s prepare_site.py \
 --runtime /pscratch/sd/m/monarin/psana2-gpu/pr-regression-runtimes-20261117/runtime-01.json \
 --runtime-sha256 6082ff8edb359bd8596cddb6e7bb070b09ddb641ec8800a26a1ace36283a5b35 \
 --output "$root/site-$SLURM_JOB_ID.json" \
 --cores-per-rank 4 --step-timeout 90 --controller-timeout 900
