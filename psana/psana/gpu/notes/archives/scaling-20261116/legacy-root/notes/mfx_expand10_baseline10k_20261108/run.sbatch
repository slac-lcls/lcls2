#!/usr/bin/env bash
#SBATCH --account=lcls:data@ampere
#SBATCH --qos=normal
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
set -euo pipefail
source_dir=${1:?absolute source directory}
output=${2:?new project scratch directory}
umask 0007
node_root=${SLURM_TMPDIR:-/lscratch/$USER}
local_root="$node_root/mfx-baseline10k-${SLURM_JOB_ID}"
mkdir -p "$node_root"
mkdir -m 700 "$local_root"
mkdir -m 770 "$output"
export TMPDIR="$local_root/tmp" CUPY_CACHE_DIR="$local_root/cupy-cache"
mkdir "$TMPDIR" "$CUPY_CACHE_DIR"
prep="$source_dir/../mfx_expand10_performance_20261106"
runtime="$source_dir/../mfx_expand10_stage2_20261105/runtime.sh"
fixture=/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sample10_10k
sha256sum "$source_dir/run.sbatch" "$prep"/*.py "$runtime" > "$output/launch-sha256.txt"
# Static snapshots outside measured iterations, not continuous utilization sampling.
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu --format=csv > "$output/gpu-before.csv"
for repeat in 1 2 3; do
    for streams in 1 2; do
        bash "$runtime" "$prep/runner.py" --root "$fixture" --output "$output/clean-s${streams}-r${repeat}" \
            --mode throughput --streams "$streams" --warmup 1000 --bins 100 \
            --batch-size 7 --framework-budget-gib 4 --bulk-target-bytes 524288
    done
done
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu --format=csv > "$output/gpu-after.csv"
