#!/usr/bin/env bash
#SBATCH --account=lcls:data@ampere
#SBATCH --qos=normal
#SBATCH --partition=ampere
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
set -euo pipefail
source_dir=${1:?absolute source directory}
output=${2:?new project scratch directory}
umask 0007
node_root=${SLURM_TMPDIR:-/lscratch/$USER}
local_root="$node_root/mfx-512k-${SLURM_JOB_ID}"
mkdir -p "$node_root"
mkdir -m 700 "$local_root"
mkdir -m 770 "$output"
export TMPDIR="$local_root/tmp"
export CUPY_CACHE_DIR="$local_root/cupy-cache"
mkdir "$TMPDIR" "$CUPY_CACHE_DIR"
prep="$source_dir/../mfx_expand10_performance_20261106"
runtime="$source_dir/../mfx_expand10_stage2_20261105/runtime.sh"
pilot=/sdf/scratch/users/m/monarin/gpu-validation/mfx-expand10/20261105-stage2-r1/pilot
nsys=/sdf/group/lcls/ds/tools/nsight-2025.3.1/bin/nsys
"$nsys" --version > "$output/nsys-version.txt"
sha256sum "$prep/runner.py" "$source_dir/run.sbatch" > "$output/launch-sha256.txt"
bash "$runtime" -m unittest discover -s "$prep" -p test_preparation.py -v
bash "$runtime" "$prep/device_checks.py" "$output/device-checks.json"
for repeat in 1 2 3; do
    for streams in 1 2; do
        bash "$runtime" "$prep/runner.py" --root "$pilot" --output "$output/clean-s${streams}-r${repeat}" \
            --streams "$streams" --warmup 100 --bins 100
    done
done
bash "$runtime" "$prep/runner.py" --root "$pilot" --output "$output/control-8m" \
    --streams 2 --warmup 100 --bins 100 --bulk-target-bytes 8388608
for target in 8388608 524288; do
    "$nsys" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
        --output="$output/trace-$target" \
        bash "$runtime" "$prep/runner.py" --root "$pilot" --output "$output/traced-$target" \
        --streams 2 --warmup 100 --bins 100 --bulk-target-bytes "$target"
    "$nsys" export --type=sqlite --output="$output/trace-$target.sqlite" "$output/trace-$target.nsys-rep"
done
