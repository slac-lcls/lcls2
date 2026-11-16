#!/usr/bin/env bash
#SBATCH --account=lcls:data@ampere
#SBATCH --qos=normal
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --mem=128G
#SBATCH --time=00:25:00
#SBATCH --signal=B:TERM@120
set -euo pipefail
source_dir=${1:?source}
output=${2:?output}
child=''
trap 'if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" || true; fi; exit 143' TERM INT
bash "$source_dir/../jf_raw_cache_20261109/runtime.sh" /sdf/scratch/users/m/monarin/gpu-validation/kvikio-2610-validation-20261015-r1/venv/bin/python "$source_dir/run.py" "$output" &
child=$!
wait "$child"
