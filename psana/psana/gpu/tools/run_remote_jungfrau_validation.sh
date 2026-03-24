#!/usr/bin/env bash
set -eo pipefail

repo_root="/cds/home/m/monarin/lcls2"
shim_dir="/tmp/monarin-cuda12shim"

mkdir -p "${shim_dir}"
ln -sf /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.13 "${shim_dir}/libnvrtc.so.12"
ln -sf /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc-builtins.so.13.2 "${shim_dir}/libnvrtc-builtins.so.12.0"

# Conda activation hooks used by setup_env.sh reference optional variables
# before defining them, so delay nounset until after activation.
set +u
source "${repo_root}/setup_env.sh"
set -u

export LD_LIBRARY_PATH="${shim_dir}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${repo_root}/psana:${repo_root}/psdaq${PYTHONPATH:+:${PYTHONPATH}}"

python "${repo_root}/psana/psana/gpu/tools/validate_jungfrau_gpu.py" "$@"
