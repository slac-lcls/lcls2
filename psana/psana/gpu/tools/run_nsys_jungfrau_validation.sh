#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"

nsys_bin="${NSYS_BIN:-/sdf/home/m/monarin/tools/nsight-systems/2026.1.1/pkg/bin/nsys}"
output_prefix="${NSYS_OUTPUT_PREFIX:-${PWD}/nsys-jungfrau}"
trace_kinds="${NSYS_TRACE:-cuda,nvtx,osrt}"
sample_mode="${NSYS_SAMPLE:-none}"
cuda_shim_root="${CUDA_PATH:-${HOME}/.local/share/psana2-gpu/cuda12shim}"
force_overwrite="true"
show_stats="false"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [wrapper options] -- [validate_jungfrau_gpu.py args]

Wrapper options:
  -o, --output PREFIX     Nsight Systems output prefix
  --nsys PATH             Nsight Systems binary path
  --trace LIST            Trace domains (default: ${trace_kinds})
  --sample MODE           Sampling mode (default: ${sample_mode})
  --no-force-overwrite    Do not pass --force-overwrite=true to nsys
  --show-stats            Run 'nsys stats' after profiling and print CUDA memcopy summaries
  -h, --help              Show this help

Examples:
  $(basename "$0") -- \\
    -e mfx101344525 -r 125 --max-events 100 --xtc-dir /lscratch/monarin/mfx/mfx101344525/xtc

  $(basename "$0") -o /lscratch/monarin/nsys/jf500 -- \\
    -e mfx101344525 -r 125 --max-events 500 --xtc-dir /lscratch/monarin/mfx/mfx101344525/xtc
EOF
}

validator_args=()
while (($#)); do
  case "$1" in
    -o|--output)
      output_prefix="$2"
      shift 2
      ;;
    --nsys)
      nsys_bin="$2"
      shift 2
      ;;
    --trace)
      trace_kinds="$2"
      shift 2
      ;;
    --sample)
      sample_mode="$2"
      shift 2
      ;;
    --no-force-overwrite)
      force_overwrite="false"
      shift
      ;;
    --show-stats)
      show_stats="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      validator_args+=("$@")
      break
      ;;
    *)
      validator_args+=("$1")
      shift
      ;;
  esac
done

if [[ ! -x "${nsys_bin}" ]]; then
  echo "error: nsys binary not found or not executable: ${nsys_bin}" >&2
  exit 1
fi

mkdir -p "$(dirname "${output_prefix}")"

# setup_env.sh references optional variables before setting them.
set +u
source "${repo_root}/setup_env.sh"
set -u

export PYTHONPATH="${repo_root}/psana:${repo_root}/psdaq${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -d "${cuda_shim_root}/lib" ]]; then
  export CUDA_PATH="${cuda_shim_root}"
  export LD_LIBRARY_PATH="${cuda_shim_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

have_gpu_profile_arg="false"
for arg in "${validator_args[@]}"; do
  if [[ "${arg}" == "--gpu-profile" ]]; then
    have_gpu_profile_arg="true"
    break
  fi
done
if [[ "${have_gpu_profile_arg}" == "false" ]]; then
  validator_args+=(--gpu-profile off)
fi

nsys_args=(
  profile
  --sample="${sample_mode}"
  --trace="${trace_kinds}"
  --cuda-memory-usage=true
  -o "${output_prefix}"
)
if [[ "${force_overwrite}" == "true" ]]; then
  nsys_args+=(--force-overwrite=true)
fi

set -x
"${nsys_bin}" "${nsys_args[@]}" \
  python "${repo_root}/psana/psana/gpu/tools/validate_jungfrau_gpu.py" \
  "${validator_args[@]}"

if [[ "${show_stats}" == "true" ]]; then
  report_path="${output_prefix}.nsys-rep"
  if [[ ! -f "${report_path}" ]]; then
    echo "error: expected report not found: ${report_path}" >&2
    exit 1
  fi
  echo
  echo "=== nsys stats: CUDA memcopy summaries ==="
  (
    cd "$(dirname "${report_path}")"
    "${nsys_bin}" stats \
      --report cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
      "${report_path}"
  )
fi
