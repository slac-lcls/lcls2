#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"

nsys_bin="${NSYS_BIN:-/sdf/home/m/monarin/tools/nsight-systems/2026.1.1/pkg/bin/nsys}"
output_prefix="${NSYS_OUTPUT_PREFIX:-${PWD}/nsys-jungfrau-multiowner}"
log_path="${NSYS_LOG_PATH:-}"
trace_kinds="${NSYS_TRACE:-cuda,nvtx,osrt,mpi}"
sample_mode="${NSYS_SAMPLE:-none}"
force_overwrite="true"
show_stats="true"
nranks="1"
launcher="python"
python_bin="${PYTHON_BIN:-python}"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [wrapper options] -- [run_jungfrau_multiowner.py args]

Wrapper options:
  -n, --nranks N         Launch under 'mpirun -n N' (default: 1)
  -o, --output PREFIX    Nsight Systems output prefix
  --nsys PATH            Nsight Systems binary path
  --python PATH          Python executable to run inside the activated env
  --trace LIST           Trace domains (default: ${trace_kinds})
  --sample MODE          Sampling mode (default: ${sample_mode})
  --no-force-overwrite   Do not pass --force-overwrite=true to nsys
  --no-stats             Do not run 'nsys stats' after profiling
  -h, --help             Show this help

Examples:
  $(basename "$0") -n 1 -- \\
    -e mfx101344525 -r 125 --xtc-dir /lscratch/mfx/mfx101344525/xtc --max-events 20

  $(basename "$0") -n 4 -o /lscratch/monarin/nsys/jf-multiowner -- \\
    -e mfx101344525 -r 125 --xtc-dir /lscratch/monarin/mfx/mfx101344525/xtc --max-events 100 --gpu-auto
EOF
}

script_args=()
while (($#)); do
  case "$1" in
    -n|--nranks)
      nranks="$2"
      shift 2
      ;;
    -o|--output)
      output_prefix="$2"
      shift 2
      ;;
    --nsys)
      nsys_bin="$2"
      shift 2
      ;;
    --python)
      python_bin="$2"
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
    --no-stats)
      show_stats="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      script_args+=("$@")
      break
      ;;
    *)
      script_args+=("$1")
      shift
      ;;
  esac
done

if [[ ! -x "${nsys_bin}" ]]; then
  echo "error: nsys binary not found or not executable: ${nsys_bin}" >&2
  exit 1
fi

if [[ -z "${log_path}" ]]; then
  log_path="${output_prefix}.log"
fi

mkdir -p "$(dirname "${output_prefix}")"
mkdir -p "$(dirname "${log_path}")"

exec >> "${log_path}" 2>&1

set +u
source "${HOME}/lcls2/setup_env.sh"
set -u

if declare -F activate_psana2_gpu_cupy >/dev/null 2>&1; then
  activate_psana2_gpu_cupy >/dev/null
else
  cuda_shim_root="${CUDA_PATH:-${HOME}/.local/share/psana2-gpu/cuda12shim}"
  if [[ -d "${cuda_shim_root}/lib" ]]; then
    export CUDA_PATH="${cuda_shim_root}"
    export LD_LIBRARY_PATH="${cuda_shim_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  else
    echo "warning: activate_psana2_gpu_cupy is unavailable and no CUDA shim found at ${cuda_shim_root}" >&2
  fi
fi

target_script="${repo_root}/psana/psana/gpu/multiowner/run_jungfrau_multiowner.py"
if [[ ! -f "${target_script}" ]]; then
  echo "error: target script not found: ${target_script}" >&2
  exit 1
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

if [[ "${nranks}" == "1" ]]; then
  launcher="${python_bin}"
  launch_args=("${target_script}" "${script_args[@]}")
else
  launcher="mpirun"
  launch_args=(-n "${nranks}" "${python_bin}" "${target_script}" "${script_args[@]}")
fi

echo "repo_root=${repo_root}"
echo "nsys_bin=${nsys_bin}"
echo "output_prefix=${output_prefix}"
echo "log_path=${log_path}"
echo "nranks=${nranks}"
echo "trace=${trace_kinds}"

set -x
"${nsys_bin}" "${nsys_args[@]}" "${launcher}" "${launch_args[@]}"
set +x

if [[ "${show_stats}" != "true" ]]; then
  exit 0
fi

report_path="${output_prefix}.nsys-rep"
if [[ ! -f "${report_path}" ]]; then
  echo "error: expected report not found: ${report_path}" >&2
  exit 1
fi

echo
echo "=== nsys stats: ownership / contention focused summaries ==="
(
  cd "$(dirname "${report_path}")"
  "${nsys_bin}" stats \
    --force-export=true \
    --report cuda_api_gpu_sum,cuda_gpu_sum,cuda_kern_exec_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,nvtx_pushpop_sum,mpi_event_sum,osrt_sum \
    "${report_path}"
)

echo
echo "Report for desktop Nsight Systems:"
echo "  ${report_path}"
echo "Log:"
echo "  ${log_path}"
