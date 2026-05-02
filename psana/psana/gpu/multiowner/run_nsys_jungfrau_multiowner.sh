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
mps_enabled="false"
mps_gpu_device="${MPS_GPU_DEVICE:-0}"
MPS_STARTED=0
MPS_CLEANUP_DIRS=()
MPS_MONITOR_PID=""
MPS_MONITOR_LOG=""
GPU_MONITOR_PID=""
GPU_MONITOR_LOG=""

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
  --mps                  Start CUDA MPS before profiling and stop it on exit
  --mps-gpu-device N     CUDA_VISIBLE_DEVICES value used for the MPS server (default: ${mps_gpu_device})
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
    --mps)
      mps_enabled="true"
      shift
      ;;
    --mps-gpu-device)
      mps_gpu_device="$2"
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

stop_mps_monitor() {
  if [[ -n "${MPS_MONITOR_PID}" ]]; then
    kill "${MPS_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${MPS_MONITOR_PID}" >/dev/null 2>&1 || true
    MPS_MONITOR_PID=""
  fi
}

stop_gpu_monitor() {
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    GPU_MONITOR_PID=""
  fi
}

stop_mps() {
  stop_mps_monitor
  stop_gpu_monitor
  if [[ "${MPS_STARTED}" == "1" ]]; then
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    MPS_STARTED=0
  fi
  local dir
  for dir in "${MPS_CLEANUP_DIRS[@]}"; do
    rm -rf "${dir}" >/dev/null 2>&1 || true
  done
}

log_mps_server_list() {
  local label="$1"
  local server_list
  echo "MPS_CONTROL_${label}_GET_SERVER_LIST_BEGIN"
  server_list="$(echo get_server_list | nvidia-cuda-mps-control 2>&1 || true)"
  printf '%s\n' "${server_list}"
  if [[ -n "${server_list}" ]]; then
    echo "MPS_SERVER_PIDS_${label}=$(printf '%s\n' "${server_list}" | tr '\n' ' ')"
  else
    echo "MPS_SERVER_PIDS_${label}=none"
  fi
  echo "MPS_CONTROL_${label}_GET_SERVER_LIST_END"
}

start_gpu_monitor() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU_MEMORY_MONITOR=unavailable"
    return
  fi

  GPU_MONITOR_LOG="${log_path}.gpu-mem"
  : > "${GPU_MONITOR_LOG}"
  (
    while true; do
      echo "GPU_MONITOR_TS=$(date -Is)"
      nvidia-smi \
        --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits |
        awk -F, '{
          for (i = 1; i <= NF; i++) {
            gsub(/^[ \t]+|[ \t]+$/, "", $i)
          }
          print "GPU_TOTAL index="$1" used_mib="$2" total_mib="$3" util_pct="$4
        }' || true
      nvidia-smi \
        --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null |
        awk -F, '{
          for (i = 1; i <= NF; i++) {
            gsub(/^[ \t]+|[ \t]+$/, "", $i)
          }
          if (NF >= 3) {
            print "GPU_PROCESS pid="$1" used_mib="$3" name="$2
          }
        }' || true
      sleep 1
    done >> "${GPU_MONITOR_LOG}" 2>&1
  ) &
  GPU_MONITOR_PID=$!
  echo "GPU_MEMORY_MONITOR_LOG=${GPU_MONITOR_LOG}"
  echo "GPU_MEMORY_MONITOR_PID=${GPU_MONITOR_PID}"
}

print_gpu_monitor_summary() {
  if [[ -z "${GPU_MONITOR_LOG}" || ! -f "${GPU_MONITOR_LOG}" ]]; then
    return
  fi

  echo "GPU_MEMORY_MONITOR_SUMMARY_BEGIN"
  awk '
    $1 == "GPU_TOTAL" {
      for (i = 1; i <= NF; i++) {
        split($i, kv, "=")
        if (kv[1] == "used_mib" && kv[2] > max_total) {
          max_total = kv[2]
        }
      }
    }
    $1 == "GPU_PROCESS" {
      pid = ""
      mem = 0
      name = ""
      for (i = 1; i <= NF; i++) {
        split($i, kv, "=")
        if (kv[1] == "pid") {
          pid = kv[2]
        } else if (kv[1] == "used_mib") {
          mem = kv[2]
        } else if (kv[1] == "name") {
          name = kv[2]
        }
      }
      if (pid != "" && mem > max_by_pid[pid]) {
        max_by_pid[pid] = mem
        name_by_pid[pid] = name
      }
    }
    END {
      printf("GPU_MEMORY_MAX_TOTAL_MIB=%d\n", max_total)
      for (pid in max_by_pid) {
        printf("GPU_MEMORY_MAX_PROCESS pid=%s used_mib=%d name=%s\n", pid, max_by_pid[pid], name_by_pid[pid])
      }
    }
  ' "${GPU_MONITOR_LOG}" | sort
  echo "GPU_MEMORY_MONITOR_SUMMARY_END"
}

start_mps_monitor() {
  MPS_MONITOR_LOG="${log_path}.mps-pids"
  : > "${MPS_MONITOR_LOG}"
  (
    while true; do
      date -Is
      pgrep -a 'nvidia-cuda-mps|nvidia-mps' || true
      sleep 1
    done >> "${MPS_MONITOR_LOG}" 2>&1
  ) &
  MPS_MONITOR_PID=$!
  echo "MPS_PID_MONITOR_LOG=${MPS_MONITOR_LOG}"
  echo "MPS_PID_MONITOR_PID=${MPS_MONITOR_PID}"
}

print_mps_monitor_summary() {
  if [[ -n "${MPS_MONITOR_LOG}" && -f "${MPS_MONITOR_LOG}" ]]; then
    echo "MPS_PID_MONITOR_UNIQUE_PROCESSES_BEGIN"
    awk '/nvidia-cuda-mps|nvidia-mps/ {print}' "${MPS_MONITOR_LOG}" | sort -u || true
    echo "MPS_PID_MONITOR_UNIQUE_PROCESSES_END"
  fi
}

trap stop_mps EXIT

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

if [[ "${mps_enabled}" == "true" ]]; then
  if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "error: nvidia-cuda-mps-control not found" >&2
    exit 1
  fi

  export CUDA_VISIBLE_DEVICES="${mps_gpu_device}"
  mps_tag="$(basename "${output_prefix}")_gpu${mps_gpu_device}_$$_${RANDOM}"
  if [[ -z "${CUDA_MPS_PIPE_DIRECTORY:-}" ]]; then
    export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${USER}-${mps_tag}"
    MPS_CLEANUP_DIRS+=("${CUDA_MPS_PIPE_DIRECTORY}")
  fi
  if [[ -z "${CUDA_MPS_LOG_DIRECTORY:-}" ]]; then
    export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-${USER}-${mps_tag}-log"
    MPS_CLEANUP_DIRS+=("${CUDA_MPS_LOG_DIRECTORY}")
  fi
  mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

  echo "Starting CUDA MPS"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY}"
  echo "CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY}"
  nvidia-cuda-mps-control -d
  MPS_STARTED=1
  log_mps_server_list "AFTER_START"
  sleep 2
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
echo "mps_enabled=${mps_enabled}"
if [[ "${mps_enabled}" == "true" ]]; then
  echo "mps_gpu_device=${mps_gpu_device}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-}"
  echo "CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-}"
fi

if [[ "${mps_enabled}" == "true" ]]; then
  start_mps_monitor
fi
start_gpu_monitor

set -x
"${nsys_bin}" "${nsys_args[@]}" "${launcher}" "${launch_args[@]}"
set +x

stop_gpu_monitor
print_gpu_monitor_summary

if [[ "${mps_enabled}" == "true" ]]; then
  stop_mps_monitor
  print_mps_monitor_summary
  log_mps_server_list "POST_PROFILE"
fi

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
