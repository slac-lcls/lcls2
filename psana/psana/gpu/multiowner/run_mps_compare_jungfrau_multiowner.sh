#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"

events=16000
extra_gpu_work=0
gpu_device=0
output_dir="${PWD}/mps_compare_jungfrau_multiowner"
exp="mfx101344525"
run="125"
xtc_dir=""
skip_calib_load=""
rank_list=(1 2 4 8 16)
python_bin="${PYTHON_BIN:-python}"
warmup="true"
modes="nomps,mps"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options]

Options:
  -o, --output-dir DIR       Output directory (default: ${output_dir})
  -e, --exp EXP              Experiment (default: ${exp})
  -r, --run RUN              Run number (default: ${run})
  --xtc-dir DIR              Pass --xtc-dir to run_jungfrau_multiowner.py
  --events N                 Fixed max events for every run (default: ${events})
  --extra-gpu-work N         Extra GPU work iterations (default: ${extra_gpu_work})
  --gpu-device N             CUDA device used by all BD ranks (default: ${gpu_device})
  --skip-calib-load VALUE    Pass --skip-calib-load to run_jungfrau_multiowner.py
  --ranks LIST               Comma-separated BD counts (default: 1,2,4,8,16)
  --python PATH              Python executable (default: ${python_bin})
  --no-warmup                Skip the 16-BD cache warm-up run
  --modes LIST               Comma-separated modes: nomps,mps (default: ${modes})
  -h, --help                 Show this help
EOF
}

parse_rank_list() {
  local raw="$1"
  IFS=',' read -r -a rank_list <<< "${raw}"
}

while (($#)); do
  case "$1" in
    -o|--output-dir)
      output_dir="$2"
      shift 2
      ;;
    -e|--exp)
      exp="$2"
      shift 2
      ;;
    -r|--run)
      run="$2"
      shift 2
      ;;
    --xtc-dir)
      xtc_dir="$2"
      shift 2
      ;;
    --events)
      events="$2"
      shift 2
      ;;
    --extra-gpu-work)
      extra_gpu_work="$2"
      shift 2
      ;;
    --gpu-device)
      gpu_device="$2"
      shift 2
      ;;
    --skip-calib-load)
      skip_calib_load="$2"
      shift 2
      ;;
    --ranks)
      parse_rank_list "$2"
      shift 2
      ;;
    --python)
      python_bin="$2"
      shift 2
      ;;
    --no-warmup)
      warmup="false"
      shift
      ;;
    --modes)
      modes="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

target_script="${repo_root}/psana/psana/gpu/multiowner/run_jungfrau_multiowner.py"
summary_script="${output_dir}/summarize_mps_compare.py"

mkdir -p "${output_dir}"

set +u
source "${HOME}/lcls2/setup_env.sh"
set -u

if declare -F activate_psana2_gpu_cupy >/dev/null 2>&1; then
  activate_psana2_gpu_cupy >/dev/null
fi

cuda_shim_root="${CUDA_PATH:-${HOME}/.local/share/psana2-gpu/cuda12shim}"
if [[ -d "${cuda_shim_root}/lib" ]]; then
  export CUDA_PATH="${cuda_shim_root}"
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${cuda_shim_root}/lib:"*) ;;
    *) export LD_LIBRARY_PATH="${cuda_shim_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
fi

if [[ ! -f "${target_script}" ]]; then
  echo "error: target script not found: ${target_script}" >&2
  exit 1
fi

if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "error: nvidia-cuda-mps-control not found" >&2
  exit 1
fi

common_args=(
  -e "${exp}"
  -r "${run}"
  --max-events "${events}"
  --gpu-device "${gpu_device}"
  --print-interval 1000
  --extra-gpu-work "${extra_gpu_work}"
)
if [[ -n "${xtc_dir}" ]]; then
  common_args+=(--xtc-dir "${xtc_dir}")
fi
if [[ -n "${skip_calib_load}" ]]; then
  common_args+=(--skip-calib-load "${skip_calib_load}")
fi

stop_monitor() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]]; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  fi
}

stop_mps() {
  if [[ "${MPS_STARTED:-0}" == "1" ]]; then
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    MPS_STARTED=0
  fi
}

trap 'stop_monitor "${MON_PID:-}"; stop_mps' EXIT

run_case() {
  local mode="$1"
  local n_bds="$2"
  local nranks=$((n_bds + 2))
  local prefix="${output_dir}/${mode}_${n_bds}bd_1gpu_extra${extra_gpu_work}_${events}ev"
  local log="${prefix}.log"
  local gpu_csv="${prefix}.gpu.csv"

  echo "RUN mode=${mode} n_bds=${n_bds} nranks=${nranks} events=${events} extra=${extra_gpu_work} start=$(date +%H:%M:%S)"
  {
    echo "mode=${mode}"
    echo "n_bds=${n_bds}"
    echo "nranks=${nranks}"
    echo "events=${events}"
    echo "extra_gpu_work=${extra_gpu_work}"
    echo "gpu_device=${gpu_device}"
    echo "hostname=$(hostname)"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "CUDA_PATH=${CUDA_PATH:-}"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    echo "CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-}"
    echo "CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-}"
    echo "MPS_ENV_FILE=${MPS_ENV_FILE:-}"
    if [[ "${mode}" == "mps" ]]; then
      echo "MPS_EXPECTED=1"
      echo "MPS_CONTROL_GET_SERVER_LIST_BEGIN"
      echo get_server_list | nvidia-cuda-mps-control || true
      echo "MPS_CONTROL_GET_SERVER_LIST_END"
    else
      echo "MPS_EXPECTED=0"
    fi
    nvidia-smi --query-gpu=index,compute_mode,mig.mode.current,memory.used,memory.total --format=csv || true
  } > "${log}"

  (
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
        --format=csv,noheader,nounits
      sleep 1
    done > "${gpu_csv}"
  ) &
  MON_PID=$!

  mpirun -n "${nranks}" "${python_bin}" "${target_script}" \
    "${common_args[@]}" \
    >> "${log}" 2>&1
  local status=$?

  if [[ "${mode}" == "mps" ]]; then
    {
      echo "MPS_CONTROL_POST_RUN_GET_SERVER_LIST_BEGIN"
      echo get_server_list | nvidia-cuda-mps-control || true
      echo "MPS_CONTROL_POST_RUN_GET_SERVER_LIST_END"
    } >> "${log}" 2>&1
  fi

  stop_monitor "${MON_PID}"
  MON_PID=""
  if [[ "${status}" -ne 0 ]]; then
    echo "FAILED mode=${mode} n_bds=${n_bds} status=${status} log=${log}" >&2
    return "${status}"
  fi
  echo "DONE mode=${mode} n_bds=${n_bds} end=$(date +%H:%M:%S)"
}

run_nomps="false"
run_mps="false"
case ",${modes}," in
  *,nomps,*) run_nomps="true" ;;
esac
case ",${modes}," in
  *,mps,*) run_mps="true" ;;
esac
if [[ "${run_nomps}" != "true" && "${run_mps}" != "true" ]]; then
  echo "error: --modes must include nomps, mps, or both" >&2
  exit 2
fi

if [[ "${warmup}" == "true" ]]; then
  warmup_log="${output_dir}/warmup_16bd_1gpu_extra${extra_gpu_work}_${events}ev.log"
  echo "WARMUP n_bds=16 events=${events} extra=${extra_gpu_work} start=$(date +%H:%M:%S)"
  mpirun -n 18 "${python_bin}" "${target_script}" \
    "${common_args[@]}" \
    > "${warmup_log}" 2>&1 || {
      echo "warning: warmup failed; continuing. log=${warmup_log}" >&2
    }
  echo "WARMUP done end=$(date +%H:%M:%S)"
fi

MPS_STARTED=0
if [[ "${run_nomps}" == "true" ]]; then
  for n_bds in "${rank_list[@]}"; do
    run_case "nomps" "${n_bds}"
  done
fi

if [[ "${run_mps}" == "true" ]]; then
  export CUDA_VISIBLE_DEVICES="${gpu_device}"
  mps_tag="$(basename "${output_dir}")_gpu${gpu_device}_$$_${RANDOM}"
  export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${USER}-${mps_tag}"
  export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-${USER}-${mps_tag}-log"
  export MPS_ENV_FILE="${output_dir}/mps_env.sh"
  mps_start_log="${output_dir}/mps_start.log"
  rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
  mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

  {
    echo "export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'"
    echo "export CUDA_MPS_PIPE_DIRECTORY='${CUDA_MPS_PIPE_DIRECTORY}'"
    echo "export CUDA_MPS_LOG_DIRECTORY='${CUDA_MPS_LOG_DIRECTORY}'"
  } > "${MPS_ENV_FILE}"

  {
    echo "Starting MPS with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "MPS pipe=${CUDA_MPS_PIPE_DIRECTORY}"
    echo "MPS log=${CUDA_MPS_LOG_DIRECTORY}"
    echo "MPS env file=${MPS_ENV_FILE}"
    nvidia-cuda-mps-control -d
    echo "MPS_CONTROL_AFTER_START_GET_SERVER_LIST_BEGIN"
    echo get_server_list | nvidia-cuda-mps-control || true
    echo "MPS_CONTROL_AFTER_START_GET_SERVER_LIST_END"
  } 2>&1 | tee "${mps_start_log}"
  MPS_STARTED=1
  sleep 2

  for n_bds in "${rank_list[@]}"; do
    run_case "mps" "${n_bds}"
  done

  stop_mps
fi

cat > "${summary_script}" <<'PY'
from pathlib import Path
import csv
import re
import statistics

out_dir = Path(__file__).resolve().parent
rows = []

def quantile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    return values[int(round((len(values) - 1) * q))]

for log in sorted(out_dir.glob("*_*bd_1gpu_extra*ev.log")):
    if log.name.startswith("warmup_"):
        continue
    m = re.match(r"(nomps|mps)_(\d+)bd_1gpu_extra(\d+)_(\d+)ev\.log", log.name)
    if not m:
        continue
    mode = m.group(1)
    n_bds = int(m.group(2))
    extra = int(m.group(3))
    events = int(m.group(4))
    text = log.read_text(errors="replace")
    g = re.search(
        r"all_ranks_summary total_events=(\d+) elapsed_s=([0-9.]+) total_rate_evt_s=([0-9.]+) bd_ranks=(\d+) world_size=(\d+)",
        text,
    )
    if not g:
        continue
    rank_summaries = list(
        re.finditer(
            r"summary events=(\d+) loop_s=([0-9.]+) rate_evt_s=([0-9.]+).*?gpu_device=([0-9]+).*?"
            r"extra_gpu_work=(\d+) ccons_upload_s=([0-9.]+) copy_total_s=([0-9.]+) "
            r"kernel_total_s=([0-9.]+) extra_total_s=([0-9.]+).*?copied_mib=([0-9.]+)",
            text,
        )
    )
    total_events = int(g.group(1))
    elapsed_s = float(g.group(2))
    rate_hz = float(g.group(3))
    copy_total_s = sum(float(r.group(7)) for r in rank_summaries)
    kernel_total_s = sum(float(r.group(8)) for r in rank_summaries)
    extra_total_s = sum(float(r.group(9)) for r in rank_summaries)
    ev_counts = [int(r.group(1)) for r in rank_summaries]

    gpu_samples = []
    gpu_csv = log.with_suffix(".gpu.csv")
    if gpu_csv.exists():
        with gpu_csv.open() as f:
            for rec in csv.reader(f):
                if len(rec) < 6:
                    continue
                try:
                    idx = int(rec[1].strip())
                    gpu_util = float(rec[2].strip())
                    mem_util = float(rec[3].strip())
                    mem_used = float(rec[4].strip())
                    mem_total = float(rec[5].strip())
                except ValueError:
                    continue
                if idx == 0 and mem_used > 100:
                    gpu_samples.append((gpu_util, mem_util, mem_used, mem_total))

    gpu_util = [s[0] for s in gpu_samples]
    mem_util = [s[1] for s in gpu_samples]
    mem_used = [s[2] for s in gpu_samples]
    rows.append(
        {
            "mode": mode,
            "n_bds": n_bds,
            "extra_gpu_work": extra,
            "events_requested": events,
            "total_events": total_events,
            "elapsed_s": elapsed_s,
            "rate_hz": rate_hz,
            "avg_gpu_util_pct": statistics.mean(gpu_util) if gpu_util else 0.0,
            "p90_gpu_util_pct": quantile(gpu_util, 0.90),
            "max_gpu_util_pct": max(gpu_util) if gpu_util else 0.0,
            "avg_gpu_mem_util_pct": statistics.mean(mem_util) if mem_util else 0.0,
            "p90_gpu_mem_util_pct": quantile(mem_util, 0.90),
            "max_gpu_mem_util_pct": max(mem_util) if mem_util else 0.0,
            "max_mem_used_mib": max(mem_used) if mem_used else 0.0,
            "avg_mem_used_mib": statistics.mean(mem_used) if mem_used else 0.0,
            "copy_total_s": copy_total_s,
            "kernel_total_s": kernel_total_s,
            "extra_total_s": extra_total_s,
            "min_rank_events": min(ev_counts) if ev_counts else 0,
            "max_rank_events": max(ev_counts) if ev_counts else 0,
            "gpu_samples": len(gpu_samples),
        }
    )

base_rate = {}
for row in rows:
    if row["n_bds"] == 1:
        base_rate[row["mode"]] = row["rate_hz"]

for row in rows:
    baseline_rate = base_rate.get(row["mode"], 0.0)
    row["speedup_vs_1bd"] = row["rate_hz"] / baseline_rate if baseline_rate else 0.0
    row["efficiency_pct"] = row["speedup_vs_1bd"] / row["n_bds"] * 100.0

rows.sort(key=lambda r: (r["mode"], r["extra_gpu_work"], r["n_bds"]))
fields = [
    "mode",
    "n_bds",
    "extra_gpu_work",
    "events_requested",
    "total_events",
    "elapsed_s",
    "rate_hz",
    "speedup_vs_1bd",
    "efficiency_pct",
    "avg_gpu_util_pct",
    "p90_gpu_util_pct",
    "max_gpu_util_pct",
    "avg_gpu_mem_util_pct",
    "p90_gpu_mem_util_pct",
    "max_gpu_mem_util_pct",
    "max_mem_used_mib",
    "avg_mem_used_mib",
    "copy_total_s",
    "kernel_total_s",
    "extra_total_s",
    "min_rank_events",
    "max_rank_events",
    "gpu_samples",
]
summary = out_dir / "summary.csv"
with summary.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: (f"{row[key]:.3f}" if isinstance(row[key], float) else row[key])
                for key in fields
            }
        )

print(summary)
print(
    "mode n_bds events elapsed_s rate_hz speedup eff_pct avg_gpu p90_gpu "
    "avg_memutil p90_memutil max_mem_mib"
)
for row in rows:
    print(
        f"{row['mode']} {row['n_bds']} {row['total_events']} "
        f"{row['elapsed_s']:.3f} {row['rate_hz']:.3f} "
        f"{row['speedup_vs_1bd']:.2f} {row['efficiency_pct']:.1f} "
        f"{row['avg_gpu_util_pct']:.1f} {row['p90_gpu_util_pct']:.1f} "
        f"{row['avg_gpu_mem_util_pct']:.1f} {row['p90_gpu_mem_util_pct']:.1f} "
        f"{row['max_mem_used_mib']:.0f}"
    )
PY

"${python_bin}" "${summary_script}"

echo "Summary:"
echo "  ${output_dir}/summary.csv"
