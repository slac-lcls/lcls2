#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"

nsys_bin="${NSYS_BIN:-$(command -v nsys 2>/dev/null || echo /sdf/home/m/monarin/tools/nsight-systems/2026.1.1/pkg/bin/nsys)}"
scratch_root="${PSANA_GPU_SCRATCH:-${SCRATCH:-/pscratch/sd/m/${USER}}/psana2-gpu}"
output_dir="${NSYS_OUTPUT_DIR:-${scratch_root}/jungfrau_queue_compare/nsys}"
trace_kinds="${NSYS_TRACE:-cuda,nvtx,osrt,mpi}"
sample_mode="${NSYS_SAMPLE:-none}"
force_overwrite="true"
show_stats="true"
export_sqlite="true"

exp="mfx101344525"
run="125"
input_file=""
xtc_dir=""
events=16000
gpu_device=0
queue_mode="one-side"
bd_local_depth=2
gpu_leases=2
lease_sleep_us=50
extra_gpu_work=0
compare_cpu_events=0
skip_calib_load=""
rank_list=(1 2 4 8 16)
python_bin="${PYTHON_BIN:-python}"
print_interval=1000
log_level="INFO"
mps_enabled="false"
mps_gpu_device="${MPS_GPU_DEVICE:-0}"
MPS_STARTED=0
MPS_CLEANUP_DIRS=()

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options]

Runs run_jungfrau_queue_compare.py under Nsight Systems for one or more BD
rank counts. Each case uses nranks = BDs + 2 to account for psana SMD/EB ranks.

Options:
  -o, --output-dir DIR       Output directory (default: ${output_dir})
  --nsys PATH                Nsight Systems binary (default: ${nsys_bin})
  --python PATH              Python executable inside the psana env (default: ${python_bin})
  -e, --exp EXP              Experiment for DataSource(exp=..., run=...) (default: ${exp})
  -r, --run RUN              Run number (default: ${run})
  --file FILE                Input xtc2 file; mutually exclusive with --exp/--run
  --xtc-dir DIR              Pass --xtc-dir to run_jungfrau_queue_compare.py
  --events N                 Fixed max_events for every run (default: ${events})
  --gpu-device N             CUDA device used by all BD ranks (default: ${gpu_device})
  --queue-mode MODE          baseline, one-side, or both-sides-lease (default: ${queue_mode})
  --bd-local-depth N         Per-BD async queue depth (default: ${bd_local_depth})
  --gpu-leases N             Node-local leases for both-sides-lease (default: ${gpu_leases})
  --lease-sleep-us N         Lease wait sleep interval (default: ${lease_sleep_us})
  --extra-gpu-work N         Extra GPU work iterations (default: ${extra_gpu_work})
  --compare-cpu-events N     Compare first N completed BD events (default: ${compare_cpu_events})
  --skip-calib-load VALUE    Pass --skip-calib-load to the queue script
  --ranks LIST               Comma-separated BD counts (default: 1,2,4,8,16)
  --trace LIST               Nsight trace domains (default: ${trace_kinds})
  --sample MODE              Nsight sampling mode (default: ${sample_mode})
  --print-interval N         Queue script print interval (default: ${print_interval})
  --log-level LEVEL          DataSource log level (default: ${log_level})
  --mps                      Start CUDA MPS before running cases
  --mps-gpu-device N         CUDA_VISIBLE_DEVICES value for MPS (default: ${mps_gpu_device})
  --no-force-overwrite       Do not pass --force-overwrite=true to nsys
  --no-stats                 Skip nsys stats reports
  --no-sqlite                Skip nsys SQLite export and timeline metrics
  -h, --help                 Show this help

Output:
  Per case:
    <prefix>.log
    <prefix>.nsys-rep
    <prefix>.stats.log
    <prefix>.sqlite
  Summary:
    ${output_dir}/summary.csv

Example:
  $(basename "$0") --ranks 1,2,4 --events 4000 --queue-mode both-sides-lease \\
    --gpu-leases 2 -e mfx101344525 -r 125
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
    --nsys)
      nsys_bin="$2"
      shift 2
      ;;
    --python)
      python_bin="$2"
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
    --file)
      input_file="$2"
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
    --gpu-device)
      gpu_device="$2"
      shift 2
      ;;
    --queue-mode)
      queue_mode="$2"
      shift 2
      ;;
    --bd-local-depth)
      bd_local_depth="$2"
      shift 2
      ;;
    --gpu-leases)
      gpu_leases="$2"
      shift 2
      ;;
    --lease-sleep-us)
      lease_sleep_us="$2"
      shift 2
      ;;
    --extra-gpu-work)
      extra_gpu_work="$2"
      shift 2
      ;;
    --compare-cpu-events)
      compare_cpu_events="$2"
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
    --trace)
      trace_kinds="$2"
      shift 2
      ;;
    --sample)
      sample_mode="$2"
      shift 2
      ;;
    --print-interval)
      print_interval="$2"
      shift 2
      ;;
    --log-level)
      log_level="$2"
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
    --no-sqlite)
      export_sqlite="false"
      shift
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

target_script="${repo_root}/psana/psana/gpu/multiowner/run_jungfrau_queue_compare.py"
summary_script="${output_dir}/summarize_nsys_queue_compare.py"

if [[ -n "${input_file}" && ( -n "${exp}" || -n "${run}" ) ]]; then
  if [[ "${exp}" != "mfx101344525" || "${run}" != "125" ]]; then
    echo "error: --file is mutually exclusive with explicit --exp/--run" >&2
    exit 2
  fi
fi

mkdir -p "${output_dir}"

if [[ -d "/cds/sw/" || -d "/sdf/group/lcls/" ]]; then
  set +u
  source "${HOME}/lcls2/setup_env.sh"
  set -u

  if declare -F activate_psana2_gpu_cupy >/dev/null 2>&1; then
    activate_psana2_gpu_cupy >/dev/null
  fi
elif [[ -f "/global/homes/m/monarin/psana-nersc/activate_psana_build_env.sh" ]]; then
  set +u
  source "/global/homes/m/monarin/psana-nersc/activate_psana_build_env.sh" >/dev/null
  if declare -F activate_psana >/dev/null 2>&1; then
    activate_psana >/dev/null
  fi
  set -u
else
  echo "error: no known psana environment setup found" >&2
  exit 1
fi

pyver="$("${python_bin}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
install_site="${repo_root}/install_psana/lib/python${pyver}/site-packages"
if [[ -d "${install_site}" ]]; then
  export PYTHONPATH="${install_site}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}${repo_root}/psana"

cuda_shim_root="${CUDA_PATH:-${HOME}/.local/share/psana2-gpu/cuda12shim}"
if [[ -d "${cuda_shim_root}/lib" ]]; then
  export CUDA_PATH="${cuda_shim_root}"
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${cuda_shim_root}/lib:"*) ;;
    *) export LD_LIBRARY_PATH="${cuda_shim_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
fi

if [[ ! -x "${nsys_bin}" ]]; then
  echo "error: nsys binary not found or not executable: ${nsys_bin}" >&2
  exit 1
fi
if [[ ! -f "${target_script}" ]]; then
  echo "error: target script not found: ${target_script}" >&2
  exit 1
fi

stop_mps() {
  if [[ "${MPS_STARTED}" == "1" ]]; then
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    MPS_STARTED=0
  fi
  local dir
  for dir in "${MPS_CLEANUP_DIRS[@]}"; do
    rm -rf "${dir}" >/dev/null 2>&1 || true
  done
}
trap stop_mps EXIT

if [[ "${mps_enabled}" == "true" ]]; then
  if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "error: nvidia-cuda-mps-control not found" >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="${mps_gpu_device}"
  mps_tag="$(basename "${output_dir}")_gpu${mps_gpu_device}_$$_${RANDOM}"
  export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps-${USER}-${mps_tag}}"
  export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-${USER}-${mps_tag}-log}"
  MPS_CLEANUP_DIRS+=("${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}")
  rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
  mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
  nvidia-cuda-mps-control -d
  MPS_STARTED=1
  sleep 2
fi

common_args=(--queue-mode "${queue_mode}")
if [[ -n "${input_file}" ]]; then
  common_args+=(--file "${input_file}")
else
  common_args+=(-e "${exp}" -r "${run}")
fi
if [[ -n "${xtc_dir}" ]]; then
  common_args+=(--xtc-dir "${xtc_dir}")
fi
if [[ -n "${skip_calib_load}" ]]; then
  common_args+=(--skip-calib-load "${skip_calib_load}")
fi
common_args+=(
  --max-events "${events}"
  --gpu-device "${gpu_device}"
  --bd-local-depth "${bd_local_depth}"
  --gpu-leases "${gpu_leases}"
  --lease-sleep-us "${lease_sleep_us}"
  --extra-gpu-work "${extra_gpu_work}"
  --compare-cpu-events "${compare_cpu_events}"
  --print-interval "${print_interval}"
  --log-level "${log_level}"
)

run_case() {
  local n_bds="$1"
  local nranks=$((n_bds + 2))
  local prefix="${output_dir}/${queue_mode}_${n_bds}bd_depth${bd_local_depth}_leases${gpu_leases}_extra${extra_gpu_work}_${events}ev"
  local log="${prefix}.log"
  local stats_log="${prefix}.stats.log"
  local sqlite_path="${prefix}.sqlite"
  local report_path="${prefix}.nsys-rep"

  echo "RUN queue_mode=${queue_mode} bds=${n_bds} nranks=${nranks} events=${events} start=$(date +%H:%M:%S)"
  {
    echo "queue_mode=${queue_mode}"
    echo "bds=${n_bds}"
    echo "nranks=${nranks}"
    echo "events=${events}"
    echo "gpu_device=${gpu_device}"
    echo "bd_local_depth=${bd_local_depth}"
    echo "gpu_leases=${gpu_leases}"
    echo "extra_gpu_work=${extra_gpu_work}"
    echo "hostname=$(hostname)"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "CUDA_PATH=${CUDA_PATH:-}"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    echo "CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-}"
    echo "CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-}"
  } > "${log}"

  local nsys_args=(
    profile
    --sample="${sample_mode}"
    --trace="${trace_kinds}"
    --cuda-memory-usage=true
    -o "${prefix}"
  )
  if [[ "${force_overwrite}" == "true" ]]; then
    nsys_args+=(--force-overwrite=true)
  fi

  if command -v mpirun >/dev/null 2>&1; then
    set +e
    "${nsys_bin}" "${nsys_args[@]}" \
      mpirun -n "${nranks}" "${python_bin}" "${target_script}" "${common_args[@]}" \
      >> "${log}" 2>&1
    local launch_status=$?
    set -e
  else
    local per_rank_nsys_args=(
      profile
      --sample="${sample_mode}"
      --trace="${trace_kinds}"
      --cuda-memory-usage=true
      -o "${prefix}_rank%q{SLURM_PROCID}"
    )
    if [[ "${force_overwrite}" == "true" ]]; then
      per_rank_nsys_args+=(--force-overwrite=true)
    fi
    set +e
    srun \
      --overlap \
      --nodes=1 \
      --ntasks="${nranks}" \
      --gres=gpu:1 \
      --gpu-bind=none \
      "${nsys_bin}" "${per_rank_nsys_args[@]}" \
      "${python_bin}" "${target_script}" "${common_args[@]}" \
      >> "${log}" 2>&1
    local launch_status=$?
    set -e
  fi
  if [[ "${launch_status}" -ne 0 ]]; then
    echo "warning: profiled launch exited with status ${launch_status}; continuing if reports were generated" | tee -a "${log}" >&2
  fi

  shopt -s nullglob
  local report_paths=("${prefix}".nsys-rep "${prefix}"_rank*.nsys-rep)
  shopt -u nullglob
  if ((${#report_paths[@]} == 0)); then
    echo "warning: expected report not found: ${report_path} or ${prefix}_rank*.nsys-rep" | tee -a "${log}" >&2
    return 0
  fi

  if [[ "${show_stats}" == "true" ]]; then
    {
      echo "=== nsys stats: queue compare summaries ==="
      for report in "${report_paths[@]}"; do
        echo "=== report: ${report} ==="
        "${nsys_bin}" stats \
          --force-export=true \
          --report cuda_api_gpu_sum,cuda_gpu_sum,cuda_kern_exec_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,nvtx_pushpop_sum,mpi_event_sum,osrt_sum \
          "${report}"
      done
    } > "${stats_log}" 2>&1 || {
      echo "warning: nsys stats failed for one or more reports; see ${stats_log}" | tee -a "${log}" >&2
    }
  fi

  if [[ "${export_sqlite}" == "true" ]]; then
    rm -f "${sqlite_path}" "${prefix}"_rank*.sqlite
    for report in "${report_paths[@]}"; do
      local sqlite_out="${report%.nsys-rep}.sqlite"
      "${nsys_bin}" export \
        --type sqlite \
        --force-overwrite=true \
        --output "${sqlite_out}" \
        "${report}" \
        >> "${log}" 2>&1 || {
          echo "warning: nsys sqlite export failed for ${report}" | tee -a "${log}" >&2
        }
    done
  fi

  echo "DONE queue_mode=${queue_mode} bds=${n_bds} end=$(date +%H:%M:%S)"
}

for n_bds in "${rank_list[@]}"; do
  run_case "${n_bds}"
done

cat > "${summary_script}" <<'PY'
from pathlib import Path
import csv
import re
import sqlite3
import statistics
import sys

out_dir = Path(__file__).resolve().parent


def parse_psana_log(path):
    text = path.read_text(errors="replace")
    result = {
        "bds": None,
        "total_events": 0,
        "elapsed_s": 0.0,
        "total_evt_s": 0.0,
        "raw_extract_avg_s": 0.0,
    }
    m = re.search(
        r"all_ranks_summary total_events=(\d+) elapsed_s=([0-9.]+) "
        r"total_rate_evt_s=([0-9.]+) bd_ranks=(\d+)",
        text,
    )
    if m:
        result["total_events"] = int(m.group(1))
        result["elapsed_s"] = float(m.group(2))
        result["total_evt_s"] = float(m.group(3))
        result["bds"] = int(m.group(4))
    m = re.search(r"\nbds=(\d+)\n", text)
    if result["bds"] is None and m:
        result["bds"] = int(m.group(1))

    table = re.search(r"\[Rank 0\] all_bd_summary\n(?P<body>.*?)(?:\n\[Rank 0\] all_ranks_summary|\Z)", text, re.S)
    if table:
        for line in table.group("body").splitlines():
            fields = line.split()
            if fields and fields[0] == "raw_extract_avg_s" and len(fields) >= 2:
                try:
                    result["raw_extract_avg_s"] = float(fields[1])
                except ValueError:
                    pass
    return result


def table_names(conn):
    return [
        row[0]
        for row in conn.execute(
            "select name from sqlite_master where type='table' order by name"
        )
    ]


def columns(conn, table):
    return [row[1] for row in conn.execute(f'pragma table_info("{table}")')]


def first_col(cols, names):
    lowered = {c.lower(): c for c in cols}
    for name in names:
        found = lowered.get(name.lower())
        if found is not None:
            return found
    return None


def load_string_map(conn, tables):
    for table in tables:
        if table.lower() != "stringids":
            continue
        cols = columns(conn, table)
        id_col = first_col(cols, ["id"])
        value_col = first_col(cols, ["value", "string", "name"])
        if id_col and value_col:
            return {
                row[0]: row[1]
                for row in conn.execute(f'select "{id_col}", "{value_col}" from "{table}"')
            }
    return {}


def load_enum_maps(conn, tables):
    maps = {}
    for table in tables:
        if not table.upper().startswith("ENUM_"):
            continue
        cols = columns(conn, table)
        id_col = first_col(cols, ["id", "value"])
        label_col = first_col(cols, ["name", "label", "value"])
        if not id_col or not label_col or id_col == label_col:
            continue
        maps[table.upper()] = {
            row[0]: row[1]
            for row in conn.execute(f'select "{id_col}", "{label_col}" from "{table}"')
        }
    return maps


def query_intervals(conn, table):
    cols = columns(conn, table)
    start_col = first_col(cols, ["start", "startNs", "startTime"])
    end_col = first_col(cols, ["end", "endNs", "endTime"])
    if not start_col or not end_col:
        return []
    return [
        (int(row[0]), int(row[1]))
        for row in conn.execute(f'select "{start_col}", "{end_col}" from "{table}"')
        if row[0] is not None and row[1] is not None and int(row[1]) >= int(row[0])
    ]


def classify_h2d(value, enum_maps):
    if value is None:
        return False
    labels = [str(value)]
    for mapping in enum_maps.values():
        if value in mapping:
            labels.append(str(mapping[value]))
    label = " ".join(labels).lower()
    h2d_tokens = ("htod", "h2d", "host_to_device", "host to device", "host-device")
    return any(token in label for token in h2d_tokens)


def query_memcpy(conn, table, enum_maps):
    cols = columns(conn, table)
    start_col = first_col(cols, ["start", "startNs", "startTime"])
    end_col = first_col(cols, ["end", "endNs", "endTime"])
    bytes_col = first_col(cols, ["bytes", "numBytes", "size"])
    kind_col = first_col(cols, ["copyKind", "copyKindId", "memcpyKind", "kind"])
    if not start_col or not end_col:
        return []

    select_cols = [start_col, end_col]
    if bytes_col:
        select_cols.append(bytes_col)
    if kind_col:
        select_cols.append(kind_col)
    sql = "select %s from \"%s\"" % (", ".join(f'"{c}"' for c in select_cols), table)
    rows = []
    for raw in conn.execute(sql):
        start = int(raw[0])
        end = int(raw[1])
        if end < start:
            continue
        offset = 2
        nbytes = int(raw[offset]) if bytes_col and raw[offset] is not None else 0
        offset += 1 if bytes_col else 0
        kind = raw[offset] if kind_col and raw[offset] is not None else None
        rows.append(
            {
                "start": start,
                "end": end,
                "bytes": nbytes,
                "is_h2d": classify_h2d(kind, enum_maps),
            }
        )
    return rows


def query_nvtx_durations(conn, table, string_map, needle):
    cols = columns(conn, table)
    start_col = first_col(cols, ["start", "startNs", "startTime"])
    end_col = first_col(cols, ["end", "endNs", "endTime"])
    text_col = first_col(cols, ["text", "message", "name"])
    text_id_col = first_col(cols, ["textId", "messageId", "nameId"])
    if not start_col or not end_col or (not text_col and not text_id_col):
        return []
    select_cols = [start_col, end_col]
    if text_col:
        select_cols.append(text_col)
    if text_id_col:
        select_cols.append(text_id_col)
    sql = "select %s from \"%s\"" % (", ".join(f'"{c}"' for c in select_cols), table)
    durations = []
    for raw in conn.execute(sql):
        start = raw[0]
        end = raw[1]
        if start is None or end is None:
            continue
        values = []
        idx = 2
        if text_col:
            values.append(raw[idx])
            idx += 1
        if text_id_col:
            values.append(string_map.get(raw[idx], raw[idx]))
        text = " ".join(str(v) for v in values if v is not None)
        if needle in text:
            durations.append((int(end) - int(start)) / 1e6)
    return durations


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    return float(ordered[idx])


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return merged


def parse_sqlite(path):
    empty = {
        "cuda_work_s": 0.0,
        "cuda_span_s": 0.0,
        "work_span": 0.0,
        "cuda_gaps_s": 0.0,
        "p95_gap_ms": 0.0,
        "p99_gap_ms": 0.0,
        "h2d_bandwidth_gb_s": 0.0,
        "h2d_nvtx_avg_ms": 0.0,
        "raw_extract_nvtx_avg_ms": 0.0,
        "_gaps_ms": [],
        "_h2d_bytes": 0,
        "_h2d_ns": 0,
        "_h2d_nvtx_ms": [],
        "_raw_extract_nvtx_ms": [],
    }
    if not path.exists():
        return empty
    conn = sqlite3.connect(str(path))
    try:
        tables = table_names(conn)
        string_map = load_string_map(conn, tables)
        enum_maps = load_enum_maps(conn, tables)
        kernel_tables = []
        memcpy_tables = []
        for table in tables:
            cols = columns(conn, table)
            has_interval = (
                first_col(cols, ["start", "startNs", "startTime"]) is not None
                and first_col(cols, ["end", "endNs", "endTime"]) is not None
            )
            if not has_interval:
                continue
            if "KERNEL" in table.upper():
                kernel_tables.append(table)
            if "MEMCPY" in table.upper():
                memcpy_tables.append(table)
        nvtx_tables = [t for t in tables if "NVTX" in t.upper()]

        intervals = []
        work_ns = 0
        for table in kernel_tables:
            for start, end in query_intervals(conn, table):
                intervals.append((start, end))
                work_ns += end - start

        h2d_bytes = 0
        h2d_ns = 0
        for table in memcpy_tables:
            for item in query_memcpy(conn, table, enum_maps):
                intervals.append((item["start"], item["end"]))
                dur = item["end"] - item["start"]
                work_ns += dur
                if item["is_h2d"]:
                    h2d_bytes += item["bytes"]
                    h2d_ns += dur

        if intervals:
            span_ns = max(end for _start, end in intervals) - min(start for start, _end in intervals)
            merged = merge_intervals(intervals)
            gaps = [
                (merged[i][0] - merged[i - 1][1]) / 1e6
                for i in range(1, len(merged))
                if merged[i][0] > merged[i - 1][1]
            ]
        else:
            span_ns = 0
            gaps = []

        h2d_nvtx = []
        raw_extract_nvtx = []
        for table in nvtx_tables:
            h2d_nvtx.extend(query_nvtx_durations(conn, table, string_map, "psana2-gpu/h2d"))
            raw_extract_nvtx.extend(query_nvtx_durations(conn, table, string_map, "psana2-gpu/raw_extract"))

        cuda_work_s = work_ns / 1e9
        cuda_span_s = span_ns / 1e9
        return {
            "cuda_work_s": cuda_work_s,
            "cuda_span_s": cuda_span_s,
            "work_span": cuda_work_s / cuda_span_s if cuda_span_s else 0.0,
            "cuda_gaps_s": sum(gaps) / 1e3,
            "p95_gap_ms": percentile(gaps, 0.95),
            "p99_gap_ms": percentile(gaps, 0.99),
            "h2d_bandwidth_gb_s": (h2d_bytes / 1e9) / (h2d_ns / 1e9) if h2d_ns else 0.0,
            "h2d_nvtx_avg_ms": statistics.mean(h2d_nvtx) if h2d_nvtx else 0.0,
            "raw_extract_nvtx_avg_ms": statistics.mean(raw_extract_nvtx) if raw_extract_nvtx else 0.0,
            "_gaps_ms": gaps,
            "_h2d_bytes": h2d_bytes,
            "_h2d_ns": h2d_ns,
            "_h2d_nvtx_ms": h2d_nvtx,
            "_raw_extract_nvtx_ms": raw_extract_nvtx,
        }
    finally:
        conn.close()


def combine_sqlites(paths):
    parsed = [parse_sqlite(path) for path in paths]
    if not parsed:
        return parse_sqlite(Path("__missing__.sqlite"))
    gaps = [gap for item in parsed for gap in item["_gaps_ms"]]
    h2d_bytes = sum(item["_h2d_bytes"] for item in parsed)
    h2d_ns = sum(item["_h2d_ns"] for item in parsed)
    h2d_nvtx = [value for item in parsed for value in item["_h2d_nvtx_ms"]]
    raw_extract_nvtx = [
        value for item in parsed for value in item["_raw_extract_nvtx_ms"]
    ]
    cuda_work_s = sum(item["cuda_work_s"] for item in parsed)
    cuda_span_s = max((item["cuda_span_s"] for item in parsed), default=0.0)
    return {
        "cuda_work_s": cuda_work_s,
        "cuda_span_s": cuda_span_s,
        "work_span": cuda_work_s / cuda_span_s if cuda_span_s else 0.0,
        "cuda_gaps_s": sum(gaps) / 1e3,
        "p95_gap_ms": percentile(gaps, 0.95),
        "p99_gap_ms": percentile(gaps, 0.99),
        "h2d_bandwidth_gb_s": (h2d_bytes / 1e9) / (h2d_ns / 1e9) if h2d_ns else 0.0,
        "h2d_nvtx_avg_ms": statistics.mean(h2d_nvtx) if h2d_nvtx else 0.0,
        "raw_extract_nvtx_avg_ms": statistics.mean(raw_extract_nvtx) if raw_extract_nvtx else 0.0,
    }


rows = []
for log in sorted(out_dir.glob("*.log")):
    if log.name.endswith(".stats.log"):
        continue
    psana = parse_psana_log(log)
    bds = psana["bds"]
    if bds is None:
        m = re.search(r"_(\d+)bd_", log.name)
        bds = int(m.group(1)) if m else 0
    sqlite_paths = []
    single_sqlite = log.with_suffix(".sqlite")
    if single_sqlite.exists():
        sqlite_paths.append(single_sqlite)
    sqlite_paths.extend(sorted(log.parent.glob(log.stem + "_rank*.sqlite")))
    nsys = combine_sqlites(sqlite_paths)
    total_evt_s = psana["total_evt_s"]
    raw_extract_avg_ms = nsys["raw_extract_nvtx_avg_ms"] or psana["raw_extract_avg_s"] * 1e3
    rows.append(
        {
            "BDs": bds,
            "total evt/s": total_evt_s,
            "avg evt/s per BD rank": total_evt_s / bds if bds else 0.0,
            "CUDA kernel+memcpy work (s)": nsys["cuda_work_s"],
            "active CUDA span (s)": nsys["cuda_span_s"],
            "work/span": nsys["work_span"],
            "total CUDA gaps (s)": nsys["cuda_gaps_s"],
            "p95 gap (ms)": nsys["p95_gap_ms"],
            "p99 gap (ms)": nsys["p99_gap_ms"],
            "H2D bandwidth (GB/s)": nsys["h2d_bandwidth_gb_s"],
            "h2d NVTX avg (ms)": nsys["h2d_nvtx_avg_ms"],
            "raw_extract avg (ms)": raw_extract_avg_ms,
            "events": psana["total_events"],
            "elapsed_s": psana["elapsed_s"],
            "log": log.name,
            "sqlite": " ".join(path.name for path in sqlite_paths),
        }
    )

rows.sort(key=lambda row: row["BDs"])
summary = out_dir / "summary.csv"
fields = [
    "BDs",
    "total evt/s",
    "avg evt/s per BD rank",
    "CUDA kernel+memcpy work (s)",
    "active CUDA span (s)",
    "work/span",
    "total CUDA gaps (s)",
    "p95 gap (ms)",
    "p99 gap (ms)",
    "H2D bandwidth (GB/s)",
    "h2d NVTX avg (ms)",
    "raw_extract avg (ms)",
    "events",
    "elapsed_s",
    "log",
    "sqlite",
]
with summary.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: (f"{row[key]:.6g}" if isinstance(row[key], float) else row[key])
                for key in fields
            }
        )

print(summary)
headers = fields[:12]
print("\t".join(headers))
for row in rows:
    print(
        "\t".join(
            str(row[key]) if not isinstance(row[key], float) else f"{row[key]:.6g}"
            for key in headers
        )
    )
PY

"${python_bin}" "${summary_script}"

echo "Summary:"
echo "  ${output_dir}/summary.csv"
