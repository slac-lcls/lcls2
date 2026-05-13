#!/usr/bin/env bash
#SBATCH -A lcls
#SBATCH -q premium
#SBATCH -C gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/pscratch/sd/m/monarin/psana2-gpu/slurm_logs/%x-%j.out
#SBATCH --error=/pscratch/sd/m/monarin/psana2-gpu/slurm_logs/%x-%j.err

set -euo pipefail

mode="${1:?usage: run_queue_compare_nsys_mode_sweep.sbatch MODE [OUTPUT_ROOT]}"
job_id="${SLURM_JOB_ID:-manual}"
scratch_root="${PSANA_GPU_SCRATCH:-${SCRATCH:-/pscratch/sd/m/${USER}}/psana2-gpu}"
output_root="${2:-${scratch_root}/jungfrau_queue_compare/queue_compare_nsys_sweep_${job_id}}"
repo_root="/global/homes/m/monarin/lcls2"
xtc_dir="/pscratch/sd/p/psdatmgr/psdm/mfx/mfx101629726/xtc"
events=16000
gpu_device=0
bd_local_depth=2
gpu_leases=2
nsys_bin="${NSYS_BIN:-$(command -v nsys)}"

case "${mode}" in
  baseline|one-side|both-sides-lease) ;;
  *)
    echo "error: unsupported mode=${mode}" >&2
    exit 2
    ;;
esac

mkdir -p "${output_root}/${mode}" "${scratch_root}/slurm_logs"

cd "${repo_root}"
source "${HOME}/psana-nersc/activate_psana_build_env.sh" "${HOME}/.conda-envs/psana-build" >/dev/null
if ! declare -F activate_psana >/dev/null 2>&1 && [[ -f "${HOME}/goodstuffs/bashrc" ]]; then
  source "${HOME}/goodstuffs/bashrc" >/dev/null
fi
if declare -F activate_psana >/dev/null 2>&1; then
  activate_psana >/dev/null
else
  export PYTHONPATH="${repo_root}/install_psana/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
  export PATH="${repo_root}/install_psana/bin:${PATH}"
fi

echo "QUEUE_NSYS_SWEEP_BEGIN mode=${mode} output=${output_root}/${mode} host=$(hostname) start=$(date -Is)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_QOS=${SLURM_JOB_QOS:-}"
echo "SLURM_JOB_ACCOUNT=${SLURM_JOB_ACCOUNT:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "NSYS_BIN=${nsys_bin}"

for bds in $(seq 1 16); do
  nranks=$((bds + 2))
  prefix="${output_root}/${mode}/${mode}_${bds}bd_${events}ev"
  tmp_prefix="/tmp/${USER}_queue_nsys_${job_id}_${mode}_${bds}bd_${events}ev"
  log="${prefix}.log"
  report="${prefix}.nsys-rep"
  sqlite="${prefix}.sqlite"

  echo "RUN mode=${mode} bds=${bds} nranks=${nranks} events=${events} start=$(date -Is)"
  rm -f "${tmp_prefix}.nsys-rep" "${tmp_prefix}.qdstrm" "${sqlite}" "${report}"
  {
    echo "mode=${mode}"
    echo "bds=${bds}"
    echo "nranks=${nranks}"
    echo "events=${events}"
    echo "xtc_dir=${xtc_dir}"
    echo "gpu_device=${gpu_device}"
    echo "bd_local_depth=${bd_local_depth}"
    echo "gpu_leases=${gpu_leases}"
    echo "start=$(date -Is)"
    "${nsys_bin}" profile \
      --sample=none \
      --trace=cuda,nvtx \
      --cuda-memory-usage=true \
      --force-overwrite=true \
      -o "${tmp_prefix}" \
      srun -n "${nranks}" --gres=gpu:1 --gpu-bind=none \
        python psana/psana/gpu/multiowner/run_jungfrau_queue_compare.py \
          -e mfx101629726 \
          -r 356 \
          --xtc-dir "${xtc_dir}" \
          --max-events "${events}" \
          --queue-mode "${mode}" \
          --gpu-device "${gpu_device}" \
          --bd-local-depth "${bd_local_depth}" \
          --gpu-leases "${gpu_leases}" \
          --print-interval "${events}"
    echo "end=$(date -Is)"
  } > "${log}" 2>&1

  if [[ -f "${tmp_prefix}.nsys-rep" ]]; then
    cp "${tmp_prefix}.nsys-rep" "${report}"
    "${nsys_bin}" export \
      --type sqlite \
      --force-overwrite=true \
      --output "${sqlite}" \
      "${tmp_prefix}.nsys-rep" >> "${log}" 2>&1
  else
    echo "warning: missing Nsight report ${tmp_prefix}.nsys-rep" | tee -a "${log}" >&2
  fi
  echo "DONE mode=${mode} bds=${bds} end=$(date -Is)"
done

python - "${output_root}/${mode}" <<'PY'
from pathlib import Path
import csv
import re
import sqlite3
import statistics
import sys

out_dir = Path(sys.argv[1])
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
]

def first_col(cols, names):
    lower = {c.lower(): c for c in cols}
    for name in names:
        found = lower.get(name.lower())
        if found:
            return found
    return None

def columns(conn, table):
    return [row[1] for row in conn.execute(f'pragma table_info("{table}")')]

def string_map(conn, tables):
    for table in tables:
        if table.lower() != "stringids":
            continue
        cols = columns(conn, table)
        id_col = first_col(cols, ["id"])
        val_col = first_col(cols, ["value", "string", "name"])
        if id_col and val_col:
            return dict(conn.execute(f'select "{id_col}", "{val_col}" from "{table}"'))
    return {}

def enum_maps(conn, tables):
    maps = {}
    for table in tables:
        if not table.upper().startswith("ENUM_"):
            continue
        cols = columns(conn, table)
        id_col = first_col(cols, ["id", "value"])
        label_col = first_col(cols, ["name", "label", "value"])
        if id_col and label_col and id_col != label_col:
            maps[table] = dict(conn.execute(f'select "{id_col}", "{label_col}" from "{table}"'))
    return maps

def classify_h2d(kind, enums):
    labels = [str(kind)]
    for mapping in enums.values():
        if kind in mapping:
            labels.append(str(mapping[kind]))
    text = " ".join(labels).lower()
    return any(token in text for token in ("htod", "h2d", "host_to_device", "host to device", "host-device"))

def pct(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    return float(values[int(round((len(values) - 1) * q))])

def merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return merged

def nvtx_durations(conn, table, smap, needle):
    cols = columns(conn, table)
    start_col = first_col(cols, ["start", "startNs", "startTime"])
    end_col = first_col(cols, ["end", "endNs", "endTime"])
    text_col = first_col(cols, ["text", "message", "name"])
    text_id_col = first_col(cols, ["textId", "messageId", "nameId"])
    if not start_col or not end_col or (not text_col and not text_id_col):
        return []
    select = [start_col, end_col]
    if text_col:
        select.append(text_col)
    if text_id_col:
        select.append(text_id_col)
    vals = []
    for row in conn.execute("select %s from \"%s\"" % (", ".join(f'"{c}"' for c in select), table)):
        parts = []
        idx = 2
        if text_col:
            parts.append(row[idx])
            idx += 1
        if text_id_col:
            parts.append(smap.get(row[idx], row[idx]))
        if needle in " ".join(str(p) for p in parts if p is not None):
            vals.append((int(row[1]) - int(row[0])) / 1e6)
    return vals

def parse_sqlite(path):
    if not path.exists():
        return {k: 0.0 for k in ("work", "span", "work_span", "gaps", "p95", "p99", "h2d_bw", "h2d_nvtx", "raw_nvtx")}
    conn = sqlite3.connect(str(path))
    try:
        tables = [r[0] for r in conn.execute("select name from sqlite_master where type='table'")]
        smap = string_map(conn, tables)
        emaps = enum_maps(conn, tables)
        intervals = []
        work_ns = 0
        h2d_bytes = 0
        h2d_ns = 0
        h2d_nvtx = []
        raw_nvtx = []
        for table in tables:
            cols = columns(conn, table)
            start_col = first_col(cols, ["start", "startNs", "startTime"])
            end_col = first_col(cols, ["end", "endNs", "endTime"])
            if not start_col or not end_col:
                continue
            upper = table.upper()
            if "KERNEL" in upper:
                for start, end in conn.execute(f'select "{start_col}", "{end_col}" from "{table}"'):
                    if start is not None and end is not None and int(end) >= int(start):
                        start, end = int(start), int(end)
                        intervals.append((start, end))
                        work_ns += end - start
            if "MEMCPY" in upper:
                bytes_col = first_col(cols, ["bytes", "numBytes", "size"])
                kind_col = first_col(cols, ["copyKind", "copyKindId", "memcpyKind", "kind"])
                select = [start_col, end_col]
                if bytes_col:
                    select.append(bytes_col)
                if kind_col:
                    select.append(kind_col)
                for row in conn.execute("select %s from \"%s\"" % (", ".join(f'"{c}"' for c in select), table)):
                    start, end = int(row[0]), int(row[1])
                    if end < start:
                        continue
                    intervals.append((start, end))
                    dur = end - start
                    work_ns += dur
                    idx = 2
                    nbytes = int(row[idx]) if bytes_col and row[idx] is not None else 0
                    idx += 1 if bytes_col else 0
                    kind = row[idx] if kind_col and row[idx] is not None else None
                    if classify_h2d(kind, emaps):
                        h2d_bytes += nbytes
                        h2d_ns += dur
            if "NVTX" in upper:
                h2d_nvtx.extend(nvtx_durations(conn, table, smap, "psana2-gpu/h2d"))
                raw_nvtx.extend(nvtx_durations(conn, table, smap, "psana2-gpu/raw_extract"))
        span_ns = max((e for _s, e in intervals), default=0) - min((s for s, _e in intervals), default=0)
        gaps_ms = []
        m = merge(intervals)
        for i in range(1, len(m)):
            if m[i][0] > m[i - 1][1]:
                gaps_ms.append((m[i][0] - m[i - 1][1]) / 1e6)
        work_s = work_ns / 1e9
        span_s = span_ns / 1e9
        return {
            "work": work_s,
            "span": span_s,
            "work_span": work_s / span_s if span_s else 0.0,
            "gaps": sum(gaps_ms) / 1000.0,
            "p95": pct(gaps_ms, 0.95),
            "p99": pct(gaps_ms, 0.99),
            "h2d_bw": (h2d_bytes / 1e9) / (h2d_ns / 1e9) if h2d_ns else 0.0,
            "h2d_nvtx": statistics.mean(h2d_nvtx) if h2d_nvtx else 0.0,
            "raw_nvtx": statistics.mean(raw_nvtx) if raw_nvtx else 0.0,
        }
    finally:
        conn.close()

rows = []
for log in sorted(out_dir.glob("*bd_16000ev.log"), key=lambda p: int(re.search(r"_(\d+)bd_", p.name).group(1))):
    text = log.read_text(errors="replace")
    bds = int(re.search(r"^bds=(\d+)$", text, re.M).group(1))
    m = re.search(r"all_ranks_summary total_events=(\d+) elapsed_s=([0-9.]+) total_rate_evt_s=([0-9.]+) bd_ranks=(\d+)", text)
    rate = float(m.group(3)) if m else 0.0
    ns = parse_sqlite(log.with_suffix(".sqlite"))
    rows.append({
        "BDs": bds,
        "total evt/s": rate,
        "avg evt/s per BD rank": rate / bds if bds else 0.0,
        "CUDA kernel+memcpy work (s)": ns["work"],
        "active CUDA span (s)": ns["span"],
        "work/span": ns["work_span"],
        "total CUDA gaps (s)": ns["gaps"],
        "p95 gap (ms)": ns["p95"],
        "p99 gap (ms)": ns["p99"],
        "H2D bandwidth (GB/s)": ns["h2d_bw"],
        "h2d NVTX avg (ms)": ns["h2d_nvtx"],
        "raw_extract avg (ms)": ns["raw_nvtx"],
    })

summary = out_dir / "summary.csv"
with summary.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})

print(summary)
print("\t".join(fields))
for row in rows:
    print("\t".join(str(row[k]) if not isinstance(row[k], float) else f"{row[k]:.6g}" for k in fields))
PY

echo "QUEUE_NSYS_SWEEP_DONE mode=${mode} summary=${output_root}/${mode}/summary.csv end=$(date -Is)"
