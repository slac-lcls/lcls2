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

mode="${1:?usage: run_queue_compare_mode_sweep.sbatch MODE [OUTPUT_ROOT]}"
job_id="${SLURM_JOB_ID:-manual}"
scratch_root="${PSANA_GPU_SCRATCH:-${SCRATCH:-/pscratch/sd/m/${USER}}/psana2-gpu}"
output_root="${2:-${scratch_root}/jungfrau_queue_compare/queue_compare_sweep_${job_id}}"
repo_root="/global/homes/m/monarin/lcls2"
xtc_dir="/pscratch/sd/p/psdatmgr/psdm/mfx/mfx101629726/xtc"
events=16000
gpu_device=0
bd_local_depth=2
gpu_leases=2

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

echo "QUEUE_SWEEP_BEGIN mode=${mode} output=${output_root}/${mode} host=$(hostname) start=$(date -Is)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_QOS=${SLURM_JOB_QOS:-}"
echo "SLURM_JOB_ACCOUNT=${SLURM_JOB_ACCOUNT:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

for bds in $(seq 1 16); do
  nranks=$((bds + 2))
  log="${output_root}/${mode}/${mode}_${bds}bd_${events}ev.log"
  echo "RUN mode=${mode} bds=${bds} nranks=${nranks} events=${events} start=$(date -Is)"
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
  echo "DONE mode=${mode} bds=${bds} end=$(date -Is)"
done

python - "${output_root}/${mode}" <<'PY'
from pathlib import Path
import csv
import re
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

def table_metric(text, name):
    m = re.search(rf"^{re.escape(name)}\s+([0-9.+-eE]+)\s+", text, re.M)
    return float(m.group(1)) if m else 0.0

rows = []
for log in sorted(out_dir.glob("*bd_16000ev.log"), key=lambda p: int(re.search(r"_(\d+)bd_", p.name).group(1))):
    text = log.read_text(errors="replace")
    bds_match = re.search(r"^bds=(\d+)$", text, re.M)
    bds = int(bds_match.group(1)) if bds_match else int(re.search(r"_(\d+)bd_", log.name).group(1))

    all_ranks = re.search(
        r"all_ranks_summary total_events=(\d+) elapsed_s=([0-9.]+) total_rate_evt_s=([0-9.]+) bd_ranks=(\d+)",
        text,
    )
    total_rate = float(all_ranks.group(3)) if all_ranks else 0.0

    all_bd = re.search(
        r"all_bd_summary\nqueue_mode=.*?copied_mib_total=([0-9.]+) copy_rate_mib_s_global=([0-9.]+)\n(?P<table>.*?)(?:\n\[Rank 0\] all_ranks_summary|\Z)",
        text,
        re.S,
    )
    table = all_bd.group("table") if all_bd else ""
    copy_avg = table_metric(table, "copy_total_s")
    kernel_avg = table_metric(table, "kernel_total_s")
    raw_extract_avg = table_metric(table, "raw_extract_avg_s")
    copy_rate_mib_s = float(all_bd.group(2)) if all_bd else 0.0

    rows.append({
        "BDs": bds,
        "total evt/s": total_rate,
        "avg evt/s per BD rank": total_rate / bds if bds else 0.0,
        "CUDA kernel+memcpy work (s)": (copy_avg + kernel_avg) * bds,
        "active CUDA span (s)": "",
        "work/span": "",
        "total CUDA gaps (s)": "",
        "p95 gap (ms)": "",
        "p99 gap (ms)": "",
        "H2D bandwidth (GB/s)": copy_rate_mib_s * 1048576.0 / 1e9,
        "h2d NVTX avg (ms)": "",
        "raw_extract avg (ms)": raw_extract_avg * 1000.0,
    })

summary = out_dir / "summary.csv"
with summary.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({
            key: (f"{value:.6g}" if isinstance(value, float) else value)
            for key, value in row.items()
        })

print(summary)
print("\t".join(fields))
for row in rows:
    print("\t".join(str(row[key]) if not isinstance(row[key], float) else f"{row[key]:.6g}" for key in fields))
PY

echo "QUEUE_SWEEP_DONE mode=${mode} summary=${output_root}/${mode}/summary.csv end=$(date -Is)"
