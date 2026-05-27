#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"

nsys_bin="${NSYS_BIN:-$(command -v nsys 2>/dev/null || echo /sdf/home/m/monarin/tools/nsight-systems/2026.1.1/pkg/bin/nsys)}"
python_bin="${PYTHON_BIN:-python}"
nranks="${NRANKS:-4}"
output_prefix="${NSYS_OUTPUT_PREFIX:-${PWD}/nsys-gpu-context-overlap}"
log_path="${NSYS_LOG_PATH:-}"
trace_kinds="${NSYS_TRACE:-cuda,nvtx,osrt}"
sample_mode="${NSYS_SAMPLE:-none}"
force_overwrite="true"
show_stats="true"
per_rank_reports="false"
srun_gres_gpu=""
srun_nodes=""
srun_args=()
script_args=()

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [wrapper options] -- [run_gpu_context_overlap.py args]

Wrapper options:
  -n, --nranks N             Number of MPI ranks for 'srun -n' (default: ${nranks})
  -o, --output PREFIX        Nsight Systems output prefix (default: ${output_prefix})
  --log PATH                 Wrapper log path (default: PREFIX.log)
  --nsys PATH                Nsight Systems binary (default: ${nsys_bin})
  --python PATH              Python executable inside the activated env
  --trace LIST               Nsight trace domains (default: ${trace_kinds})
  --sample MODE              Nsight sampling mode (default: ${sample_mode})
  --per-rank-reports         Run nsys inside srun and emit PREFIX_rank<rank>.nsys-rep
  --gres-gpu N               Add --gres=gpu:N to srun
  --nodes N                  Add --nodes=N to srun
  --srun-arg ARG             Append one raw argument to srun; may be repeated
  --no-force-overwrite       Do not pass --force-overwrite=true to nsys
  --no-stats                 Skip 'nsys stats'
  --no-profile               Run the benchmark under srun without Nsight
  -h, --help                 Show this help

Examples:
  $(basename "$0") -n 4 --gres-gpu 1 -- \\
    --iterations 80 --data-size 64M --compute-iters 4096 --pipeline-depth 2

  $(basename "$0") -n 1 --gres-gpu 1 -- \\
    --iterations 80 --streams-per-rank 16 --data-size 64M --compute-iters 4096 --pipeline-depth 1

  $(basename "$0") -n 1 --gres-gpu 1 -- \\
    --iterations 80 --pipeline-mode staged --pipeline-depth 3 \\
    --data-size 64M --compute-iters 4096

  $(basename "$0") -n 4 --per-rank-reports -o /tmp/context-overlap -- \\
    --io-mode pread --io-file /path/to/input.bin --data-size 128M --compute-iters 8192
EOF
}

profile_enabled="true"

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
    --log)
      log_path="$2"
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
    --per-rank-reports)
      per_rank_reports="true"
      shift
      ;;
    --gres-gpu)
      srun_gres_gpu="$2"
      shift 2
      ;;
    --nodes)
      srun_nodes="$2"
      shift 2
      ;;
    --srun-arg)
      srun_args+=("$2")
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
    --no-profile)
      profile_enabled="false"
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

target_script="${repo_root}/psana/psana/gpu/multiowner/run_gpu_context_overlap.py"
if [[ ! -f "${target_script}" ]]; then
  echo "error: target script not found: ${target_script}" >&2
  exit 1
fi

if [[ -z "${log_path}" ]]; then
  log_path="${output_prefix}.log"
fi

mkdir -p "$(dirname "${output_prefix}")"
mkdir -p "$(dirname "${log_path}")"

exec >> "${log_path}" 2>&1

set +u
if [[ -f "${HOME}/goodstuffs/bashrc" ]]; then
  source "${HOME}/goodstuffs/bashrc"
fi
source "${HOME}/psana-nersc/activate_psana_build_env.sh" "${PSANA_BUILD_ENV_PREFIX:-${HOME}/.conda-envs/psana-build}"
if declare -F activate_psana >/dev/null 2>&1; then
  activate_psana
elif command -v activate_psana >/dev/null 2>&1; then
  activate_psana
elif [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "error: psana environment activation did not set CONDA_PREFIX, and activate_psana was not found" >&2
  exit 1
else
  echo "activate_psana not found; continuing with sourced psana build environment: CONDA_PREFIX=${CONDA_PREFIX}"
fi
if declare -F activate_psana2_gpu_cupy >/dev/null 2>&1; then
  activate_psana2_gpu_cupy >/dev/null
fi
set -u

if [[ "${profile_enabled}" == "true" && ! -x "${nsys_bin}" ]]; then
  echo "error: nsys binary not found or not executable: ${nsys_bin}" >&2
  exit 1
fi

if [[ -n "${srun_gres_gpu}" ]]; then
  srun_args+=(--gres="gpu:${srun_gres_gpu}")
fi
if [[ -n "${srun_nodes}" ]]; then
  srun_args+=(--nodes="${srun_nodes}")
fi

echo "repo_root=${repo_root}"
echo "target_script=${target_script}"
echo "nsys_bin=${nsys_bin}"
echo "python_bin=${python_bin}"
echo "nranks=${nranks}"
echo "output_prefix=${output_prefix}"
echo "log_path=${log_path}"
echo "trace=${trace_kinds}"
echo "sample=${sample_mode}"
echo "per_rank_reports=${per_rank_reports}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "script_args=${script_args[*]}"
echo "srun_args=${srun_args[*]}"

nsys_args=(
  profile
  --sample="${sample_mode}"
  --trace="${trace_kinds}"
  --cuda-memory-usage=true
)
if [[ "${force_overwrite}" == "true" ]]; then
  nsys_args+=(--force-overwrite=true)
fi

set -x
if [[ "${profile_enabled}" != "true" ]]; then
  srun "${srun_args[@]}" -n "${nranks}" "${python_bin}" "${target_script}" "${script_args[@]}"
elif [[ "${per_rank_reports}" == "true" ]]; then
  srun "${srun_args[@]}" -n "${nranks}" \
    "${nsys_bin}" "${nsys_args[@]}" -o "${output_prefix}_rank%q{SLURM_PROCID}" \
    "${python_bin}" "${target_script}" "${script_args[@]}"
else
  "${nsys_bin}" "${nsys_args[@]}" -o "${output_prefix}" \
    srun "${srun_args[@]}" -n "${nranks}" "${python_bin}" "${target_script}" "${script_args[@]}"
fi
set +x

if [[ "${profile_enabled}" != "true" || "${show_stats}" != "true" ]]; then
  echo "Log:"
  echo "  ${log_path}"
  exit 0
fi

echo
echo "=== nsys stats: CUDA/NVTX summaries ==="
if [[ "${per_rank_reports}" == "true" ]]; then
  shopt -s nullglob
  reports=("${output_prefix}"_rank*.nsys-rep)
  shopt -u nullglob
else
  reports=("${output_prefix}.nsys-rep")
fi

if ((${#reports[@]} == 0)); then
  echo "warning: no Nsight reports found for ${output_prefix}" >&2
else
  for report in "${reports[@]}"; do
    if [[ ! -f "${report}" ]]; then
      echo "warning: expected report not found: ${report}" >&2
      continue
    fi
    echo
    echo "Report: ${report}"
    (
      cd "$(dirname "${report}")"
      "${nsys_bin}" stats \
        --force-export=true \
        --report cuda_api_gpu_sum,cuda_gpu_sum,cuda_kern_exec_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,nvtx_pushpop_sum,osrt_sum \
        "$(basename "${report}")"
    )
  done
fi

echo
echo "Report prefix:"
echo "  ${output_prefix}"
echo "Log:"
echo "  ${log_path}"
