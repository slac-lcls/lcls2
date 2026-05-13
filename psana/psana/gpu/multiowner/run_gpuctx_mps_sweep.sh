#!/usr/bin/env bash
set -euo pipefail

node="${NODE:-nid001028}"
scratch="${SCRATCH:-/pscratch/sd/m/monarin}"
base="${scratch}/psana2-gpu/multiowner"
nomps_dir="${base}/nomps"
mps_dir="${base}/mps"
mps_tmp="${scratch}/psana2-gpu/tmp/mps"
mps_pipe="/tmp/nvidia-mps-${USER}-gpuctx-sweep-${node}"
wrapper="/global/homes/m/monarin/lcls2/psana/psana/gpu/multiowner/run_nsys_gpu_context_overlap.sh"
summary="${base}/gpuctx_n1_16_mps_nomps_summary.txt"

ranks=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
script_args=(
  --iterations 80
  --data-size 64M
  --compute-iters 4096
  --pipeline-depth 1
  --gpu-id 0
)

mkdir -p "${nomps_dir}" "${mps_dir}" "${mps_tmp}"

run_on_node() {
  srun \
    --overlap \
    --nodes=1 \
    --nodelist="${node}" \
    --ntasks=1 \
    --gres=gpu:1 \
    --gpu-bind=none \
    "$@"
}

stop_mps() {
  echo "[$(date -Is)] stopping any existing MPS on ${node}"
  run_on_node bash -lc '
    export CUDA_MPS_PIPE_DIRECTORY="'"${mps_pipe}"'"
    export CUDA_MPS_LOG_DIRECTORY="'"${mps_tmp}"'"
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    sleep 1
    echo "server_list_after_stop=$(echo get_server_list | nvidia-cuda-mps-control 2>/dev/null | tr "\n" " " || true)"
  '
}

start_mps() {
  echo "[$(date -Is)] starting MPS on ${node}"
  rm -rf "${mps_tmp:?}/"*
  run_on_node bash -lc '
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_MPS_PIPE_DIRECTORY="'"${mps_pipe}"'"
    export CUDA_MPS_LOG_DIRECTORY="'"${mps_tmp}"'"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d
    set +u
    source "$HOME/goodstuffs/bashrc" 2>/dev/null || true
    source "$HOME/psana-nersc/activate_psana_build_env.sh" "${PSANA_BUILD_ENV_PREFIX:-$HOME/.conda-envs/psana-build}" >/dev/null
    activate_psana 2>/dev/null || true
    activate_psana2_gpu_cupy 2>/dev/null || true
    set -u
    python - <<'"'"'PY'"'"'
import cupy as cp
x = cp.arange(1)
cp.cuda.Stream.null.synchronize()
print("mps_warmup_device_count", cp.cuda.runtime.getDeviceCount())
print("mps_warmup_value", int(x.get()[0]))
PY
    echo "MPS server list:"
    echo get_server_list | nvidia-cuda-mps-control | tee "'"${mps_tmp}"'/server_list_after_start.txt"
    echo "MPS log directory: $CUDA_MPS_LOG_DIRECTORY"
    ls -l "$CUDA_MPS_LOG_DIRECTORY"
  '
}

run_one() {
  local mode="$1"
  local nranks="$2"
  local outdir="$3"
  local prefix="${outdir}/gpuctx_n${nranks}_d1_${mode}"

  echo "[$(date -Is)] ${mode}: nranks=${nranks} prefix=${prefix}"
  export CUDA_VISIBLE_DEVICES=0
  export CUDA_MPS_PIPE_DIRECTORY="${mps_pipe}"
  export CUDA_MPS_LOG_DIRECTORY="${mps_tmp}"
  "${wrapper}" \
    -n "${nranks}" \
    --gres-gpu 1 \
    --srun-arg "--nodelist=${node}" \
    --srun-arg "--gpu-bind=none" \
    -o "${prefix}" \
    -- \
    "${script_args[@]}"
}

{
  echo "sweep_start=$(date -Is)"
  echo "node=${node}"
  echo "nomps_dir=${nomps_dir}"
  echo "mps_dir=${mps_dir}"
  echo "mps_tmp=${mps_tmp}"
  echo "mps_pipe=${mps_pipe}"
  echo "script_args=${script_args[*]}"
} | tee "${summary}"

stop_mps

for nranks in "${ranks[@]}"; do
  run_one "nomps" "${nranks}" "${nomps_dir}"
done

start_mps

for nranks in "${ranks[@]}"; do
  run_one "mps" "${nranks}" "${mps_dir}"
done

echo "[$(date -Is)] final MPS server list"
run_on_node bash -lc '
  export CUDA_MPS_PIPE_DIRECTORY="'"${mps_pipe}"'"
  export CUDA_MPS_LOG_DIRECTORY="'"${mps_tmp}"'"
  echo get_server_list | nvidia-cuda-mps-control | tee "'"${mps_tmp}"'/server_list_after_sweep.txt"
  ls -l "$CUDA_MPS_LOG_DIRECTORY"
'

stop_mps

{
  echo
  echo "sweep_end=$(date -Is)"
  echo
  echo "=== aggregate summaries ==="
  for f in "${nomps_dir}"/gpuctx_n*_d1_nomps.log "${mps_dir}"/gpuctx_n*_d1_mps.log; do
    [[ -f "$f" ]] || continue
    printf "%s " "$(basename "$f")"
    grep "aggregate" "$f" | tail -n 1 || true
  done
  echo
  echo "=== report paths ==="
  find "${nomps_dir}" "${mps_dir}" -maxdepth 1 -name "*.nsys-rep" -print | sort
  echo
  echo "=== mps logs ==="
  find "${mps_tmp}" -maxdepth 1 -type f -print | sort
} | tee -a "${summary}"

echo "Summary: ${summary}"
