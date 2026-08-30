#!/bin/bash
#SBATCH --job-name=jf_1bd_profiles
#SBATCH --partition=ampere
#SBATCH --account=lcls
#SBATCH --nodelist=sdfampere032
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:45:00
#SBATCH --output=/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-d2h-pipeline/notes/jungfrau_1bd_bottleneck_20260828/profiles_%j.log

repo=/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-d2h-pipeline
source_xtc=/sdf/data/lcls/ds/mfx/mfx101210926/xtc
stage=/lscratch/monarin/jf_1bd_bottleneck_${SLURM_JOB_ID}
log_dir=${repo}/notes/jungfrau_1bd_bottleneck_20260828
cpu_driver=${log_dir}/cpu_1bd_profile.py
gpu_driver=${log_dir}/gpu_1bd_profile.py
kvikio_driver=${log_dir}/kvikio_read_baseline.py
h2d_driver=${log_dir}/h2d_baseline.py

source "${repo}/setup_env.sh" >/dev/null 2>&1
set -euo pipefail

export PS_EB_NODES=1
export PS_SRV_NODES=0
export PS_PARALLEL=mpi
export PS_SMD_N_EVENTS=1000
export PYTHONUNBUFFERED=1
export OMPI_MCA_btl='^smcuda'
export TMPDIR=/lscratch/monarin/tmp
mkdir -p "${TMPDIR}" "${stage}/smalldata"

echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID}"
echo "commit=$(git -C "${repo}" rev-parse HEAD)"
echo "stage=${stage}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
awk '/Cpus_allowed_list/{print "batch_" $0}' /proc/self/status

python - "${source_xtc}" "${stage}" <<'PY'
import os
import sys

source_xtc, stage = sys.argv[1:]
prefix_sizes = {
    5: 25905070080,
    6: 26842497024,
    7: 25494966272,
    8: 26439843840,
    9: 25500319744,
}
copy_size = 16 * 1024 * 1024
for stream, nbytes in prefix_sizes.items():
    name = f"mfx101210926-r0387-s{stream:03d}-c000.xtc2"
    source_path = os.path.join(source_xtc, name)
    stage_path = os.path.join(stage, name)
    remaining = nbytes
    with open(source_path, "rb", buffering=0) as source, open(
        stage_path, "wb", buffering=0
    ) as target:
        while remaining:
            data = source.read(min(copy_size, remaining))
            if not data:
                raise RuntimeError(f"short source while staging {source_path}")
            target.write(data)
            remaining -= len(data)
    print(f"staged={stage_path} bytes={nbytes}", flush=True)
PY
cp "${source_xtc}"/smalldata/mfx101210926-r0387-s00[5-9]-c000.smd.xtc2 "${stage}/smalldata/"
sync
findmnt -T "${stage}" -o TARGET,SOURCE,FSTYPE,OPTIONS
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv

python - <<'PY'
import cupy as cp
import kvikio
import kvikio.defaults
print(f"cupy={cp.__version__}")
print(f"cuda_runtime={cp.cuda.runtime.runtimeGetVersion()}")
print(f"kvikio={kvikio.__version__}")
print(f"gds_available={kvikio.DriverProperties().is_gds_available}")
print(f"compat_mode={kvikio.defaults.compat_mode()}")
print(f"default_nthreads={kvikio.defaults.get_num_threads()}")
print(f"default_task_size={kvikio.defaults.task_size()}")
PY

files=("${stage}"/*.xtc2)

evict_bigdata() {
    sync
    python - "${stage}" <<'PY'
import glob
import os
import sys
paths = sorted(glob.glob(os.path.join(sys.argv[1], "*.xtc2")))
if len(paths) != 5:
    raise RuntimeError(f"expected five BigData files, found {len(paths)}")
for path in paths:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
print(f"evicted_bigdata_files={len(paths)}", flush=True)
PY
}

run_kvikio() {
    local nthreads=$1
    local task_mib=$2
    local tag=nt${nthreads}_task${task_mib}
    evict_bigdata
    echo "KVIKIO_CASE_BEGIN ${tag} date=$(date -Is)"
    KVIKIO_NTHREADS=${nthreads} KVIKIO_TASK_SIZE=$((task_mib * 1024 * 1024)) \
        python -u "${kvikio_driver}" \
        --chunk-mib 64 --bytes-per-file-gib 4 --task-mib "${task_mib}" \
        "${files[@]}" 2>&1 | tee "${log_dir}/kvikio_${tag}_${SLURM_JOB_ID}.log"
    echo "KVIKIO_CASE_END ${tag} date=$(date -Is)"
}

for nthreads in 1 2 4 8 16; do
    run_kvikio "${nthreads}" 4
done
for task_mib in 1 16 64; do
    run_kvikio 8 "${task_mib}"
done

for mode in pageable pinned; do
    echo "H2D_CASE_BEGIN mode=${mode} date=$(date -Is)"
    python -u "${h2d_driver}" --mode "${mode}" --buffer-mib 256 --total-gib 20 \
        2>&1 | tee "${log_dir}/h2d_${mode}_${SLURM_JOB_ID}.log"
    echo "H2D_CASE_END mode=${mode} date=$(date -Is)"
done

run_cpu() {
    local chunk_mib=$1
    local read_threads=$2
    local preadv=$3
    local tag=chunk${chunk_mib}_rt${read_threads}_preadv${preadv}
    local preadv_arg=()
    if [[ ${preadv} -eq 1 ]]; then
        preadv_arg=(--preadv)
    fi
    evict_bigdata
    echo "CPU_CASE_BEGIN ${tag} date=$(date -Is)"
    PS_BD_CHUNKSIZE=$((chunk_mib * 1024 * 1024)) \
        mpirun -n 3 --bind-to none \
        python -u "${cpu_driver}" \
        --dir "${stage}" --max-events 1000 \
        --read-threads "${read_threads}" "${preadv_arg[@]}" \
        2>&1 | tee "${log_dir}/cpu_${tag}_${SLURM_JOB_ID}.log"
    echo "CPU_CASE_END ${tag} date=$(date -Is)"
}

run_cpu 16 1 0
run_cpu 64 1 0
run_cpu 256 1 0
run_cpu 16 1 1
run_cpu 64 1 1
run_cpu 256 1 1
run_cpu 64 2 1
run_cpu 64 4 1
run_cpu 16 5 0
run_cpu 64 5 0
run_cpu 256 5 0
run_cpu 16 5 1
run_cpu 64 5 1
run_cpu 256 5 1

run_gpu() {
    local cache=$1
    local nthreads=$2
    local task_mib=$3
    local tag=${cache}_nt${nthreads}_task${task_mib}
    if [[ ${cache} == cold ]]; then
        evict_bigdata
    fi
    echo "GPU_CASE_BEGIN ${tag} date=$(date -Is)"
    KVIKIO_NTHREADS=${nthreads} KVIKIO_TASK_SIZE=$((task_mib * 1024 * 1024)) \
        mpirun -n 3 --bind-to none \
        python -u "${gpu_driver}" \
        --dir "${stage}" --max-events 1000 --batch-size 20 --pool-depth 1 \
        2>&1 | tee "${log_dir}/gpu_${tag}_${SLURM_JOB_ID}.log"
    echo "GPU_CASE_END ${tag} date=$(date -Is)"
}

run_gpu cold 8 4
run_gpu warm 8 4
run_gpu cold 16 4
run_gpu cold 8 1
run_gpu cold 8 16
run_gpu cold 8 64

echo "profiles_complete=true date=$(date -Is)"
echo "keep_stage=${stage}"
