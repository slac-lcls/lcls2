#!/bin/bash
#SBATCH --job-name=jf_gpu_10k
#SBATCH --partition=ampere
#SBATCH --account=lcls
#SBATCH --nodelist=sdfampere032
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:a100:4
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-d2h-pipeline/notes/jungfrau_gpu_scale_20260828/scale10k_%j.log

repo=/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-d2h-pipeline
source_xtc=/sdf/data/lcls/ds/mfx/mfx101210926/xtc
stage=/lscratch/monarin/jf_gpu_10k_${SLURM_JOB_ID}
log_dir=${repo}/notes/jungfrau_gpu_scale_20260828
driver=${log_dir}/gpu_scale_profile.py
max_events=10000

source "${repo}/setup_env.sh" >/dev/null 2>&1
set -uo pipefail

export PS_EB_NODES=1
export PS_SRV_NODES=0
export PS_PARALLEL=mpi
export PS_SMD_N_EVENTS=1000
export PYTHONUNBUFFERED=1
export OMPI_MCA_btl='^smcuda'
export TMPDIR=/lscratch/monarin/tmp
export KVIKIO_NTHREADS=8
export KVIKIO_TASK_SIZE=$((1024 * 1024))
mkdir -p "${TMPDIR}" "${stage}/smalldata" "${log_dir}"

echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID}"
echo "commit=$(git -C "${repo}" rev-parse HEAD)"
echo "git_status_begin"
git -C "${repo}" status --short
echo "git_status_end"
echo "stage=${stage}"
echo "config=kvikio_threads:8,kvikio_task_mib:1,batch_size:20,pool_depth:1,max_events:${max_events}"
echo "matrix=g1:bd1,2,4,6,8;g2:bd2,4,6;g4:bd4,8,12;repetitions:2"

cleanup_stage() {
    case "${stage}" in
        /lscratch/monarin/jf_gpu_10k_[0-9]*)
            rm -rf -- "${stage}"
            echo "removed_stage=${stage}"
            ;;
        *)
            echo "refusing_to_remove_unexpected_stage=${stage}" >&2
            ;;
    esac
}
trap cleanup_stage EXIT

# Prefix endpoints come from derive_stage_extents.py over the first 10,000
# GPUBAT1 events, plus a 16-MiB safety margin on each stream.
python - "${source_xtc}" "${stage}" <<'PY'
import concurrent.futures
import os
import sys

source_xtc, stage = sys.argv[1:]
prefix_sizes = {
    5: 62_936_655_408,
    6: 73_423_260_710,
    7: 52_450_050_106,
    8: 73_423_260_710,
    9: 73_423_260_710,
}
required = sum(prefix_sizes.values())
available = os.statvfs(stage).f_frsize * os.statvfs(stage).f_bavail
reserve = 20 * 1024**3
print(f"stage_required_bytes={required}")
print(f"stage_available_bytes={available}")
print(f"stage_fraction_of_available={required / available:.6f}")
if available < required + reserve:
    raise RuntimeError(
        f"insufficient /lscratch: required={required} reserve={reserve} "
        f"available={available}"
    )

copy_size = 16 * 1024 * 1024


def stage_stream(item):
    stream, nbytes = item
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
    if os.path.getsize(stage_path) != nbytes:
        raise RuntimeError(f"wrong staged size for {stage_path}")
    return stage_path, nbytes


# The five source files are independent. Concurrent staging avoids making a
# long benchmark allocation wait on one remote sequential stream at a time.
with concurrent.futures.ThreadPoolExecutor(max_workers=len(prefix_sizes)) as pool:
    futures = [pool.submit(stage_stream, item) for item in prefix_sizes.items()]
    for future in concurrent.futures.as_completed(futures):
        stage_path, nbytes = future.result()
        print(f"staged={stage_path} bytes={nbytes}", flush=True)
PY
cp "${source_xtc}"/smalldata/mfx101210926-r0387-s00[5-9]-c000.smd.xtc2 "${stage}/smalldata/"
sync

findmnt -T "${stage}" -o TARGET,SOURCE,FSTYPE,OPTIONS,AVAIL,SIZE
df -hT "${stage}"
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv
nvidia-smi topo -m

CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import cupy as cp
import kvikio
import kvikio.defaults

print(f"cupy={cp.__version__}")
print(f"cuda_runtime={cp.cuda.runtime.runtimeGetVersion()}")
print(f"kvikio={kvikio.__version__}")
print(f"gds_available={kvikio.DriverProperties().is_gds_available}")
print(f"compat_mode={kvikio.defaults.compat_mode()}")
print(f"nthreads={kvikio.defaults.get_num_threads()}")
print(f"task_size={kvikio.defaults.task_size()}")
PY

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
    print(f"evicted={path}", flush=True)
PY
}

run_case() {
    local n_gpus=$1
    local n_bds=$2
    local repetition=$3
    local n_ranks=$((2 + n_bds))
    local tag=g${n_gpus}_bd${n_bds}_r${repetition}
    local case_log=${log_dir}/gpu10k_${tag}_${SLURM_JOB_ID}.log
    local mem_log=${log_dir}/gpu10k_mem_${tag}_${SLURM_JOB_ID}.csv

    evict_bigdata
    echo "CASE_BEGIN tag=${tag} ranks=${n_ranks} date=$(date -Is)"
    export SLURM_GPUS_ON_NODE=${n_gpus}

    nvidia-smi \
        --query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory \
        --format=csv,noheader,nounits -lms 500 2>&1 | tee "${mem_log}" >/dev/null &
    local monitor_pid=$!

    timeout --signal=TERM 900s \
        mpirun -n "${n_ranks}" --oversubscribe --bind-to none \
        python -u "${driver}" \
        --dir "${stage}" \
        --max-events "${max_events}" \
        --batch-size 20 \
        --pool-depth 1 \
        --case "${tag}" \
        2>&1 | tee "${case_log}"
    local status=${PIPESTATUS[0]}

    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true

    local observed_gpus
    observed_gpus=$(awk -F',' '
        {
            gsub(/ /, "", $2); gsub(/ /, "", $3)
            if (($3 + 0) > peak[$2]) peak[$2] = $3 + 0
        }
        END {
            count = 0
            for (gpu in peak) {
                printf("GPU_PHYSICAL_PEAK index=%s mib=%d\n", gpu, peak[gpu]) > "/dev/stderr"
                if (peak[gpu] > 1000) count++
            }
            print count
        }
    ' "${mem_log}")
    echo "PHYSICAL_GPU_CHECK tag=${tag} expected=${n_gpus} observed=${observed_gpus}"
    if [[ ${status} -eq 0 && ${observed_gpus} -ne ${n_gpus} ]]; then
        echo "PHYSICAL_GPU_CHECK_FAILED tag=${tag}" >&2
        status=97
    fi
    echo "CASE_END tag=${tag} status=${status} date=$(date -Is)"
}

for repetition in 1 2; do
    for n_bds in 1 2 4 6 8; do
        run_case 1 "${n_bds}" "${repetition}"
    done
    for n_bds in 2 4 6; do
        run_case 2 "${n_bds}" "${repetition}"
    done
    for n_bds in 4 8 12; do
        run_case 4 "${n_bds}" "${repetition}"
    done
done

echo "scale_complete=true date=$(date -Is)"
