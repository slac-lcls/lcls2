#!/bin/bash
#SBATCH --job-name=jf_gpu_bs_pd_cold
#SBATCH --partition=ampere
#SBATCH --account=lcls
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=sdfampere010
#SBATCH --time=00:45:00
#SBATCH --output=/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-d2h-pipeline/notes/jungfrau_gpu_sweep_20260827/sweep_%j.log

repo=/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-d2h-pipeline
source_xtc=/sdf/data/lcls/ds/mfx/mfx101210926/xtc
stage=/lscratch/monarin/jf_gpu_bs_pd_${SLURM_JOB_ID}
log_dir=${repo}/notes/jungfrau_gpu_sweep_20260827
driver=${repo}/psana/psana/debugtools/ds_count_events.py
max_events=1000

source "${repo}/setup_env.sh" >/dev/null 2>&1
set -uo pipefail
export PS_EB_NODES=1
export PS_SRV_NODES=0
export PS_PARALLEL=mpi
export PYTHONUNBUFFERED=1
export SLURM_GPUS_ON_NODE=1
export OMPI_MCA_btl='^smcuda'
export TMPDIR=/lscratch/monarin/tmp
mkdir -p "${TMPDIR}" "${log_dir}"

echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID}"
echo "commit=$(git -C "${repo}" rev-parse HEAD)"
echo "git_status_begin"
git -C "${repo}" status --short
echo "git_status_end"
echo "source_xtc=${source_xtc}"
echo "stage=${stage}"
echo "max_events=${max_events}"
echo "batch_sizes=1,5,10,20"
echo "pool_depths=1,2,4"
echo "kvikio_thread_probe=1,2,4,8_at_bs10_pd2"
echo "batch_pool_matrix_kvikio_nthreads=8"
echo "mpi_topology=smd0:1,eb:1,bd:1"
echo "allocation_mode=shared"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
awk '/Cpus_allowed_list/{print "batch_" $0}' /proc/self/status

cleanup_stage() {
    case "${stage}" in
        /lscratch/monarin/jf_gpu_bs_pd_[0-9]*)
            rm -rf -- "${stage}"
            echo "removed_stage=${stage}"
            ;;
        *)
            echo "refusing_to_remove_unexpected_stage=${stage}" >&2
            ;;
    esac
}
trap cleanup_stage EXIT

echo "GPU_PREFLIGHT_BEGIN date=$(date -Is)"
mpirun -n 3 \
    --bind-to none \
    python -u "${log_dir}/test_gpu_launch.py"
preflight_status=$?
echo "GPU_PREFLIGHT_END status=${preflight_status} date=$(date -Is)"
if [[ ${preflight_status} -ne 0 ]]; then
    exit "${preflight_status}"
fi

mkdir -p "${stage}/smalldata"
cp "${source_xtc}"/smalldata/mfx101210926-r0387-s00[5-9]-c000.smd.xtc2 "${stage}/smalldata/"
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
chunk_size = 16 * 1024 * 1024
for stream, nbytes in prefix_sizes.items():
    name = f"mfx101210926-r0387-s{stream:03d}-c000.xtc2"
    src = os.path.join(source_xtc, name)
    dst = os.path.join(stage, name)
    remaining = nbytes
    with open(src, "rb", buffering=0) as source, open(dst, "wb", buffering=0) as target:
        while remaining:
            block = source.read(min(chunk_size, remaining))
            if not block:
                raise RuntimeError(f"short source while staging {src}: {remaining} bytes remain")
            target.write(block)
            remaining -= len(block)
    print(f"staged={dst} bytes={nbytes}", flush=True)
PY
sync

findmnt -T "${stage}" -o TARGET,SOURCE,FSTYPE,OPTIONS
df -hT "${stage}"
lsblk -o NAME,TYPE,FSTYPE,SIZE,MOUNTPOINTS,MODEL,TRAN
find "${stage}" -maxdepth 2 -type f -printf '%s %p\n' | sort -n

echo "gpu_inventory_begin"
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv
nvidia-smi topo -m
echo "gpu_inventory_end"

echo "pcie_inventory_begin"
lspci -D | grep -Ei 'Non-Volatile memory controller|VGA compatible controller|3D controller'
while read -r pci_addr; do
    echo "pci_device=${pci_addr}"
    lspci -s "${pci_addr#0000:}" -vv | grep -E 'LnkCap:|LnkSta:' | head -2
done < <(lspci -D | awk '/Non-Volatile memory controller|3D controller/{print $1}')
echo "pcie_inventory_end"

python - <<'PY'
import kvikio
import kvikio.defaults

dp = kvikio.DriverProperties()
print(f"kvikio_gds_available={dp.is_gds_available}")
print(f"kvikio_compat_mode={kvikio.defaults.compat_mode()}")
print(f"kvikio_default_nthreads={kvikio.defaults.get_num_threads()}")
print(f"kvikio_default_task_size={kvikio.defaults.task_size()}")
print(dp)
PY

evict_bigdata() {
    sync
    python - "${stage}" <<'PY'
import glob
import os
import sys

paths = sorted(glob.glob(os.path.join(sys.argv[1], "*.xtc2")))
if len(paths) != 5:
    raise RuntimeError(f"expected five staged BigData files, found {len(paths)}: {paths}")
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
    local batch_size=$1
    local pool_depth=$2
    local nthreads=$3
    local tag=bs${batch_size}_pd${pool_depth}_nt${nthreads}
    local case_log=${log_dir}/gpu_${tag}_evt${max_events}_cold_${SLURM_JOB_ID}.log

    evict_bigdata
    echo "CASE_BEGIN tag=${tag} date=$(date -Is)"
    export KVIKIO_NTHREADS=${nthreads}
    mpirun -n 3 \
        --bind-to none \
        python -u "${driver}" \
        --exp mfx101210926 \
        --run 387 \
        --dir "${stage}" \
        --gpu_det jungfrau \
        --batch_size "${batch_size}" \
        --gpu_pool_depth "${pool_depth}" \
        --gpu_d2h_interval 0 \
        --max_events "${max_events}" \
        --print_interval 200 \
        --show_rank_stats \
        2>&1 | tee "${case_log}"
    local status=${PIPESTATUS[0]}
    echo "CASE_END tag=${tag} status=${status} date=$(date -Is)"
    return 0
}

echo "WARMUP_BEGIN date=$(date -Is)"
export KVIKIO_NTHREADS=8
mpirun -n 3 \
    --bind-to none \
    python -u "${driver}" \
    --exp mfx101210926 \
    --run 387 \
    --dir "${stage}" \
    --gpu_det jungfrau \
    --batch_size 1 \
    --gpu_pool_depth 1 \
    --gpu_d2h_interval 0 \
    --max_events 10 \
    --print_interval 10 \
    --show_rank_stats
warmup_status=$?
echo "WARMUP_END status=${warmup_status} date=$(date -Is)"
if [[ ${warmup_status} -ne 0 ]]; then
    exit "${warmup_status}"
fi

for nthreads in 1 2 4 8; do
    run_case 10 2 "${nthreads}"
done

for pool_depth in 1 2 4; do
    for batch_size in 1 5 10 20; do
        # Already measured as the final thread-probe point.
        if [[ ${batch_size} -eq 10 && ${pool_depth} -eq 2 ]]; then
            continue
        fi
        run_case "${batch_size}" "${pool_depth}" 8
    done
done

echo "sweep_complete=true date=$(date -Is)"
