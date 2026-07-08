#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:data
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_mn_sweep
# --nodes / --ntasks / --ntasks-per-node are given on the sbatch command line.
# Required exported vars: SUFFIX (mn2|mn4), NEV (per-BD-rank event target).

set -x

cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh
export PS_EB_NODES=1

: "${SUFFIX:?need SUFFIX}" "${NEV:?need NEV}"

OUTDIR=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep
BENCH=/sdf/scratch/users/a/ajshack/lcls2/psana/psana/gpu/bench_calib.py
XTCDIR=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101210926/xtc
NP=${SLURM_NTASKS}

echo "which mpirun: $(which mpirun)"
echo "nodelist: ${SLURM_JOB_NODELIST}  nnodes: ${SLURM_NNODES}  ntasks: ${NP}"
srun --overlap -N ${SLURM_NNODES} --ntasks-per-node=1 hostname

# Per-node NIC samplers
srun --overlap -N ${SLURM_NNODES} --ntasks-per-node=1 bash -c "
  while true; do
    date +%s >> ${OUTDIR}/nic_${SUFFIX}_\$(hostname).log
    cat /proc/net/dev >> ${OUTDIR}/nic_${SUFFIX}_\$(hostname).log
    sleep 2
  done" &
SAMPLER_PID=$!

# Pick a launcher: mpirun only if it really spans all nodes
LAUNCHER=none
if command -v mpirun >/dev/null; then
    SPAN=$(timeout 90 mpirun --bind-to none -n ${NP} hostname 2>/dev/null | sort -u | grep -c .)
    echo "mpirun span test: ${SPAN} distinct hosts (need ${SLURM_NNODES})"
    if [ "${SPAN}" -eq "${SLURM_NNODES}" ]; then
        LAUNCHER=mpirun
    fi
fi
if [ "${LAUNCHER}" = "none" ]; then
    LAUNCHER=pmix
fi
echo "launcher: ${LAUNCHER}"

run_bench() {
    case "$1" in
        mpirun) mpirun --bind-to none -n ${NP} python ${BENCH} \
                    -e mfx101210926 -r 387 -n ${NEV} --warmup 10 --dir ${XTCDIR} \
                    > ${OUTDIR}/sweep_${SUFFIX}.log 2>&1 ;;
        pmix)   srun --mpi=pmix -n ${NP} python ${BENCH} \
                    -e mfx101210926 -r 387 -n ${NEV} --warmup 10 --dir ${XTCDIR} \
                    > ${OUTDIR}/sweep_${SUFFIX}.log 2>&1 ;;
        pmi2)   srun --mpi=pmi2 -n ${NP} python ${BENCH} \
                    -e mfx101210926 -r 387 -n ${NEV} --warmup 10 --dir ${XTCDIR} \
                    > ${OUTDIR}/sweep_${SUFFIX}.log 2>&1 ;;
    esac
}

T0=$(date +%s)
run_bench ${LAUNCHER}
RC=$?

# If srun/pmix failed at startup (no aggregate), retry with pmi2
if [ ${RC} -ne 0 ] && [ "${LAUNCHER}" = "pmix" ] && \
   ! grep -q "\[aggregate\]" ${OUTDIR}/sweep_${SUFFIX}.log; then
    echo "pmix launch failed (rc=${RC}); retrying with pmi2"
    LAUNCHER=pmi2
    T0=$(date +%s)
    run_bench pmi2
    RC=$?
fi

T1=$(date +%s)
WALL=$((T1 - T0))

kill "${SAMPLER_PID}" 2>/dev/null
wait "${SAMPLER_PID}" 2>/dev/null

echo "=== CONFIG MARKER: SUFFIX=${SUFFIX} NP=${NP} NNODES=${SLURM_NNODES} launcher=${LAUNCHER} exit_code=${RC} wall_seconds=${WALL} ==="
echo "=== multinode config done ==="
