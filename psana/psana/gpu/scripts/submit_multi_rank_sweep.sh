#!/bin/bash
# =============================================================================
# submit_multi_rank_sweep.sh
# Sweep multi-GPU MPI correctness test over (batch_size × pool_depth) configs.
#
# Topology (PS_EB_NODES=1, 4 total ranks):
#   rank 0  — SMD0
#   rank 1  — EB
#   rank 2  — BD GPU 0 (A100 #0)
#   rank 3  — BD GPU 1 (A100 #1)
#
# Sweep grid:
#   batch_size : 1  5  10  20  50
#   pool_depth : 2  4  8
#   max_events : 100 per run  (enough to be statistically meaningful)
#
# Each config runs to completion before the next starts (sequential within
# this job step) so output is clearly separated.  Total runtime ~25 min.
#
# Usage (from ~/lcls2 on a login node):
#   sbatch psana/psana/gpu/scripts/submit_multi_rank_sweep.sh
# =============================================================================
#SBATCH -p ampere
#SBATCH -A lcls
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:2
#SBATCH --gpu-bind=none
#SBATCH -t 00:45:00
#SBATCH -o /sdf/home/s/seema/lcls2/psana/psana/gpu/notes/multi_rank_sweep_%j.out
#SBATCH -e /sdf/home/s/seema/lcls2/psana/psana/gpu/notes/multi_rank_sweep_%j.err
#SBATCH -J gpu_multi_rank_sweep

set -e

REPO_ROOT=/sdf/home/s/seema/lcls2
TEST_SCRIPT="${REPO_ROOT}/psana/psana/tests/test_gpu_multi_rank.py"
PYTHON=/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/ps_20241122/bin/python3

cd "${REPO_ROOT}"
source setup_env.sh

export PS_EB_NODES=1
export PS_SRV_NODES=0
export SLURM_GPUS_ON_NODE=2
export OMPI_MCA_btl='^smcuda'
unset PS_TEST_GPU_STREAM_IDS

BATCH_SIZES="1 5 10 20 50"
POOL_DEPTHS="2 4 8"
MAX_EVENTS=100

echo "======================================================================"
echo " psana2 Multi-Rank MPI Sweep"
echo " Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo " Started: $(date)"
echo " Topology: 4 ranks (smd0=1, eb=1, bd=2), 2 A100s"
echo " Grid: batch_sizes=[${BATCH_SIZES}]  pool_depths=[${POOL_DEPTHS}]"
echo " max_events=${MAX_EVENTS} per config"
echo "======================================================================"
echo ""

# Rank wrapper: sets CUDA_VISIBLE_DEVICES per rank then runs the test script.
# PS_EB=1 means ranks >1 are BD workers.
_RANK_WRAPPER_TPL='
    R=${SLURM_PROCID:-0}
    if [ "${R}" -gt "1" ]; then
        export CUDA_VISIBLE_DEVICES=$(( (R - 2) % 2 ))
    else
        export CUDA_VISIBLE_DEVICES=""
    fi
    exec __PYTHON__ __SCRIPT__ __ARGS__
'

_FILTER='grep -v "shmem: mmap\|create_and_attach\|unable to create shared\|coordinating structure\|UCX  WARN\|A requested component\|was not found\|unable to be opened\|not installed\|shared libraries\|that the component\|unable to be found\|PMIx stopped\|Framework:\|Component:\|Host:\|^---\|PSANA-INFO"'

# Accumulate summary lines for the final grid table.
SUMMARY_LINES=""

for BS in ${BATCH_SIZES}; do
    for PD in ${POOL_DEPTHS}; do
        echo "----------------------------------------------------------------------"
        echo "  Config: batch_size=${BS}  pool_depth=${PD}  max_events=${MAX_EVENTS}"
        echo "  Time:   $(date +%H:%M:%S)"
        echo "----------------------------------------------------------------------"

        ARGS="--max-events ${MAX_EVENTS} --batch-size ${BS} --pool-depth ${PD}"
        WRAPPER="${_RANK_WRAPPER_TPL//__PYTHON__/${PYTHON}}"
        WRAPPER="${WRAPPER//__SCRIPT__/${TEST_SCRIPT}}"
        WRAPPER="${WRAPPER//__ARGS__/${ARGS}}"

        # Capture output so we can both print it and extract the PASS line.
        RUN_OUT=$(srun \
            --ntasks=4 \
            --ntasks-per-node=4 \
            --cpus-per-task=2 \
            --gpu-bind=none \
            --mpi=pmix \
            --export=ALL \
            bash -c "${WRAPPER}" \
        2>&1 | eval "${_FILTER}" || true)

        echo "${RUN_OUT}"
        echo ""

        # Extract aggregate evt/s from PASS line for summary table.
        PASS_LINE=$(echo "${RUN_OUT}" | grep "^PASS" || echo "FAIL")
        if echo "${PASS_LINE}" | grep -q "^PASS"; then
            AGG=$(echo "${PASS_LINE}" | grep -oP '[\d.]+(?= evt/s)' | tail -1)
            STATUS="PASS"
        else
            AGG="--"
            STATUS="FAIL"
        fi
        SUMMARY_LINES="${SUMMARY_LINES}  bs=${BS} pd=${PD}  ${STATUS}  ${AGG} evt/s\n"
    done
done

echo ""
echo "======================================================================"
echo " Sweep summary  (max_events=${MAX_EVENTS}, 2 BD ranks, 2 A100s)"
echo "----------------------------------------------------------------------"
echo "  bs   pd   result   agg evt/s"
echo "----------------------------------------------------------------------"
printf "${SUMMARY_LINES}"
echo "======================================================================"
echo ""
echo "Finished: $(date)"
