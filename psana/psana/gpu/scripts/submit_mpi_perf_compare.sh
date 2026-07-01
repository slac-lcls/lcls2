#!/bin/bash
#SBATCH -p ampere
#SBATCH -A lcls
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:2
#SBATCH --gpu-bind=none
#SBATCH -t 00:45:00
#SBATCH -o /sdf/home/s/seema/lcls2/psana/psana/gpu/notes/perf_compare_%j.out
#SBATCH -J mpi_perf_compare

# =============================================================================
# submit_mpi_perf_compare.sh
# Batch wrapper for the CPU vs GPU MPI performance benchmark.  Submits via
# sbatch so the run can proceed without blocking a terminal session.
#
# Usage (from ~/lcls2 on a login node):
#   sbatch psana/psana/gpu/scripts/submit_mpi_perf_compare.sh
#
# Monitor progress:
#   tail -f psana/psana/gpu/notes/perf_compare_<JOBID>.out
# =============================================================================

# Absolute paths — $0 inside sbatch resolves to the Slurm temp copy
# (/var/spool/slurmd/...) so dirname-based derivation does not work.
REPO_ROOT="/sdf/home/s/seema/lcls2"
BENCHMARK="${REPO_ROOT}/psana/psana/gpu/gpu_mpi_perf_compare.py"
PYTHON="/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/ps_20241122/bin/python3"

# Topology constants — must match the #SBATCH headers above.
# PS_EB_NODES_VAL=1: start_gpu() probes source=0 in bd_comm which is the
# primary EB rank.  Multiple EB ranks do not improve throughput until
# start_gpu() is updated to interleave requests across all EB ranks.
PS_EB_NODES_VAL=1
N_GPUS_PER_NODE=2

source "${REPO_ROOT}/setup_env.sh" 2>/dev/null || true

# Force unbuffered Python stdout so output appears in the log file immediately
# rather than being held in Python's 8 KB pipe buffer until the run ends.
export PYTHONUNBUFFERED=1

export PS_EB_NODES=${PS_EB_NODES_VAL}
export PS_SRV_NODES=0
export SLURM_GPUS_ON_NODE=${N_GPUS_PER_NODE}
export OMPI_MCA_btl='^smcuda'
unset  PS_TEST_GPU_STREAM_IDS
mkdir -p "/lscratch/${USER:-nobody}/tmp"
export TMPDIR="/lscratch/${USER:-nobody}/tmp"

echo "======================================================"
echo " psana2 MPI CPU vs GPU performance benchmark"
echo " Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo " Started: $(date)"
N_BD=${N_GPUS_PER_NODE}
N_TOTAL=$(( 1 + PS_EB_NODES_VAL + N_BD ))
echo " Topology: ${N_TOTAL} ranks (smd0=1, eb=${PS_EB_NODES_VAL}, bd=${N_BD}), ${N_GPUS_PER_NODE} A100s"
echo "======================================================"
echo ""

srun --mpi=pmix bash -c '
    R=${SLURM_PROCID:-0}
    PS_EB='"${PS_EB_NODES_VAL}"'
    N_GPUS='"${N_GPUS_PER_NODE}"'
    if [ "${R}" -gt "${PS_EB}" ]; then
        BD_IDX=$(( R - PS_EB - 1 ))
        export CUDA_VISIBLE_DEVICES=$(( BD_IDX % N_GPUS ))
    else
        export CUDA_VISIBLE_DEVICES=""
    fi
    exec '"\"${PYTHON}\" -u \"${BENCHMARK}\""' \
        --n-events 2000 \
        --n-warmup 100 \
        --batch-sizes 10,20,50 \
        --pool-depths 4
' 2>&1 | grep --line-buffered -v "shmem: mmap\|create_and_attach\|unable to create shared\|coordinating structure\|UCX  WARN\|A requested component\|was not found\|unable to be opened\|not installed\|shared libraries\|that the component\|unable to be found\|PMIx stopped\|Framework:\|Component:\|Host:\|^---"

echo ""
echo "Finished: $(date)"
