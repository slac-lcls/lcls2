#!/bin/bash
#SBATCH -p ampere
#SBATCH -A lcls
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:2
#SBATCH --gpu-bind=none
#SBATCH -t 01:00:00
#SBATCH -o psana/psana/gpu/notes/sweep_1000evt_%j.out
#SBATCH -J gpu_1000evt_sweep

# =============================================================================
# submit_1000evt_sweep.sh
# 1000-event GPU performance sweep.
#
# Step 1: GPU baseline (on_gpu, no D→H)  bs={1,10}  pd=2
# Step 2: _D2hPipeline (on_cpu, lazy D→H) chunk={1,10}  bs=20  pd=2
#
# Usage (from ~/lcls2):
#   sbatch psana/psana/gpu/scripts/submit_1000evt_sweep.sh
#   tail -f psana/psana/gpu/notes/sweep_1000evt_<JOBID>.out
# =============================================================================

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
BENCHMARK="${REPO_ROOT}/psana/psana/gpu/gpu_mpi_perf_compare.py"

PS_EB_NODES_VAL=1
N_GPUS_PER_NODE=2

source "${REPO_ROOT}/setup_env.sh" 2>/dev/null || true
PYTHON="$(which python3)"

export PYTHONUNBUFFERED=1 PS_EB_NODES=${PS_EB_NODES_VAL} PS_SRV_NODES=0
export SLURM_GPUS_ON_NODE=${N_GPUS_PER_NODE} OMPI_MCA_btl='^smcuda'
unset PS_TEST_GPU_STREAM_IDS
mkdir -p "/lscratch/${USER:-nobody}/tmp" && export TMPDIR="/lscratch/${USER:-nobody}/tmp"

NOISE="shmem:\|UCX WARN\|PMIx\|PSANA-INFO\|UserWarning\|self\.gpu_reader\|site-packages\|^---\|Component:\|Framework:\|Host:\|create_and_attach\|unable to"

SRUN='srun --mpi=pmix bash -c'
RANK_SETUP='
    R=${SLURM_PROCID:-0}
    [ "${R}" -gt '"${PS_EB_NODES_VAL}"' ] \
        && export CUDA_VISIBLE_DEVICES=$(( R - '"${PS_EB_NODES_VAL}"' - 1 )) \
        || export CUDA_VISIBLE_DEVICES=""'

echo "======================================================================"
echo " GPU performance sweep — 1000 events"
echo " Job: ${SLURM_JOB_ID}  Node: $(hostname)  $(date)"
echo "======================================================================"

# ── Step 1: GPU baseline (on_gpu, no D→H) ────────────────────────────────
echo ""
echo "── Step 1: GPU baseline  bs={1,10}  pd=2  (on_gpu, no D→H) ──"

$SRUN "${RANK_SETUP}
    exec \"${PYTHON}\" -u \"${BENCHMARK}\" \
        --skip-cpu \
        --batch-sizes 1,10 --pool-depths 2 \
        --n-events 1000 --n-warmup 10 \
        --exp mfx100852324 --run 77 --det jungfrau
" 2>&1 | grep --line-buffered -v "${NOISE}"

# ── Step 2: _D2hPipeline (on_cpu, lazy D→H) ─────────────────────────────
echo ""
echo "── Step 2: _D2hPipeline  chunk={1,10}  bs=20  pd=2  (on_cpu) ──"

$SRUN "${RANK_SETUP}
    exec \"${PYTHON}\" -u \"${BENCHMARK}\" \
        --skip-cpu \
        --batch-sizes 20 --pool-depths 2 \
        --n-events 1000 --n-warmup 10 \
        --gpu-d2h-chunk-sizes 1,10 \
        --exp mfx100852324 --run 77 --det jungfrau
" 2>&1 | grep --line-buffered -v "${NOISE}"

echo ""
echo "Sweep complete: $(date)"
