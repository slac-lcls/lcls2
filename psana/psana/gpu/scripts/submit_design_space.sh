#!/bin/bash
# =============================================================================
# submit_design_space.sh
# Design-space matrix on one A100, mfx101572426 r47 (FFB):
#
#   knee_*  — atomic-kernel contention knee: k_azint ms vs BD-rank count.
#             Metric is per-event kernel time (cache-warming immune).
#   sen*    — drift sentinels: identical config at start/middle/end measures
#             page-cache warming across the job; anchors absolute Hz claims.
#   sv_*    — streams-vs-ranks at matched total concurrency (ranks x streams
#             = 32), run forward then reversed (a/b) to cancel window drift.
#             Trade-off under test: per-rank CUDA contexts + constants vs
#             per-stream constant caches (~384 MB per stream per rank).
#   bs_*    — batch-size sweep at 4 BD (slot bufs = 4 slots x bs x 67 MB per
#             rank keeps bs=16 inside 40 GB only at low rank counts).
#   d2h_*   — full-frame D2H (the naive user workflow) vs the 3 KB histogram.
#   mps_*   — same shapes under CUDA MPS: does one shared context lower the
#             OOM ceiling and/or the atomic contention dilation?
#
# Usage: sbatch psana/psana/gpu/scripts/submit_design_space.sh
# =============================================================================
#SBATCH -p ampere
#SBATCH -A lcls:data
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:a100:1
#SBATCH -t 04:00:00
#SBATCH -o /sdf/scratch/users/a/ajshack/.claude_tmp/design_space_%j.out
#SBATCH -J gpu_design_space

REPO=/sdf/scratch/users/a/ajshack/lcls2-kernels
OUT=/sdf/scratch/users/a/ajshack/.claude_tmp
source "$REPO/setup_env.sh" >/dev/null 2>&1
export TMPDIR=/tmp
cd "$REPO"

EXP=mfx101572426
RUN=47
DIR=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc
COMMON="-e $EXP -r $RUN --dir $DIR --n_warmup 100 --n_events 500"

run_cfg () {
    local tag="$1" nbd="$2"; shift 2
    echo "════ config: $tag (${nbd} BD) $(date +%H:%M:%S) ════"
    local SD="$OUT/stats_${tag}_${SLURM_JOB_ID}"
    mkdir -p "$SD"
    PS_EB_NODES=1 timeout 12m mpirun -n $((nbd + 2)) --bind-to none --oversubscribe \
        python psana/psana/gpu/bench_pipeline_stages.py $COMMON \
        --stats-dir "$SD" "$@" \
        2>&1 | grep -vE "UserWarning|self.gpu_reader|kvikio I/O|\[stages\] rank"
    python psana/psana/gpu/bench_pipeline_stages.py --report-dir "$SD"
    echo
}

SORTED2="--azint sorted --n_gpu_streams 2 --batch_size 1"

# ── Sentinel 1 ───────────────────────────────────────────────────────────────
run_cfg sen1_16bd        16 $SORTED2

# ── Atomic contention knee (metric: k_azint ms/event) ────────────────────────
run_cfg knee_atomic_1bd   1 --azint atomic --n_gpu_streams 2 --batch_size 1 --n_events 300
run_cfg knee_atomic_2bd   2 --azint atomic --n_gpu_streams 2 --batch_size 1 --n_events 300
run_cfg knee_atomic_4bd   4 --azint atomic --n_gpu_streams 2 --batch_size 1
run_cfg knee_atomic_8bd   8 --azint atomic --n_gpu_streams 2 --batch_size 1
run_cfg knee_atomic_16bd 16 --azint atomic --n_gpu_streams 2 --batch_size 1

# ── Sentinel 2 ───────────────────────────────────────────────────────────────
run_cfg sen2_16bd        16 $SORTED2

# ── Streams vs ranks, matched ranks x streams = 32, forward then reversed ────
run_cfg sv_16bd_2st_a    16 --azint sorted --n_gpu_streams 2  --batch_size 1
run_cfg sv_8bd_4st_a      8 --azint sorted --n_gpu_streams 4  --batch_size 1
run_cfg sv_4bd_8st_a      4 --azint sorted --n_gpu_streams 8  --batch_size 1
run_cfg sv_2bd_16st_a     2 --azint sorted --n_gpu_streams 16 --batch_size 1 --n_events 300
run_cfg sv_2bd_16st_b     2 --azint sorted --n_gpu_streams 16 --batch_size 1 --n_events 300
run_cfg sv_4bd_8st_b      4 --azint sorted --n_gpu_streams 8  --batch_size 1
run_cfg sv_8bd_4st_b      8 --azint sorted --n_gpu_streams 4  --batch_size 1
run_cfg sv_16bd_2st_b    16 --azint sorted --n_gpu_streams 2  --batch_size 1

# ── Batch-size sweep at 4 BD x 4 streams ─────────────────────────────────────
run_cfg bs_4bd_bs1        4 --azint sorted --n_gpu_streams 4 --batch_size 1
run_cfg bs_4bd_bs4        4 --azint sorted --n_gpu_streams 4 --batch_size 4
run_cfg bs_4bd_bs16       4 --azint sorted --n_gpu_streams 4 --batch_size 16
run_cfg bs_4bd_bs1r       4 --azint sorted --n_gpu_streams 4 --batch_size 1

# ── Full-frame D2H (naive user workflow) ─────────────────────────────────────
run_cfg d2h_calib_16bd   16 $SORTED2 --d2h calib

# ── Sentinel 3 ───────────────────────────────────────────────────────────────
run_cfg sen3_16bd        16 $SORTED2

# ── MPS: shared CUDA context ─────────────────────────────────────────────────
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$SLURM_JOB_ID
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if nvidia-cuda-mps-control -d; then
    echo "── MPS daemon started ──"
    run_cfg mps_sorted_16bd  16 $SORTED2
    run_cfg mps_atomic_16bd  16 --azint atomic --n_gpu_streams 2 --batch_size 1
    run_cfg mps_sorted_24bd  24 $SORTED2
    run_cfg mps_sorted_32bd  32 $SORTED2
    echo quit | nvidia-cuda-mps-control
    echo "── MPS daemon stopped ──"
else
    echo "── MPS unavailable on this node; skipping mps_* configs ──"
fi

echo "all configs done $(date +%H:%M:%S)"
