#!/bin/bash
# =============================================================================
# submit_design_space_d2h.sh
# Design-space matrix on the d2h pipeline (features/psana2-gpu-d2h-kernels),
# one A100, mfx101572426 r200 (FFB, same 32-seg Jungfrau as the r47 campaign).
#
# Mirrors the r47 campaign (job 32765825) so results are comparable, plus
# branch-specific probes:
#   sen*     — drift sentinels (page-cache warming, start/middle/end)
#   knee_*   — atomic contention: k_azint vs rank count (cache-immune metric)
#   sv_*     — streams-vs-ranks at ranks x streams = 32, forward + reversed
#   bs_*     — batch-size sweep at 4 BD
#   ceil_*   — 24/32 BD ceiling probes: did leader-skip + budget + subbatch
#              move the 16-24 rank OOM ceiling?
#   d2h_*    — full-frame D2H: lazy per-call vs gpu_d2h_chunk_size pipeline
#
# Usage: sbatch psana/psana/gpu/scripts/submit_design_space_d2h.sh
# =============================================================================
#SBATCH -p ampere
#SBATCH -A lcls:data
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:a100:1
#SBATCH -t 04:00:00
#SBATCH -o /sdf/scratch/users/a/ajshack/.claude_tmp/design_space_d2h_%j.out
#SBATCH -J gpu_design_d2h

REPO=/sdf/scratch/users/a/ajshack/lcls2-kernels
OUT=/sdf/scratch/users/a/ajshack/.claude_tmp
source "$REPO/setup_env.sh" >/dev/null 2>&1
export TMPDIR=/tmp
cd "$REPO"

EXP=mfx101572426
RUN=200
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
        2>&1 | grep -vE "UserWarning|kvikio I/O|\[stages\] rank"
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

# ── Ceiling probes: did the memory work move the 16-24 rank OOM wall? ────────
run_cfg ceil_24bd        24 $SORTED2
run_cfg ceil_32bd        32 $SORTED2

# ── Full-frame D2H: lazy per-call vs the async chunk pipeline ────────────────
run_cfg d2h_lazy_16bd    16 $SORTED2 --d2h calib
run_cfg d2h_chunk10_16bd 16 $SORTED2 --d2h calib --d2h-chunk 10

# ── Sentinel 3 ───────────────────────────────────────────────────────────────
run_cfg sen3_16bd        16 $SORTED2

echo "all configs done $(date +%H:%M:%S)"
