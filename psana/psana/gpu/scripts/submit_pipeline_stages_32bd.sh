#!/bin/bash
# =============================================================================
# submit_pipeline_stages_32bd.sh
# Single-node, 32 BD ranks on one A100, CUDA-IPC constant sharing:
# per-stage pipeline breakdown under the azint kernel knobs.
#
# Topology (PS_EB_NODES=1, 34 ranks): rank 0 SMD0, rank 1 EB, ranks 2-33 BD.
# Data: mfx101572426 r47 on FFB (the run all prior benchmark figures used).
#
# Configs:
#   1. azint off,    share on   — current-pipeline baseline
#   2. azint sorted, share on   — production-shaped reduction kernel
#   3. azint atomic, share on   — deliberate heavyweight (compute-bound)
#   4. azint sorted, share off  — A/B the IPC constant sharing
#   5. azint sorted, share on, --d2h azint — per-event 2 KB result D2H
#
# Usage: sbatch psana/psana/gpu/scripts/submit_pipeline_stages_32bd.sh
# =============================================================================
#SBATCH -p ampere
#SBATCH -A lcls:data
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:a100:1
#SBATCH -t 01:30:00
#SBATCH -o /sdf/scratch/users/a/ajshack/.claude_tmp/pipeline_stages_32bd_%j.out
#SBATCH -J gpu_pipeline_stages

REPO=/sdf/scratch/users/a/ajshack/lcls2-kernels
OUT=/sdf/scratch/users/a/ajshack/.claude_tmp
source "$REPO/setup_env.sh" >/dev/null 2>&1
export TMPDIR=/tmp
cd "$REPO"

EXP=mfx101572426
RUN=47
DIR=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc
COMMON="-e $EXP -r $RUN --dir $DIR --n_warmup 100 --n_events 500 --batch_size 1 --n_gpu_streams 2"

run_cfg () {
    local tag="$1" nbd="$2"; shift 2
    echo "════ config: $tag (${nbd} BD) ════"
    PS_EB_NODES=1 timeout 12m mpirun -n $((nbd + 2)) --bind-to none --oversubscribe \
        python psana/psana/gpu/bench_pipeline_stages.py $COMMON \
        --json_out "$OUT/stages_${tag}_${SLURM_JOB_ID}.json" "$@" \
        2>&1 | grep -vE "UserWarning|self.gpu_reader|kvikio I/O|\[stages\] rank"
    echo
}

# BD-rank ladder at fixed kernel (sorted), then knob A/Bs at 16 BD.
# 32 BD OOMs the 40 GB A100 on this branch: constants allocate per rank
# before IPC sharing engages, plus per-rank geometry + slot buffers.
run_cfg sorted_8bd      8   --azint sorted
run_cfg sorted_16bd     16  --azint sorted
run_cfg sorted_24bd     24  --azint sorted
run_cfg baseline_16bd   16  --azint off
run_cfg atomic_16bd     16  --azint atomic
run_cfg noshare_16bd    16  --azint sorted --no-share-calib
run_cfg d2h_16bd        16  --azint sorted --d2h azint

echo "all configs done"
