#!/bin/bash
# =============================================================================
# submit_cold_control.sh
# Cold-window control for the tuned config found by the design-space matrix
# (job 32765825): 4 BD x 4 streams x batch 16, sorted azint.
#
# Discipline: the tuned config runs FIRST in a fresh allocation, so its figure
# is a cold-window number comparable to the ~230 Hz/node raw-storage floor.
# A same-config repeat (warm) and the old default topology (warm) follow for
# the inflation factor and a reference point; they cannot contaminate phase 1.
# The node that ran the design-space job is excluded (page cache persists
# across allocations on the same node).
#
# Usage: sbatch psana/psana/gpu/scripts/submit_cold_control.sh
# =============================================================================
#SBATCH -p ampere
#SBATCH -A lcls:data
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:a100:1
#SBATCH -x sdfampere010
#SBATCH -t 00:45:00
#SBATCH -o /sdf/scratch/users/a/ajshack/.claude_tmp/cold_control_%j.out
#SBATCH -J gpu_cold_control

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
    echo "════ config: $tag (${nbd} BD) $(date +%H:%M:%S) host=$(hostname) ════"
    local SD="$OUT/stats_${tag}_${SLURM_JOB_ID}"
    mkdir -p "$SD"
    PS_EB_NODES=1 timeout 12m mpirun -n $((nbd + 2)) --bind-to none --oversubscribe \
        python psana/psana/gpu/bench_pipeline_stages.py $COMMON \
        --stats-dir "$SD" "$@" \
        2>&1 | grep -vE "UserWarning|self.gpu_reader|kvikio I/O|\[stages\] rank"
    python psana/psana/gpu/bench_pipeline_stages.py --report-dir "$SD"
    echo
}

TUNED="--azint sorted --n_gpu_streams 4 --batch_size 16"

run_cfg cold_tuned_4bd    4 $TUNED
run_cfg warm_tuned_4bd    4 $TUNED
run_cfg warm_default_16bd 16 --azint sorted --n_gpu_streams 2 --batch_size 1

echo "all configs done $(date +%H:%M:%S)"
