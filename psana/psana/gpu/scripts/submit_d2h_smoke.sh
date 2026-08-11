#!/bin/bash
# =============================================================================
# submit_d2h_smoke.sh
# Smoke test of the d2h-kernels branch: kernel unit tests + the ported
# per-stage bench in three small configs (azint off / sorted / atomic).
# Validates the port before the full design-space re-run.
#
# Usage: sbatch psana/psana/gpu/scripts/submit_d2h_smoke.sh
# =============================================================================
#SBATCH -p ampere
#SBATCH -A lcls:data
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:a100:1
#SBATCH -t 00:40:00
#SBATCH -o /sdf/scratch/users/a/ajshack/.claude_tmp/d2h_smoke_%j.out
#SBATCH -J gpu_d2h_smoke

REPO=/sdf/scratch/users/a/ajshack/lcls2-kernels
OUT=/sdf/scratch/users/a/ajshack/.claude_tmp
source "$REPO/setup_env.sh" >/dev/null 2>&1
export TMPDIR=/tmp
cd "$REPO"

echo "── unit test: JungfrauAzint vs CPU reference ──"
python psana/psana/gpu/test_azint.py || echo "TEST_AZINT_FAILED"
echo

echo "── unit test: standalone analysis kernels ──"
timeout 5m python psana/psana/gpu/test_analysis_kernels.py --iters 5 \
    || echo "TEST_ANALYSIS_FAILED"
echo

# r47 (the original campaign run) has aged off the FFB rolling buffer;
# r200 is the same 32-segment Jungfrau on the current FFB window.
EXP=mfx101572426
RUN=200
DIR=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc
COMMON="-e $EXP -r $RUN --dir $DIR --n_warmup 50 --n_events 150"

run_cfg () {
    local tag="$1" nbd="$2"; shift 2
    echo "════ config: $tag (${nbd} BD) $(date +%H:%M:%S) ════"
    local SD="$OUT/stats_${tag}_${SLURM_JOB_ID}"
    mkdir -p "$SD"
    PS_EB_NODES=1 timeout 8m mpirun -n $((nbd + 2)) --bind-to none --oversubscribe \
        python psana/psana/gpu/bench_pipeline_stages.py $COMMON \
        --stats-dir "$SD" "$@" \
        2>&1 | grep -vE "UserWarning|kvikio I/O|\[stages\] rank"
    python psana/psana/gpu/bench_pipeline_stages.py --report-dir "$SD"
    echo
}

run_cfg smoke_off_4bd     4 --azint off    --n_gpu_streams 2 --batch_size 1
run_cfg smoke_sorted_4bd  4 --azint sorted --n_gpu_streams 2 --batch_size 1
run_cfg smoke_atomic_2bd  2 --azint atomic --n_gpu_streams 2 --batch_size 1
run_cfg smoke_d2hchunk_4bd 4 --azint sorted --n_gpu_streams 2 --batch_size 1 \
    --d2h calib --d2h-chunk 10

echo "smoke done $(date +%H:%M:%S)"
