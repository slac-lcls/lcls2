#!/bin/bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=00:40:00
#SBATCH --exclude=sdfampere003,sdfampere008,sdfampere016,sdfampere024,sdfampere028,sdfampere029,sdfampere037,sdfampere039
#SBATCH --job-name=multirun
#SBATCH --output=validation/ipc-integration-20261007/multirun-%j.log
#
# Per-run rediscovery on hardware, standalone so this one case can be
# iterated without rerunning the other twelve.
#
# Three runs from one DataSource. Discovery, constant sharing and teardown all
# happen once per run, and only here is a run's cleanup failing to happen
# visible: the next run's budget is sized from a card that still holds the
# previous run's allocations, and because the sizing reduction is a job-wide
# MIN, one rank's stale pool lowers every rank on every device.
#
# Two cases, because they fail differently:
#   * one visible GPU   -- every worker is a peer, so constants are shared and
#                          the per-run establish/release cycle is exercised;
#   * two visible GPUs  -- workers split across cards, so a rank drifting to a
#                          different device between runs would show up.
#
# NOTE: no `set -u` -- ps_20241122's conda activate.d scripts reference
# unbound variables (ADDR2LINE) and abort under nounset.
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source setup_env.sh >/dev/null

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PSANA_GPU_CHECK_COLLECTIVES=1
export PSANA_GPU_COLLECTIVE_TIMEOUT=120
export CUPY_CACHE_DIR="$PWD/validation/ipc-integration-20261007/cupy-cache"
mkdir -p "$CUPY_CACHE_DIR"

D=psana/psana/tests/gpu/integration
# mfx101629726 (r0015-r0032), not mfx100848724 which has a single run, and not
# mfx100852324 whose stream 7 describes jungfrau segments twice under the
# `config` alg -- detector_field_handles refuses that and setup aborts before
# any event is read (measured; unrelated to placement). mfx101555026
# (r0008-r0012) is an equally good alternative.
EXP=mfx101629726
RUNS=15,16,17
rc=0

echo "=============================== environment ==============================="
hostname
nvidia-smi --query-gpu=index,pci.bus_id,name --format=csv,noheader
ls /sdf/data/lcls/ds/prj/public01/xtc/${EXP}-r00*-s000-c000.xtc2 | head
echo

# Sharing case. One visible card makes all four BD workers peers, so each run
# establishes a shared copy and releases it at teardown.
echo "========== 3 runs, 1 visible GPU (constants shared) =========="
export CUDA_VISIBLE_DEVICES=0
timeout 600 mpirun -n 6 -x CUDA_VISIBLE_DEVICES -x CUDA_DEVICE_ORDER \
    -x PSANA_GPU_CHECK_COLLECTIVES -x PSANA_GPU_COLLECTIVE_TIMEOUT \
    -x CUPY_CACHE_DIR \
    python -u "$D/mpi_multirun_placement.py" \
    --exp "$EXP" --runs "$RUNS" || rc=$?

# Two-card case. Workers split across devices, so this is where a rank
# selecting a different card on a later run would appear.
echo
echo "========== 3 runs, 2 visible GPUs (device stability) =========="
export CUDA_VISIBLE_DEVICES=0,1
timeout 600 mpirun -n 6 -x CUDA_VISIBLE_DEVICES -x CUDA_DEVICE_ORDER \
    -x PSANA_GPU_CHECK_COLLECTIVES -x PSANA_GPU_COLLECTIVE_TIMEOUT \
    -x CUPY_CACHE_DIR \
    python -u "$D/mpi_multirun_placement.py" \
    --exp "$EXP" --runs "$RUNS" || rc=$?

echo
echo "=============================== VERDICT ==================================="
# mpirun exits 0 even when ranks report failures, so the log decides.
log="validation/ipc-integration-20261007/multirun-${SLURM_JOB_ID}.log"
if grep -aq '"all_pass": false' "$log" 2>/dev/null; then
    echo "  a summary reported all_pass:false"; rc=1
fi
if [ "$(grep -ac '"all_pass": true' "$log" 2>/dev/null)" -lt 2 ]; then
    echo "  fewer than 2 summaries passed -- did both cases run?"; rc=1
fi
if grep -aq 'collective order diverged' "$log" 2>/dev/null; then
    echo "  a rank reported a diverging collective order"; rc=1
fi
if [ "$rc" -eq 0 ]; then echo "  OVERALL: PASS"; else echo "  OVERALL: FAIL (rc=$rc)"; fi
exit "$rc"
