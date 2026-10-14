#!/bin/bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=00:30:00
#SBATCH --exclude=sdfampere003,sdfampere008,sdfampere016,sdfampere024,sdfampere028,sdfampere029,sdfampere037,sdfampere039
#SBATCH --job-name=masked-device
#SBATCH --output=validation/ipc-integration-20261007/masked-%j.log
#
# A non-default device through the real DataSource -> Run path, on hardware.
#
# The unit test covers the ordinal mapping with an injected CuPy. What it
# cannot show is that a REAL rank, going through the real wiring, ends up
# running on a masked card -- which is what PR #175's P1 asks for and what a
# framework inheriting psana's current device (GLINT) depends on.
#
# The mask is REVERSED (`1,0`) rather than narrowed. Narrowing to one card
# makes it local ordinal 0, which proves nothing about the mapping: every
# wrong implementation also lands on ordinal 0. Reversed, CUDA presents the
# higher-numbered card first, so the card that was physical device 0 is now
# reachable only as local ordinal 1.
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
rc=0

echo "=============================== environment ==============================="
hostname
nvidia-smi --query-gpu=index,pci.bus_id,name --format=csv,noheader
echo

# Baseline: natural order. Records which PCI id is local ordinal 0.
echo "========== baseline: mask 0,1 (natural order) =========="
export CUDA_VISIBLE_DEVICES=0,1
timeout 300 mpirun -n 5 -x CUDA_VISIBLE_DEVICES -x CUDA_DEVICE_ORDER \
    -x PSANA_GPU_CHECK_COLLECTIVES -x PSANA_GPU_COLLECTIVE_TIMEOUT \
    -x CUPY_CACHE_DIR \
    python -u "$D/mpi_real_path_placement.py" --expect-devices 2 || rc=$?

# The actual test: reversed mask. A rank must be able to reach the card that
# CUDA calls ordinal 1, and its allocations must land there.
echo
echo "========== masked: mask 1,0 (reversed) =========="
export CUDA_VISIBLE_DEVICES=1,0
timeout 300 mpirun -n 5 -x CUDA_VISIBLE_DEVICES -x CUDA_DEVICE_ORDER \
    -x PSANA_GPU_CHECK_COLLECTIVES -x PSANA_GPU_COLLECTIVE_TIMEOUT \
    -x CUPY_CACHE_DIR \
    python -u "$D/mpi_real_path_placement.py" --expect-devices 2 || rc=$?

# One visible card, chosen as the SECOND entry of the allocation. Both ranks
# must share it rather than one of them reaching for the other card.
echo
echo "========== masked: single card, the non-default one =========="
export CUDA_VISIBLE_DEVICES=1
timeout 300 mpirun -n 5 -x CUDA_VISIBLE_DEVICES -x CUDA_DEVICE_ORDER \
    -x PSANA_GPU_CHECK_COLLECTIVES -x PSANA_GPU_COLLECTIVE_TIMEOUT \
    -x CUPY_CACHE_DIR \
    python -u "$D/mpi_real_path_placement.py" \
    --expect-devices 1 --expect-peers 3 || rc=$?

echo
echo "=============================== VERDICT ==================================="
# mpirun exits 0 even when ranks report failures, so the log decides.
log="validation/ipc-integration-20261007/masked-${SLURM_JOB_ID}.log"
if grep -aq '"all_pass": false' "$log" 2>/dev/null; then
    echo "  a summary reported all_pass:false"; rc=1
fi
if [ "$(grep -ac '"all_pass": true' "$log" 2>/dev/null)" -lt 3 ]; then
    echo "  fewer than 3 summaries passed -- did every case run?"; rc=1
fi
if [ "$rc" -eq 0 ]; then echo "  OVERALL: PASS"; else echo "  OVERALL: FAIL (rc=$rc)"; fi
exit "$rc"
