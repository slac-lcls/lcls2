#!/bin/bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=00:40:00
#SBATCH --exclude=sdfampere003,sdfampere008,sdfampere016,sdfampere024,sdfampere028,sdfampere029,sdfampere037,sdfampere039
#SBATCH --job-name=ipc-integration
#SBATCH --output=validation/ipc-integration-20261007/job-%j.log
#
# Integration tests for features/psana2-ipc: device peer discovery (2A) and
# shared calibration constants (2B), on real A100s with real CUDA IPC.
#
# Cases are driven by run_cases.py, one MPI job each with a private PRRTE
# session, because a shell loop of back-to-back `mpirun` calls hung: they
# share session state and an aborting case can block the next launch.
#
# NOTE: no `set -u` -- ps_20241122's conda activate.d scripts reference
# unbound variables (ADDR2LINE) and abort under nounset.
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source setup_env.sh >/dev/null

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Verify every peer makes the same collectives in the same order. Path
# divergence is what every hang on this branch turned out to be, and nothing
# else detects it; the checker is cheap enough to leave on in validation.
export PSANA_GPU_CHECK_COLLECTIVES=1
export CUPY_CACHE_DIR="$PWD/validation/ipc-integration-20261007/cupy-cache"
mkdir -p "$CUPY_CACHE_DIR"

echo "=============================== environment ==============================="
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name --format=csv
python -c 'import psana.gpu.gpu_placement as p, psana.gpu.gpu_shared_constants as s; print(p.__file__); print(s.__file__)'
echo

python -u validation/ipc-integration-20261007/run_cases.py
