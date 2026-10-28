#!/bin/bash
# To escape preemption without editing this file, use Slurm's own override
# variables (they beat #SBATCH lines):
#
#   SBATCH_ACCOUNT=lcls:prjdat21 SBATCH_QOS=normal sbatch .../run.sbatch
#
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=00:35:00
#SBATCH --exclude=sdfampere003,sdfampere008,sdfampere016,sdfampere024,sdfampere028,sdfampere029,sdfampere037,sdfampere039
#SBATCH --job-name=ipc-integration
#SBATCH --output=validation/ipc-integration-20261007/job-%j.log
#
# PREEMPTION. lcls:default grants only `preemptable` (priority 1); the
# account above is the one that grants `normal` here -- check with
#   sacctmgr -n -P show assoc user=$USER format=Account,Partition,QOS
#
# which any `normal` job evicts immediately (PreemptExemptTime=0). Measured
# three times in one afternoon; ampere usually has >100 pending normal jobs.
# Just resubmit, or use the override above.
#
# `--requeue` would NOT help and is absent: PreemptMode=CANCEL here, so a
# preempted job is gone rather than requeued (verified -- a job carrying the
# flag was cancelled with no restart).
#
# 35 minutes, not 70. The full suite measures ~20 min (13 cases; the four
# calib-* cases dominate at ~200s each). A narrower window is a smaller
# target for the scheduler and does not change what runs.
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
export CUPY_CACHE_DIR="$PWD/validation/ipc-integration-20261007/cupy-cache"
mkdir -p "$CUPY_CACHE_DIR"

echo "=============================== environment ==============================="
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name --format=csv
# run_cases.py's wall-clock guard needs SLURM_JOB_END_TIME; printed so each
# log proves it was set (absent, the guard silently does nothing).
env | grep -E "^SLURM_JOB_(ID|END_TIME|START_TIME)=" | sort \
    || echo "SLURM_JOB_END_TIME unset -- the wall-clock guard is INACTIVE"
python -c 'import psana.gpu.gpu_placement as p, psana.gpu.gpu_shared_constants as s; print(p.__file__); print(s.__file__)'
echo

# "$@" forwards case names, so one case can be re-run alone. No arguments
# runs the full suite, which is what recorded evidence must come from.
python -u validation/ipc-integration-20261007/run_cases.py "$@"
