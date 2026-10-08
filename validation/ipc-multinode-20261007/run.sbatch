#!/bin/bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=00:30:00
#SBATCH --exclude=sdfampere003,sdfampere008,sdfampere016,sdfampere024,sdfampere028,sdfampere029,sdfampere037,sdfampere039
#SBATCH --job-name=ipc-multinode
#SBATCH --output=validation/ipc-multinode-20261007/job-%j.log
#
# Multi-node peer discovery. On ONE node the hostname half of the
# (hostname, device_uuid) grouping key is constant, so a hostname that is
# ignored or inconsistently formatted is invisible -- and that is issue #155's
# first named failure mode: "multi-node GPU with single EB group spanning both,
# counting 8 peers instead of 16".
#
# 2 nodes x 5 tasks x 2 GPUs: 1 CPU-role rank + 4 GPU workers per node, so
# 2 workers share each of 4 distinct devices across 2 distinct hosts.
#
# A 2-node A100 reservation queues far worse than the single-node suite.
# Submit and leave it; do not iterate on this one.
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
export CUPY_CACHE_DIR="$PWD/validation/ipc-multinode-20261007/cupy-cache"
mkdir -p "$CUPY_CACHE_DIR"

D=psana/psana/tests/gpu/integration
rc=0

echo "=============================== environment ==============================="
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
echo "SLURM_NNODES=${SLURM_NNODES:-<unset>} SLURM_NTASKS=${SLURM_NTASKS:-<unset>}"
srun --ntasks-per-node=1 bash -c 'echo "$(hostname): $(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr "\n" " ")"'
echo

# PS_EB_NODES=1 spanning both nodes is #155's case A exactly: one EB group,
# workers on two hosts. The arithmetic it replaces counts every rank in
# bd_comm regardless of host; discovery must count only same-host peers.
echo "========== multi-node discovery : 1 EB spanning 2 nodes =========="
timeout 180 mpirun -n 10 python -u "$D/mpi_placement_discovery.py" \
    --n-eb 1 --expect-hosts 2 --expect-devices 4 || rc=$?

echo
echo "========== multi-node discovery : 2 EB groups =========="
timeout 180 mpirun -n 10 python -u "$D/mpi_placement_discovery.py" \
    --n-eb 2 --expect-hosts 2 --expect-devices 4 || rc=$?

echo
echo "=============================== VERDICT ==================================="
if [ "$rc" -eq 0 ]; then echo "  OVERALL: PASS"; else echo "  OVERALL: FAIL (rc=$rc)"; fi
exit "$rc"
