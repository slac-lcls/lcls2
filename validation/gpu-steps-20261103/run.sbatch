#!/bin/bash
# To escape preemption without editing this file, use Slurm's own override
# variables (they beat #SBATCH lines):
#
#   SBATCH_ACCOUNT=lcls:prjdat21 SBATCH_QOS=normal sbatch .../run.sbatch
#
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=ampere
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --job-name=gpu-steps
#SBATCH --output=validation/gpu-steps-20261103/job-%j.log
#
# Run from the repository root:
#
#   sbatch validation/gpu-steps-20261103/run.sbatch                    # run 51
#   PS_STEPS_DATASET=77 sbatch --export=ALL,PS_STEPS_DATASET=77 \
#     validation/gpu-steps-20261103/run.sbatch                         # run 77
#
# --export=ALL,... is needed for the dataset selection: a bare environment
# variable does not reach the batch script's own environment.
set -o pipefail
source ./setup_env.sh >/dev/null 2>&1
export PYTHONPATH="$PYTHONPATH:$PWD/psana"
export PS_PARALLEL=mpi PS_EB_NODES=1 PS_SRV_NODES=0
export PS_STEPS_DATASET=${PS_STEPS_DATASET:-51}
echo "=== node $(hostname) ==="; nvidia-smi --query-gpu=name --format=csv,noheader
echo "=== 4 ranks (mpirun): 1 smd0 + 1 EB + 2 BD, dataset=$PS_STEPS_DATASET ==="
# mpirun, not srun: srun gives each task its own singleton COMM_WORLD here.
# timeout, because the failure mode this driver is most likely to hit is a
# rank-role deadlock (a BD in bd_comm.Probe against an smd0/EB that never ran
# its service loop), and an MPI circular wait has no timeout of its own. 51
# reads 120 events in ~90 s (job 43000204); 77 reads 2400 across 3 steps in
# ~150 s (job 43001391), so the cap scales with the dataset. Failing here
# beats discovering a deadlock at the job's time limit. SIGKILL after the
# grace period: a rank blocked in UCX progress does not take SIGTERM.
if [ "$PS_STEPS_DATASET" = "77" ]; then LIMIT=25m; else LIMIT=12m; fi
timeout --kill-after=60 "$LIMIT" \
  mpirun -n 4 python3 -u psana/psana/tests/gpu/integration/mpi_gpu_steps.py 2>&1 | \
  grep -vE "Deprecation|warnings.warn|loop_scope|UserWarning|kvikio|self.gpu_reader"
status=${PIPESTATUS[0]}
echo "=== mpirun exit $status ==="
# timeout reports 124, or 137 once the kill-after SIGKILL lands. Name it in
# the log: mpirun exits non-zero for a rank assertion too, and the two need
# different follow-up.
if [ "$status" = "124" ] || [ "$status" = "137" ]; then
  echo "=== TIMED OUT: ranks deadlocked, not slow ==="
fi
exit "$status"
