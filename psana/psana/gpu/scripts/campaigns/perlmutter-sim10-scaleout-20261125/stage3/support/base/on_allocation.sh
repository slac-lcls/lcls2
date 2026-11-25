#!/bin/bash
set -euo pipefail
attempt=${1:?attempt}
[[ "$attempt" == /pscratch/sd/m/monarin/psana2-gpu/sim10-stage2-20261124/* ]]
[[ "$(readlink -f "$(dirname "$0")")" == "$attempt/bundle" ]]
[[ "${SLURM_JOB_NUM_NODES:-}" == 1 && -n "${SLURM_JOB_ID:-}" ]]
(set -o noclobber; date -u > "$attempt/job-started.txt")
set -o noclobber
exec > "$attempt/coordinator.log" 2>&1
set +o noclobber
TZ=UTC scontrol show job "$SLURM_JOB_ID" > "$attempt/slurm-job.txt"
lscpu > "$attempt/cpu.txt"
export SIM10_RUNTIME=P SIM10_NUMERICAL_REFERENCE=0
exec /usr/bin/python3 "$attempt/bundle/clean_exec.py" "$attempt" python "$attempt/bundle/coordinator.py" "$attempt"
