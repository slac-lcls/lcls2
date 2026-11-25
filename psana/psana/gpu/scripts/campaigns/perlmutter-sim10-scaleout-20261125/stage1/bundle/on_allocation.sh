#!/bin/bash
# Run on the allocated GPU node through the outer srun in launch-command.json.
set -euo pipefail
attempt=${1:?prepared attempt}
case "$attempt" in
 /pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-20261115/grouped-replay-20261123-r[123]) ;;
 *) exit 2 ;;
esac
[[ "$(readlink -f "$(dirname "$0")")" == "$attempt/bundle" ]]
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_NUM_NODES:-}" == 1 ]]
(set -o noclobber; date -u > "$attempt/job-started.txt")
set -o noclobber
exec > "$attempt/coordinator.log" 2>&1
set +o noclobber
TZ=UTC scontrol show job "$SLURM_JOB_ID" > "$attempt/slurm-job.txt"
lscpu > "$attempt/cpu.txt"
exec /usr/bin/python3 "$attempt/bundle/clean_exec.py" "$attempt" python "$attempt/bundle/campaign.py" "$attempt"
