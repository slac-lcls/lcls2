#!/bin/bash
# Run the branch or tag job for this repo's project, for every monitored hutch,
# one hutch after another
#
# PROJECT below names the project (the repo this copy lives in) and PREFIX the
# production clone directories it covers. The job uses the project's shared
# branch/tag repo under $REL. Runs never overlap: a run waits for any earlier one
# to finish. The lock is in LOG_BASE, so it is also shared with the monitor jobs
# of any other project that uses the same LOG_BASE.
#
# Usage:
#   ./run_monitor.sh [--dry-run] <branch|tag> [hutch ...]
#
#   branch     Run branch_out.sh (daily)
#   tag        Run single_push_collective_tag.sh (weekly)
#   hutch ...  Hutches to run for (default: HUTCHES below)
#   --dry-run  Passed to the job scripts; nothing is committed, tagged or pushed,
#              and no failure email is sent
#
# Logs, one per hutch/job per run:
#   <LOG_BASE>/<hutch>/<PROJECT>_<branch|tag>/<YYYY-MM-DD_HHMMSS>.log
# Branch failure reports:
#   <LOG_BASE>/<hutch>/<PROJECT>_branch/failed_runs/
#
# Exit status: 0 if every job succeeded, 1 otherwise.

# ====== CONFIGURATION ======
PROJECT=lcls2               # this repo; its shared repos are $REL/{branch,tag}_repo_$PROJECT/$PROJECT
PREFIX=lcls                 # production clones are $REL/<hutch>/$PREFIX*

REL="${REL:-/sdf/group/lcls/ds/ana/sw/conda2/rel}"

# Hutches run by default. Add a hutch here once its production clones are in $REL/<hutch>.
HUTCHES=(xpp)

LOG_BASE="${LOG_BASE:-$REL/cron_logs}"
LOG_KEEP_DAYS=90            # run logs older than this are deleted
LOCK_WAIT="${LOCK_WAIT:-7200}"   # seconds to wait for an earlier run to finish
MAIL_TO="${MAIL_TO-mavaylon@slac.stanford.edu}"   # set to empty to disable email
# ====== END CONFIGURATION ======

# cron runs with a minimal PATH
export PATH="/usr/bin:/bin:$PATH"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() {
    echo "Usage: $0 [--dry-run] <branch|tag> [hutch ...]"
}

DRY_RUN_ARGS=()
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN_ARGS=(--dry-run)
    else
        ARGS+=("$arg")
    fi
done

JOB="${ARGS[0]}"
if [ "$JOB" != "branch" ] && [ "$JOB" != "tag" ]; then
    usage
    exit 1
fi

if [ ${#ARGS[@]} -gt 1 ]; then
    RUN_HUTCHES=("${ARGS[@]:1}")
else
    RUN_HUTCHES=("${HUTCHES[@]}")
fi

send_mail() {
    local subject="$1" body="$2"
    if [ -n "$MAIL_TO" ] && [ ${#DRY_RUN_ARGS[@]} -eq 0 ]; then
        echo "$body" | mail -s "$subject" "$MAIL_TO"
    fi
}

if ! mkdir -p "$LOG_BASE"; then
    echo "Error: could not create $LOG_BASE"
    exit 1
fi

# Only one run at a time
exec 9> "$LOG_BASE/.run_monitor.lock"
if ! flock -w "$LOCK_WAIT" 9; then
    msg="run_monitor.sh ${PROJECT} ${JOB} on $(hostname) at $(date): an earlier run still held the lock after ${LOCK_WAIT}s. This run was skipped."
    echo "$msg"
    send_mail "Monitor ${PROJECT} ${JOB} job skipped" "$msg"
    exit 1
fi

if [ "$JOB" = "branch" ]; then
    REPO="$REL/branch_repo_${PROJECT}/${PROJECT}"
else
    REPO="$REL/tag_repo_${PROJECT}/${PROJECT}"
fi

RUN_STAMP=$(date +%Y-%m-%d_%H%M%S)
RESULTS=()
FAILED=()

echo "=== run_monitor.sh ${PROJECT} ${JOB} ${DRY_RUN_ARGS[*]} started $(date) on $(hostname) ==="
echo "Hutches: ${RUN_HUTCHES[*]}"

for hutch in "${RUN_HUTCHES[@]}"; do
    job_dir="$LOG_BASE/$hutch/${PROJECT}_${JOB}"
    log="$job_dir/${RUN_STAMP}.log"
    root_dir="$REL/$hutch"

    if [ ! -d "$root_dir" ]; then
        FAILED+=("$hutch: $root_dir does not exist")
        RESULTS+=("FAILED  $hutch  ($root_dir does not exist)")
        continue
    fi

    if ! mkdir -p "$job_dir"; then
        FAILED+=("$hutch: could not create $job_dir")
        RESULTS+=("FAILED  $hutch  (could not create $job_dir)")
        continue
    fi

    if [ "$JOB" = "branch" ]; then
        LOG_ROOT="$LOG_BASE/$hutch" "$SCRIPT_DIR/branch_out.sh" "${DRY_RUN_ARGS[@]}" \
            "$hutch" "$root_dir" "$REPO" "$PREFIX" > "$log" 2>&1
    else
        "$SCRIPT_DIR/single_push_collective_tag.sh" "${DRY_RUN_ARGS[@]}" \
            "$hutch" "$root_dir" "$REPO" "$PREFIX" > "$log" 2>&1
    fi
    rc=$?

    if [ $rc -eq 0 ]; then
        RESULTS+=("ok      $hutch  $log")
    else
        RESULTS+=("FAILED  $hutch  $log")
        FAILED+=("$hutch: exit $rc, see $log")
    fi

    # Remove old run logs (failure reports in failed_runs/ are kept)
    find "$job_dir" -maxdepth 1 -name '*.log' -mtime +"$LOG_KEEP_DAYS" -delete
done

echo ""
printf '%s\n' "${RESULTS[@]}"
echo "=== finished $(date): ${#FAILED[@]} failed ==="

if [ ${#FAILED[@]} -gt 0 ]; then
    send_mail "Monitor ${PROJECT} ${JOB} job FAILED on $(hostname)" "run_monitor.sh ${PROJECT} ${JOB} on $(hostname), started ${RUN_STAMP}

Failed:
$(printf '  %s\n' "${FAILED[@]}")

All results:
$(printf '  %s\n' "${RESULTS[@]}")"
    exit 1
fi
