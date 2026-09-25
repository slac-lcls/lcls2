#!/bin/bash
# Sync local changes in matching production repos to per-repo branches in a branch repo
#
# For each production clone under <root_dir> whose name starts with <prefix>, the
# branch <hutch_name>-<repo_name> is made to match that clone exactly: the commit the
# clone is on, plus its uncommitted changes (modified and deleted tracked files, and
# new files that are not ignored by .gitignore). Editor swap/backup files are skipped.
# Changes are committed and pushed to origin.
#
# The production clones are only read, never modified.
# origin (GitHub) is the source of truth for which branches exist, so the local
# branch repo can be re-cloned at any time.
#
# Arguments:
#   <hutch_name>
#       Short identifier used to name the output branch.
#
#   <root_dir>
#       Directory containing multiple cloned git repositories.
#       The script scans this directory and filters repos by prefix.
#
#   <branch_dir>
#       Path to a single base git repository where changes will be applied.
#
#   <prefix>
#       Filter applied to repo names inside <root_dir>.
#       Only repos whose names START WITH this prefix are considered.
#
#   --dry-run
#       Report what would change, without modifying the branch repo, committing
#       or pushing. Uses the branch repo's current view of origin (no fetch).
#
# Environment:
#   LOG_ROOT   Directory for failure reports
#              (default: /sdf/group/lcls/ds/ana/sw/conda2/rel/cron_logs)
#
# Usage:
#   ./branch_out.sh [--dry-run] <hutch_name> <root_dir> <branch_dir> <prefix>
#
# Exit status: 0 if every matching repo synced, 1 otherwise.

set -e

LOG_ROOT="${LOG_ROOT:-/sdf/group/lcls/ds/ana/sw/conda2/rel/cron_logs}"

# Editor swap/backup files that are never synced (vim .x.swp, emacs x~ / .#x / #x#)
JUNK_PATTERN='(^|/)(\.[^/]*\.sw[a-p]|[^/]*~|\.#[^/]*|#[^/]*#)$'

# Don't let read-only git commands (e.g. git status) rewrite the production clones' index
export GIT_OPTIONAL_LOCKS=0

# ====== LOGGING SETUP ======
# Failure reports go to <LOG_ROOT>/<project>_branch/failed_runs, one file per failure,
# where <project> is the branch repo's directory name

log_dir() {
    echo "${LOG_ROOT}/$(basename "${BRANCH_DIR:-unknown}")_branch"
}

# Writes a detailed failure report. $1 = reason
write_failure_report() {
    local reason="$1"
    local failed_log_dir failed_log timestamp

    failed_log_dir="$(log_dir)/failed_runs"
    if ! mkdir -p "$failed_log_dir"; then
        echo "Could not create ${failed_log_dir}; failure report not written" >&2
        return 0
    fi

    timestamp=$(date +%Y-%m-%d_%H%M%S)
    failed_log="${failed_log_dir}/${timestamp}_${HUTCH_NAME:-unknown}_${REPO_NAME:-setup}_FAILED.log"
    if [ "$DRY_RUN" = true ]; then
        failed_log="${failed_log%.log}_DRYRUN.log"
    fi

    {
        echo "=========================================="
        echo "BRANCH SCRIPT FAILURE REPORT"
        echo "=========================================="
        echo ""
        echo "Timestamp: $(date)"
        echo "Hostname: $(hostname)"
        echo "Script: ${BASH_SOURCE[0]}"
        echo "Reason: ${reason}"
        echo "Dry run: ${DRY_RUN}"
        echo ""
        echo "--- Input Parameters ---"
        echo "HUTCH_NAME: ${HUTCH_NAME:-<not set>}"
        echo "ROOT_DIR: ${ROOT_DIR:-<not set>}"
        echo "BRANCH_DIR: ${BRANCH_DIR:-<not set>}"
        echo "PREFIX: ${PREFIX:-<not set>}"
        echo ""
        echo "--- Environment ---"
        echo "Current Working Directory: $(pwd)"
        echo "Git User: $(git config --global user.name 2>/dev/null || echo '<not set>')"
        echo "Git Email: $(git config --global user.email 2>/dev/null || echo '<not set>')"
        echo ""
        echo "--- Processing Context ---"
        echo "Current Repo Being Processed: ${MONITOR_REPO:-<none>}"
        echo "Current Repo Name: ${REPO_NAME:-<none>}"
        echo "Git Hash: ${GIT_HASH:-<not set>}"
        echo "Branch Name: ${BRANCH_NAME:-<not set>}"
        echo "Changed Files:"
        if [ ${#CHANGED_FILES[@]} -gt 0 ]; then
            printf '%s\n' "${CHANGED_FILES[@]}"
        else
            echo "<none>"
        fi
        echo ""

        # Try to get git status from both repos if they're set
        if [ -n "$MONITOR_REPO" ] && [ -d "$MONITOR_REPO/.git" ]; then
            echo "--- Source Repo Git Status ---"
            git -C "$MONITOR_REPO" status 2>&1 || echo "Could not get git status"
            echo ""
        fi

        if [ -n "$BRANCH_DIR" ] && [ -d "$BRANCH_DIR/.git" ]; then
            echo "--- Branch Repo Git Status ---"
            git -C "$BRANCH_DIR" status 2>&1 || echo "Could not get git status"
            echo ""
        fi

        echo "=========================================="
        echo "END FAILURE REPORT"
        echo "=========================================="
    } > "$failed_log" 2>&1

    echo ""
    echo "FAILURE LOG WRITTEN TO: $failed_log" >&2
}

# Error handler for unexpected failures outside the per-repo sync (e.g. fetch)
handle_error() {
    local exit_code=$?
    local line_number=$1
    write_failure_report "Unexpected error at line ${line_number} (exit code ${exit_code})"
    exit $exit_code
}

# Set up error trap
trap 'handle_error $LINENO' ERR
# ====== END LOGGING SETUP ======

# Colors for output, only when writing to a terminal, so log files stay plain text.
# The "### " and "-- " markers are kept either way, so errors and notes are easy to grep.
if [ -t 1 ]; then
    RED='### \033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='-- \033[1;33m'
    NC='\033[0m' # No Color
else
    RED='### ' GREEN='' YELLOW='-- ' NC=''
fi

DRY_RUN=false
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN=true
    else
        ARGS+=("$arg")
    fi
done

HUTCH_NAME="${ARGS[0]}"
ROOT_DIR="${ARGS[1]}"
BRANCH_DIR="${ARGS[2]}"
PREFIX="${ARGS[3]}"

# Check for required arguments
if [ -z "$HUTCH_NAME" ] || [ -z "$ROOT_DIR" ] || [ -z "$BRANCH_DIR" ] || [ -z "$PREFIX" ]; then
    echo -e "${RED}Error: Hutch name, root directory, branch directory, and prefix are required ${NC}"
    echo "Usage: $0 [--dry-run] <hutch_name> <root_dir> <branch_dir> <prefix>"
    echo "Example: $0 tmo /path/to/clones /path/to/branch_repo <prefix>"
    exit 1
fi

echo ""
echo -e "======= ${GREEN} Start hutch analysis ${NC}"
echo "Parameters provided: "
echo -e "${YELLOW}HUTCH_NAME=${HUTCH_NAME}${NC}"
echo -e "${YELLOW}ROOT_DIR=${ROOT_DIR}${NC}"
echo -e "${YELLOW}BRANCH_DIR=${BRANCH_DIR}${NC}"
echo -e "${YELLOW}PREFIX=${PREFIX}${NC}"
if $DRY_RUN; then
    echo -e "${YELLOW}DRY RUN: the branch repo will not be modified, nothing will be committed or pushed${NC}"
fi

# Validate root directory exists
if [ ! -d "$ROOT_DIR" ]; then
    echo -e "${RED}Error: $ROOT_DIR is not a valid directory ${NC}"
    exit 1
fi

# Validate branch directory exists and is a git repo
if [ ! -d "$BRANCH_DIR/.git" ]; then
    echo -e "${RED}Error: $BRANCH_DIR is not a valid git repository ${NC}"
    exit 1
fi

PROJECT=$(basename "$BRANCH_DIR")

# Find the matching production clones
CANDIDATES=()
for candidate in "$ROOT_DIR"/*; do
    [ -d "$candidate" ] || continue
    [[ "$(basename "$candidate")" == ${PREFIX}* ]] || continue
    [ -d "$candidate/.git" ] || continue
    CANDIDATES+=("$candidate")
done

if [ ${#CANDIDATES[@]} -eq 0 ]; then
    echo -e "${RED}Error: No matching git repo found in $ROOT_DIR with prefix '$PREFIX' ${NC}"
    exit 1
fi

# Trust exactly these repos in git's ownership check (safe.directory), for this
# process only. Nothing is added to the global git config.
SAFE_DIRS=("$BRANCH_DIR" "${CANDIDATES[@]}")
export GIT_CONFIG_COUNT=${#SAFE_DIRS[@]}
for i in "${!SAFE_DIRS[@]}"; do
    export "GIT_CONFIG_KEY_${i}=safe.directory"
    export "GIT_CONFIG_VALUE_${i}=$(cd "${SAFE_DIRS[$i]}" && pwd -P)"
done

TMP_FILE=$(mktemp)
ERR_FILE=$(mktemp)   # stderr of the last checked command, so failure reasons can quote git's own message
trap 'rm -f "$TMP_FILE" "$ERR_FILE"' EXIT

# " (<git's reason>)" from ERR_FILE, or nothing if it's empty. Git often ends with a
# generic line ("error: failed to push some refs", "...and the repository exists."),
# so this quotes the last two lines that carry a real reason (fatal:, error:, remote:,
# "! [rejected] ...", "Permission denied"), falling back to the last non-empty line.
err_suffix() {
    local e
    e=$(grep -E 'fatal:|error:|remote:[[:space:]]*[^[:space:]]|![[:space:]]*\[|Permission denied' "$ERR_FILE" |
        grep -v 'failed to push some refs' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//; s/[[:space:]]\{2,\}/ /g' | tail -n 2 | paste -sd ';' - | sed 's/;/; /g')
    if [ -z "$e" ]; then
        e=$(grep -v '^[[:space:]]*$' "$ERR_FILE" | tail -n 1)
    fi
    if [ -n "$e" ]; then
        echo " ($e)"
    fi
}

# ====== PREPARE BRANCH REPO (once) ======
cd "$BRANCH_DIR"
echo ""
echo -e "${YELLOW}Preparing branch repo ${NC}"

# Note: The branch repo should never have changes. If it does, we do not want to stash and have it be forgotten
# We will abort until the user resolves a clean local repo. This command checks tracked and untracked files.
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${RED}Error: Base repo is not clean. Aborting.${NC}"
    write_failure_report "Branch repo is not clean"
    exit 1
fi

if $DRY_RUN; then
    echo -e "${YELLOW}[dry-run] Not fetching; using the branch repo's last fetch of origin${NC}"
else
    echo -e "Fetching from origin..."
    git fetch --prune origin

    echo -e "Switching to master branch..."
    git checkout -q master

    # master is only where the repo rests between syncs, so failing to update it isn't fatal
    if ! git merge --ff-only -q origin/master; then
        echo -e "${YELLOW}Warning: could not fast-forward master to origin/master; continuing${NC}"
    fi
fi
echo -e "${GREEN}Branch repo ready${NC}"
echo ""

# ====== PER-REPO SYNC ======

# Blob hash of a file in the production clone (symlinks hash their target path, as git stores them)
worktree_blob() {
    local path="$1"
    if [ -L "$path" ]; then
        printf '%s' "$(readlink "$path")" | git hash-object --stdin
    else
        git hash-object -- "$path"
    fi
}

# Blob hash of a file at a commit in the branch repo, or empty if it isn't there
commit_blob() {
    git -C "$BRANCH_DIR" rev-parse --verify --quiet "$1:$2" 2>/dev/null || true
}

# Syncs one production clone to its branch. Returns 1 on failure with FAIL_STEP set.
# This runs as the condition of an `if`, where `set -e` does not apply, so every
# step that can fail is checked explicitly.
sync_repo() {
    local base_ref base_desc file source_blob base_blob hash_blob ahead
    local -a tracked untracked touched
    local -A want=()
    local -A seen=()

    MONITOR_REPO="$1"
    REPO_NAME=$(basename "$MONITOR_REPO")
    BRANCH_NAME="${HUTCH_NAME}-${REPO_NAME}"
    GIT_HASH=""
    CHANGED_FILES=()
    FAIL_STEP=""

    echo -e "${GREEN}Processing matching repo:${NC} $MONITOR_REPO"

    # --- Read the production clone (read-only) ---
    if ! GIT_HASH=$(git -C "$MONITOR_REPO" rev-parse HEAD 2> "$ERR_FILE"); then
        cat "$ERR_FILE" >&2
        FAIL_STEP="could not read HEAD of $MONITOR_REPO$(err_suffix)"
        return 1
    fi
    echo "GIT_HASH=${GIT_HASH:0:9}"
    echo "BRANCH_NAME=${BRANCH_NAME}"

    # Modified/deleted tracked files, staged or not
    if ! git -C "$MONITOR_REPO" diff -z --name-only --no-renames HEAD > "$TMP_FILE" 2> "$ERR_FILE"; then
        cat "$ERR_FILE" >&2
        FAIL_STEP="git diff failed in $MONITOR_REPO$(err_suffix)"
        return 1
    fi
    mapfile -d '' tracked < "$TMP_FILE"

    # New files, respecting .gitignore
    if ! git -C "$MONITOR_REPO" ls-files -z --others --exclude-standard > "$TMP_FILE" 2> "$ERR_FILE"; then
        cat "$ERR_FILE" >&2
        FAIL_STEP="git ls-files failed in $MONITOR_REPO$(err_suffix)"
        return 1
    fi
    mapfile -d '' untracked < "$TMP_FILE"

    for file in "${tracked[@]}" "${untracked[@]}"; do
        if [[ "$file" =~ $JUNK_PATTERN ]]; then
            echo ".. $file skipped (editor swap/backup file)"
            continue
        fi
        want["$file"]=1
        CHANGED_FILES+=("$file")
    done

    # --- Pick what the branch starts from ---
    if ! cd "$BRANCH_DIR"; then
        FAIL_STEP="could not cd to $BRANCH_DIR"
        return 1
    fi

    if ! git rev-parse --verify --quiet "${GIT_HASH}^{commit}" >/dev/null; then
        FAIL_STEP="commit $GIT_HASH not found in branch repo (does the production clone have local commits?)"
        return 1
    fi

    if git rev-parse --verify --quiet "refs/remotes/origin/${BRANCH_NAME}" >/dev/null; then
        base_ref="origin/${BRANCH_NAME}"
        base_desc="existing branch on origin"
    elif git rev-parse --verify --quiet "refs/heads/${BRANCH_NAME}" >/dev/null; then
        base_ref="refs/heads/${BRANCH_NAME}"
        base_desc="local branch not yet on origin"
    else
        base_ref="$GIT_HASH"
        base_desc="new branch from clone commit"
    fi
    echo -e "${YELLOW}Branch '${BRANCH_NAME}': ${base_desc}${NC}"

    # Files to look at: everything changed in the clone, plus everything the branch
    # currently changes (so changes that were reverted in the clone get reverted here)
    if ! git diff -z --name-only --no-renames "$GIT_HASH" "$base_ref" > "$TMP_FILE" 2> "$ERR_FILE"; then
        cat "$ERR_FILE" >&2
        FAIL_STEP="git diff $GIT_HASH $base_ref failed$(err_suffix)"
        return 1
    fi
    mapfile -d '' touched < "$TMP_FILE"

    # --- Plan: one action per file whose branch content must change ---
    # copy    = take the file from the clone
    # delete  = the file was deleted in the clone
    # restore = the clone no longer changes this file, so reset it to the clone's commit
    local -a plan_action plan_file
    for file in "${CHANGED_FILES[@]}" "${touched[@]}"; do
        [ -z "${seen[$file]}" ] || continue
        seen["$file"]=1

        base_blob=$(commit_blob "$base_ref" "$file")
        if [ -n "${want[$file]}" ]; then
            if [ -e "$MONITOR_REPO/$file" ] || [ -L "$MONITOR_REPO/$file" ]; then
                if ! source_blob=$(worktree_blob "$MONITOR_REPO/$file"); then
                    FAIL_STEP="could not read $MONITOR_REPO/$file"
                    return 1
                fi
                [ "$source_blob" = "$base_blob" ] && continue
                plan_action+=("copy")
            else
                [ -z "$base_blob" ] && continue
                plan_action+=("delete")
            fi
        else
            hash_blob=$(commit_blob "$GIT_HASH" "$file")
            [ "$hash_blob" = "$base_blob" ] && continue
            plan_action+=("restore")
        fi
        plan_file+=("$file")
    done

    if [ ${#plan_file[@]} -eq 0 ]; then
        echo -e "${YELLOW}Branch already matches ${REPO_NAME}. Nothing to sync.${NC}"
    else
        echo -e "${YELLOW}Files to sync: ${NC}"
        for i in "${!plan_file[@]}"; do
            echo "  ${plan_action[$i]}: ${plan_file[$i]}"
        done
    fi

    if $DRY_RUN; then
        if [ ${#plan_file[@]} -gt 0 ]; then
            echo -e "${GREEN}[dry-run] Would commit ${#plan_file[@]} file(s) to ${BRANCH_NAME} and push${NC}"
        elif [ "$base_desc" = "local branch not yet on origin" ]; then
            echo -e "${GREEN}[dry-run] Would push ${BRANCH_NAME} (not yet on origin)${NC}"
        fi
        echo ""
        return 0
    fi

    # --- Apply ---
    # Also needed with nothing to sync when the branch only exists locally, so it gets pushed
    if [ ${#plan_file[@]} -gt 0 ] || [ "$base_desc" = "local branch not yet on origin" ]; then
        run git checkout -q -B "$BRANCH_NAME" "$base_ref" || return 1

        echo -e "${GREEN}Syncing files ${NC}"
        for i in "${!plan_file[@]}"; do
            file="${plan_file[$i]}"
            case "${plan_action[$i]}" in
                copy)
                    run mkdir -p "$(dirname "$file")" || return 1
                    run rm -f "$file" || return 1
                    run cp -pP "$MONITOR_REPO/$file" "$file" || return 1
                    run git add -f -- "$file" || return 1
                    echo ".. $file synced and added to git"
                    ;;
                delete)
                    run git rm -q --ignore-unmatch -- "$file" || return 1
                    echo ".. $file deleted in source"
                    ;;
                restore)
                    if [ -n "$(commit_blob "$GIT_HASH" "$file")" ]; then
                        run git checkout -q "$GIT_HASH" -- "$file" || return 1
                    else
                        run git rm -q --ignore-unmatch -- "$file" || return 1
                    fi
                    echo ".. $file no longer changed in source, reset"
                    ;;
            esac
        done

        # Commit
        echo -e "${GREEN}Commit${NC}"
        if git diff --cached --quiet; then
            echo -e "${YELLOW}No changes to commit ${NC}"
        else
            echo -e "${GREEN}Committing changes in branch ${BRANCH_NAME} ${NC}"
            run git commit -q -m "Sync from ${PROJECT} (${BRANCH_NAME})
Source: $MONITOR_REPO
Source commit: $GIT_HASH
Files synced:
$(for i in "${!plan_file[@]}"; do echo "${plan_action[$i]}: ${plan_file[$i]}"; done)" || return 1
        fi

        # Push whenever the branch has commits origin doesn't, including ones left
        # over from an earlier run whose push failed
        if git rev-parse --verify --quiet "refs/remotes/origin/${BRANCH_NAME}" >/dev/null; then
            ahead=$(git rev-list --count "origin/${BRANCH_NAME}..${BRANCH_NAME}")
        else
            ahead=1
        fi
        if [ "$ahead" -gt 0 ]; then
            echo "Pushing to $BRANCH_NAME..."
            run git push -q -u origin "$BRANCH_NAME" || return 1
            echo ""
            echo "=== Sync Complete ==="
            echo "Branch pushed: $BRANCH_NAME"
        fi

        run git checkout -q master || return 1
    fi
    echo ""
    return 0
}

# Run a command; on failure record it in FAIL_STEP together with the last line of its
# error output (e.g. "failed: git push -q -u origin xpp-x (fatal: Could not read from
# remote repository.)"). The error output is still printed to the log.
run() {
    if "$@" 2> "$ERR_FILE"; then
        cat "$ERR_FILE" >&2
        return 0
    fi
    cat "$ERR_FILE" >&2
    FAIL_STEP="failed: $*$(err_suffix)"
    return 1
}

# Put the branch repo back on a clean master after a failed sync, so the next repo starts clean.
# Safe because the repo was verified clean before any sync started.
reset_branch_repo() {
    git -C "$BRANCH_DIR" reset -q --hard &&
        git -C "$BRANCH_DIR" clean -fdq &&
        git -C "$BRANCH_DIR" checkout -q master
}

FAILED=()
SYNCED=0
for candidate in "${CANDIDATES[@]}"; do
    if sync_repo "$candidate"; then
        SYNCED=$((SYNCED + 1))
    else
        echo -e "${RED}Error: ${REPO_NAME}: ${FAIL_STEP}${NC}"
        FAILED+=("${REPO_NAME}: ${FAIL_STEP}")
        write_failure_report "$FAIL_STEP"
        if ! $DRY_RUN && ! reset_branch_repo; then
            echo -e "${RED}Error: could not reset branch repo after failure. Aborting.${NC}"
            write_failure_report "Could not reset branch repo after failure in ${REPO_NAME}"
            exit 1
        fi
        echo ""
    fi
done

echo "=== Summary ==="
echo "Repos processed: ${#CANDIDATES[@]}"
echo "Succeeded:       ${SYNCED}"
echo "Failed:          ${#FAILED[@]}"
for f in "${FAILED[@]}"; do
    echo -e "${RED}  - ${f}${NC}"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
