#!/bin/bash
# Sync tracked changes from matching source lcls2 repos to a branch repo
# Creates a branch and commits the changes locally
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
# Usage:
#   ./branch_out.sh <hutch_name> <root_dir> <branch_dir> <prefix>

set -e

# ====== LOGGING SETUP ======
# Determine log directory based on PREFIX and common path structure
# This will be set based on where cron redirects output
# For lcls prefix -> /sdf/group/lcls/ds/ana/sw/conda2/rel/cron_logs/lcls2_branch
# For ami prefix -> /sdf/group/lcls/ds/ana/sw/conda2/rel/cron_logs/ami_branch

# Error handler - writes detailed failure log
handle_error() {
    local exit_code=$?
    local line_number=$1
    
    # Determine log directory from script arguments
    # PREFIX is $4, will be set later but we need it in the trap
    local script_prefix="${4:-unknown}"
    
    # Derive log directory
    if [ "$script_prefix" = "lcls" ]; then
        LOG_DIR="/sdf/group/lcls/ds/ana/sw/conda2/rel/cron_logs/lcls2_branch"
    elif [ "$script_prefix" = "ami" ]; then
        LOG_DIR="/sdf/group/lcls/ds/ana/sw/conda2/rel/cron_logs/ami_branch"
    else
        LOG_DIR="/sdf/group/lcls/ds/ana/sw/conda2/rel/cron_logs/${script_prefix}_branch"
    fi
    
    # Create failed_runs directory if it doesn't exist
    FAILED_LOG_DIR="${LOG_DIR}/failed_runs"
    mkdir -p "$FAILED_LOG_DIR"
    
    # Generate failed log filename with timestamp
    TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
    FAILED_LOG="${FAILED_LOG_DIR}/${TIMESTAMP}_FAILED.log"
    
    # Write detailed failure information
    {
        echo "=========================================="
        echo "BRANCH SCRIPT FAILURE REPORT"
        echo "=========================================="
        echo ""
        echo "Timestamp: $(date)"
        echo "Hostname: $(hostname)"
        echo "Script: ${BASH_SOURCE[0]}"
        echo "Exit Code: $exit_code"
        echo "Failed at Line: $line_number"
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
        if [ -n "$CHANGED_FILES" ]; then
            echo "$CHANGED_FILES"
        else
            echo "<none>"
        fi
        echo ""
        
        # Try to get git status from both repos if they're set
        if [ -n "$MONITOR_REPO" ] && [ -d "$MONITOR_REPO/.git" ]; then
            echo "--- Source Repo Git Status ---"
            (cd "$MONITOR_REPO" 2>/dev/null && git status 2>&1) || echo "Could not get git status"
            echo ""
        fi
        
        if [ -n "$BRANCH_DIR" ] && [ -d "$BRANCH_DIR/.git" ]; then
            echo "--- Branch Repo Git Status ---"
            (cd "$BRANCH_DIR" 2>/dev/null && git status 2>&1) || echo "Could not get git status"
            echo ""
        fi
        
        echo "=========================================="
        echo "END FAILURE REPORT"
        echo "=========================================="
    } > "$FAILED_LOG" 2>&1
    
    echo ""
    echo "FAILURE LOG WRITTEN TO: $FAILED_LOG" >&2
    
    exit $exit_code
}

# Set up error trap
trap 'handle_error $LINENO' ERR
# ====== END LOGGING SETUP ======

# Colors for output
RED='### \033[0;31m'
GREEN='\033[0;32m'
YELLOW='-- \033[1;33m'
NC='\033[0m' # No Color

HUTCH_NAME="$1"
ROOT_DIR="$2"
BRANCH_DIR="$3"
PREFIX="$4"

# Check for required arguments
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
    echo -e "${RED}Error: Hutch name, root directory, branch directory, and prefix are required ${NC}"
    echo "Usage: $0 <hutch_name> <root_dir> <branch_dir> <prefix>"
    echo "Example: $0 tmo /path/to/clones /path/to/branch_repo lcls2"
    exit 1
fi

echo ""
echo -e "======= ${GREEN} Start hutch analysis ${NC}"
echo "Parameters provided: "
echo -e "${YELLOW}HUTCH_NAME=${HUTCH_NAME}"
echo -e "${YELLOW}ROOT_DIR=${ROOT_DIR}"
echo -e "${YELLOW}BRANCH_DIR=${BRANCH_DIR}"
echo -e "${YELLOW}PREFIX=${PREFIX}"

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

MONITOR_REPO=""
LATEST_CLONE_TIME=0

if ! git config --global --get-all safe.directory 2>/dev/null | grep -Fxq '*'; then
    git config --global --add safe.directory '*' || \
        echo "Warning: could not update global safe.directory; continuing"
fi

FOUND_MATCH=false

for candidate in "$ROOT_DIR"/*; do
    [ -d "$candidate" ] || continue

    REPO_NAME=$(basename "$candidate")

    # Only process repos that start with the requested prefix
    if [[ "$REPO_NAME" != ${PREFIX}* ]]; then
        continue
    fi

    # Must be a git repo
    if [ ! -d "$candidate/.git" ]; then
        continue
    fi

    FOUND_MATCH=true
    MONITOR_REPO="$candidate"

    echo -e "${GREEN}Processing matching repo:${NC} $MONITOR_REPO"

    # Get short git hash from source repo
    cd "$MONITOR_REPO"
    echo -e "${GREEN} REMOTE FOLDER ${NC}"
    echo -e "${YELLOW}Move to monitored folder ${NC}"

    GIT_HASH=$(git rev-parse --short HEAD)
    BRANCH_NAME="${HUTCH_NAME}-${REPO_NAME}"

    echo "GIT_HASH=${GIT_HASH}"
    echo "BRANCH_NAME=${BRANCH_NAME}"

    echo -e "${YELLOW}diff folder with origin ${NC}"
    # Get list of all modified tracked files (staged + unstaged)
    git status
    git add --all
    CHANGED_FILES=$(git diff --name-only HEAD)

    if [ -z "$CHANGED_FILES" ]; then
        echo -e "${YELLOW}No tracked changes to sync for ${REPO_NAME}. Skipping.${NC}"
        continue
    fi

    echo -e "${YELLOW}Files to sync: ${NC}"
    echo "$CHANGED_FILES"
    echo ""

    # Create new branch in branch repo
    cd "$BRANCH_DIR"
    echo -e "${GREEN}LOCAL${NC}"
    echo -e "${YELLOW}move to branch folder ${NC}"
    echo -e "${YELLOW}Restoring repository to master branch...${NC}"

    # Note: The branch repo should never have changes. If it does, we do not want to stash and have it be forgotten
    # We will abort until the user resolves a clean local repo. This command checks tracked and untracked files.
    if [ -n "$(git status --porcelain)" ]; then
        echo -e "${RED}Error: Base repo is not clean. Aborting.${NC}"
        exit 1
    fi

    # Switch to master branch
    echo -e "Switching to master branch..."
    if git show-ref --verify --quiet refs/heads/master; then
        git checkout master
    else
        echo -e "${RED}Error: 'master' branch does not exist${NC}"
        exit 1
    fi

    # Pull latest changes from remote
    echo -e "Pulling latest changes from remote..."
    git pull

    echo -e "${GREEN}Repository restored to original state in master/main${NC}"
    echo ""

    if git show-ref --verify --quiet refs/heads/"$BRANCH_NAME"; then
        echo -e "${YELLOW}Branch '$BRANCH_NAME' exists. Switching to it... ${NC}"
        git checkout "$BRANCH_NAME"

        echo -e "${YELLOW}Pulling latest changes for existing branch...${NC}"
        git pull origin "$BRANCH_NAME" || true
    else
        echo -e "${YELLOW}Branch '$BRANCH_NAME' does not exist. Creating and switching to it... ${NC}"
        git checkout "$GIT_HASH"
        echo -e "${GREEN}Checking hash ${GIT_HASH} ${NC} "
        git checkout -b "$BRANCH_NAME"
    fi

    # Copy each changed file
    echo -e "${GREEN}Synching folders ${NC}"

    for file in $CHANGED_FILES; do
        if [ -f "$MONITOR_REPO/$file" ]; then
            mkdir -p "$(dirname "$file")"
            cp "$MONITOR_REPO/$file" "$file"
            git add "$file"
            echo ".. $file synced and added to git"
        else
            if [ -f "$file" ]; then
                echo ".. $file deleted in source"
                git rm "$file"
            else
                echo ".. $file doesn't exist in local repo. Delete not needed, skipping"
            fi
        fi
    done

    # Commit
    echo -e "${GREEN}Commit${NC}"
    if git diff --cached --quiet; then
        echo -e "${YELLOW}No changes to commit ${NC}"
    else
       echo -e "${GREEN}Committing changes in branch ${BRANCH_NAME} ${NC}"
       git commit -m "Sync from lcls2 (${BRANCH_NAME})
Source: $MONITOR_REPO
Source commit: $GIT_HASH
Files synced:
$CHANGED_FILES"

       echo ""
       echo "=== Sync Complete ==="
       echo "Branch created: $BRANCH_NAME"
       echo "Pushing to $BRANCH_NAME..."
       git push -u origin "$BRANCH_NAME"
    fi
done

if ! $FOUND_MATCH; then
    echo -e "${RED}Error: No matching git repo found in $ROOT_DIR with prefix '$PREFIX' ${NC}"
    exit 1
fi
