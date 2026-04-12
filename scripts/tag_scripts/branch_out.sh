#!/bin/bash
# Sync tracked changes from the most recent matching source lcls2 repo to a branch repo
# Creates a branch and commits the changes locally
#
# Usage: ./branch_out.sh <hutch_name> <root_dir> <branch_dir> <prefix>
# Example: ./branch_out.sh tmo /sdf/data/lcls/ds/prj/prjcwang31/results/software /cds/sw/ds/ana/test_lcl2/ssh_clone/lcls2 lcls2

set -e

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

# Find the most recent cloned git repo whose name starts with PREFIX
MONITOR_REPO=""
LATEST_CLONE_TIME=0

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

    # Must have clone timestamp marker
    if [ ! -f "$candidate/.clone_time" ]; then
        continue
    fi

    CLONE_TIME=$(cat "$candidate/.clone_time")

    if [ "$CLONE_TIME" -gt "$LATEST_CLONE_TIME" ]; then
        LATEST_CLONE_TIME="$CLONE_TIME"
        MONITOR_REPO="$candidate"
    fi
done

if [ -z "$MONITOR_REPO" ]; then
    echo -e "${RED}Error: No matching git repo found in $ROOT_DIR with prefix '$PREFIX' and a .clone_time file ${NC}"
    exit 1
fi

echo -e "${GREEN}Most recent matching clone selected:${NC} $MONITOR_REPO"

# Get short git hash from source repo
cd "$MONITOR_REPO"
echo -e "${GREEN} REMOTE FOLDER ${NC}"
echo -e "${YELLOW}Move to monitored folder ${NC}"

git config --global --add safe.directory '*'

GIT_HASH=$(git rev-parse --short HEAD)
BRANCH_NAME="${HUTCH_NAME}-${GIT_HASH}"

echo "GIT_HASH=${GIT_HASH}"
echo "BRANCH_NAME=${BRANCH_NAME}"

echo -e "${YELLOW}diff folder with origin ${NC}"
# Get list of all modified tracked files (staged + unstaged)
git status
git add --all
CHANGED_FILES=$(git diff --name-only HEAD)

if [ -z "$CHANGED_FILES" ]; then
    echo -e "${RED}No tracked changes to sync. Exiting. ${NC}"
    exit 0
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

git checkout "$GIT_HASH"
echo -e "${GREEN}Checking hash ${GIT_HASH} ${NC} "

if git show-ref --verify --quiet refs/heads/"$BRANCH_NAME"; then
    echo -e "${YELLOW}Branch '$BRANCH_NAME' exists. Switching to it... ${NC}"
    git checkout "$BRANCH_NAME"

    echo -e "${YELLOW}Pulling latest changes for existing branch...${NC}"
    git pull origin "$BRANCH_NAME" || true
else
    echo -e "${YELLOW}Branch '$BRANCH_NAME' does not exist. Creating and switching to it... ${NC}"
    git checkout -b "$BRANCH_NAME"
fi

update=false
# Copy each changed file
echo -e "${GREEN}Synching folders ${NC}"

for file in $CHANGED_FILES; do
    if [ -f "$MONITOR_REPO/$file" ]; then
        mkdir -p "$(dirname "$file")"
        cp "$MONITOR_REPO/$file" "$file"
        git add "$file"
        update=true
        echo ".. $file synced and added to git"
    else
        if [ -f "$file" ]; then
            echo ".. $file deleted in source"
            git rm "$file"
            update=true
        else
            echo ".. $file doesn't exist in local repo. Delete not needed, skipping"
        fi
    fi
done

# Commit
echo -e "${GREEN}Commit${NC}"
if $update; then
   echo -e "${GREEN}Committing changes in branch ${BRANCH_NAME} ${NC}"
   git commit -m "Sync from lcls2 (${BRANCH_NAME})
   Source: $MONITOR_REPO
   Source commit: $GIT_HASH
   Files synced:
   $CHANGED_FILES"
else
   echo -e "${YELLOW}No new changes committed ${NC}"
fi

echo ""
echo "=== Sync Complete ==="
echo "Branch created: $BRANCH_NAME"
echo "Pushing to $BRANCH_NAME..."
git push -u origin "$BRANCH_NAME"