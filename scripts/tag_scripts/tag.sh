#!/bin/bash
# Create a tag for the commit used by an installed repo
#
# Usage: ./tag.sh <hutch_name> <repo_path>

set -e

# Colors for output
RED='### \033[0;31m'
GREEN='\033[0;32m'
YELLOW='-- \033[1;33m'
NC='\033[0m' # No Color

HUTCH_NAME="$1"
REPO_DIR="$2"

# Check for required arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo -e "${RED}Error: Hutch name and repo path are required ${NC}"
    echo "Usage: $0 <hutch_name> <repo_path>"
    echo "Example: $0 tmo /path/to/lcls2"
    exit 1
fi

echo ""
echo -e "======= ${GREEN} Create Tag ${NC}"
echo "Parameters provided:"
echo -e "${YELLOW}HUTCH_NAME=${HUTCH_NAME}"
echo -e "${YELLOW}REPO_DIR=${REPO_DIR}"

cd "$REPO_DIR"

# Determine commit used by installation
# Get the HASH of the commit the repo was cloned with
GIT_HASH=$(git reflog --grep-reflog=clone -n 1 --format='%H')

if [ -z "$GIT_HASH" ]; then
    echo -e "${RED}Error: Could not determine clone commit from reflog${NC}"
    exit 1
fi

TAG_NAME="${HUTCH_NAME}-${GIT_HASH:0:9}"

echo "GIT_HASH=${GIT_HASH}"
echo "TAG_NAME=${TAG_NAME}"

# Check if tag already exists locally
if git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
    echo -e "${YELLOW}Tag '${TAG_NAME}' already exists locally. Skipping creation.${NC}"
else
    echo -e "${GREEN}Creating tag ${TAG_NAME}${NC}"
    git tag -a "$TAG_NAME" "$GIT_HASH" -m "Tag for ${HUTCH_NAME} install
Commit: ${GIT_HASH}
Repo: ${REPO_DIR}"
fi

# Check if tag already exists remotely
if git ls-remote --tags origin | grep -q "refs/tags/${TAG_NAME}$"; then
    echo -e "${YELLOW}Tag '${TAG_NAME}' already exists on remote. Skipping push.${NC}"
else
    echo -e "${GREEN}Pushing tag ${TAG_NAME} to origin${NC}"
    git push origin "$TAG_NAME"
fi

echo ""
echo "=== Tag Creation Complete ==="
echo "Tag: $TAG_NAME"
echo "Commit: $GIT_HASH"