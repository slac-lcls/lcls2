#!/bin/bash
# Loop through git repos in a directory and create tags in a single tag repository
#
# Usage: ./single_push_collective_tag.sh <hutch_name> <root_dir> <tag_repo_path> <prefix>
# ./single_push_collective_tag.sh tmo /path/to/repos /path/to/tag_repo ami
# ./single_push_collective_tag.sh tmo /path/to/repos /path/to/tag_repo lcls

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

HUTCH_NAME="$1"
ROOT_DIR="$2"
TAG_REPO_PATH="$3"
PREFIX="$4"

# Check for required arguments
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
    echo -e "${RED}Error: Hutch name, root directory path, tag repo path, and prefix are required${NC}"
    echo "Usage: $0 <hutch_name> <root_dir> <tag_repo_path> <prefix>"
    echo "Example: $0 tmo /path/to/repos /path/to/tag_repo ami"
    exit 1
fi

# Check if root directory exists
if [ ! -d "$ROOT_DIR" ]; then
    echo -e "${RED}Error: Directory '$ROOT_DIR' does not exist${NC}"
    exit 1
fi

# Check if root directory is empty
if [ -z "$(ls -A "$ROOT_DIR")" ]; then
    echo -e "${RED}Error: Directory '$ROOT_DIR' is empty${NC}"
    exit 1
fi

# Check if tag repo exists
if [ ! -d "$TAG_REPO_PATH" ]; then
    echo -e "${RED}Error: Tag repository '$TAG_REPO_PATH' does not exist${NC}"
    exit 1
fi

# Check if tag repo is a git repository
if [ ! -d "$TAG_REPO_PATH/.git" ]; then
    echo -e "${RED}Error: Tag repository '$TAG_REPO_PATH' is not a git repository${NC}"
    exit 1
fi

echo ""
echo -e "======= ${GREEN}Single Push Collective Tag${NC} ======="
echo -e "${YELLOW}Hutch Name: ${HUTCH_NAME}${NC}"
echo -e "${YELLOW}Root Directory: ${ROOT_DIR}${NC}"
echo -e "${YELLOW}Tag Repository: ${TAG_REPO_PATH}${NC}"
echo -e "${YELLOW}Prefix Filter: ${PREFIX}${NC}"
echo ""

# Prepare the tag repository
echo -e "${BLUE}Preparing tag repository...${NC}"
cd "$TAG_REPO_PATH"

# Check if the tag repo is clean
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${RED}Error: Tag repo has uncommitted changes. Please commit or stash changes first.${NC}"
    exit 1
fi
echo -e "${GREEN}Tag repo is clean${NC}"

# Fetch all commits to ensure we have all the hashes
echo -e "${BLUE}Fetching latest commits...${NC}"
git fetch --all
echo -e "${GREEN}Fetch complete${NC}"
echo ""

# Loop through items in root directory
for item in "$ROOT_DIR"/*; do
    # Skip if not a directory
    if [ ! -d "$item" ]; then
        continue
    fi

    # Check if it's a git repository
    if [ -d "$item/.git" ]; then
        REPO_NAME=$(basename "$item")

        # Only process repos that start with the requested prefix
        if [[ "$REPO_NAME" != ${PREFIX}* ]]; then
            continue
        fi

        echo -e "${BLUE}Found git repo: ${REPO_NAME}${NC}"

        # Get the clone commit hash
        cd "$item"
        GIT_HASH=$(git reflog --grep-reflog=clone -n 1 --format='%H' 2>/dev/null)

        if [ -z "$GIT_HASH" ]; then
            echo -e "${YELLOW}  Warning: Could not determine clone commit from reflog${NC}"
            echo -e "${YELLOW}  Skipping tag creation for ${REPO_NAME}${NC}"
        else
            echo -e "${GREEN}  Clone Hash: ${GIT_HASH}${NC}"

            # ✅ UPDATED HERE: hutch + YYYYMMDD
            DATE_STR=$(date +%Y%m%d)
            TAG_NAME="${HUTCH_NAME}-${DATE_STR}"
            echo -e "${GREEN}  Tag Name: ${TAG_NAME}${NC}"

            # Switch to tag repository to create the tag
            cd "$TAG_REPO_PATH"

            # Check if commit exists in tag repo
            if ! git rev-parse "$GIT_HASH" >/dev/null 2>&1; then
                echo -e "${YELLOW}  Warning: Commit ${GIT_HASH} not found in tag repo. Skipping.${NC}"
            else
                # Check if tag already exists locally in tag repo
                if git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
                    echo -e "${YELLOW}  Tag '${TAG_NAME}' already exists in tag repo. Skipping creation.${NC}"
                else
                    echo -e "${GREEN}  Creating tag ${TAG_NAME} in tag repo${NC}"
                    git tag -a "$TAG_NAME" "$GIT_HASH" -m "Tag for ${HUTCH_NAME} install
Commit: ${GIT_HASH}
Repo: ${REPO_NAME}
Path: ${item}"
                fi
            fi
        fi
        echo ""
    fi
done

# Push all tags at once
echo -e "${BLUE}Pushing all tags to origin...${NC}"
cd "$TAG_REPO_PATH"

# Check if there are any tags to push
if git tag | grep -q "^${HUTCH_NAME}-"; then
    git push origin --tags
    echo -e "${GREEN}All tags pushed successfully${NC}"
else
    echo -e "${YELLOW}No tags to push${NC}"
fi

echo ""
echo "=== Single Push Collective Tag Complete ==="