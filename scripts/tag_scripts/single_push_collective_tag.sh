#!/bin/bash
# Loop through git repos in a directory and create tags in a single tag repository
#
# Each production clone under <root_dir> whose name starts with <prefix> gets an
# annotated tag, <hutch_name>-<YYYYMMDD of clone>, on the commit it was cloned at.
# GitHub (origin) is the source of truth for which tags already exist, so the local
# tag repo can be re-cloned at any time. All new tags are pushed in a single push.
#
# Usage: ./single_push_collective_tag.sh [--dry-run] <hutch_name> <root_dir> <tag_repo_path> <prefix>
# ./single_push_collective_tag.sh tmo /path/to/repos /path/to/tag_repo <prefix>
# ./single_push_collective_tag.sh --dry-run tmo /path/to/repos /path/to/tag_repo <prefix>
#
#   --dry-run   Report what would be tagged, but do not create or push any tags.
#
# Exit status: 0 if every matching repo was tagged (or already was), 1 otherwise.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
TAG_REPO_PATH="${ARGS[2]}"
PREFIX="${ARGS[3]}"

# Check for required arguments
if [ -z "$HUTCH_NAME" ] || [ -z "$ROOT_DIR" ] || [ -z "$TAG_REPO_PATH" ] || [ -z "$PREFIX" ]; then
    echo -e "${RED}Error: Hutch name, root directory path, tag repo path, and prefix are required${NC}"
    echo "Usage: $0 [--dry-run] <hutch_name> <root_dir> <tag_repo_path> <prefix>"
    echo "Example: $0 tmo /path/to/repos /path/to/tag_repo <prefix>"
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
if $DRY_RUN; then
    echo -e "${YELLOW}DRY RUN: no tags will be created or pushed${NC}"
fi
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

# Read the tags that already exist on origin. The local repo may be missing some
# (e.g. after a re-clone), so origin decides whether a tag already exists.
# Annotated tags are listed twice; the "^{}" line holds the commit they point to.
echo -e "${BLUE}Reading existing tags from origin...${NC}"
declare -A REMOTE_TAGS
REMOTE_TAG_LIST=$(git ls-remote --tags origin)
while read -r sha ref; do
    [ -n "$ref" ] || continue
    name="${ref#refs/tags/}"
    if [[ "$name" == *"^{}" ]]; then
        REMOTE_TAGS["${name%^\{\}}"]="$sha"
    elif [ -z "${REMOTE_TAGS[$name]}" ]; then
        REMOTE_TAGS["$name"]="$sha"
    fi
done <<< "$REMOTE_TAG_LIST"
echo -e "${GREEN}Found ${#REMOTE_TAGS[@]} tags on origin${NC}"
echo ""

# Pick the tag name for GIT_HASH, starting from BASE_NAME.
# Two clones made on the same day at different commits would both want the same
# name, so later ones get a -2, -3, ... suffix.
# Sets TAG_NAME and TAG_ACTION: "exists" (on origin already), "this_run" (another
# clone at the same commit already got this tag in this run), "push" (tagged
# locally by an earlier run whose push failed), or "create".
MAX_SUFFIX=20
declare -A RUN_TAGS   # tags created or queued for push in this run -> commit
resolve_tag_name() {
    local base="$1"
    local i candidate local_hash
    for ((i = 1; i <= MAX_SUFFIX; i++)); do
        if [ "$i" -eq 1 ]; then
            candidate="$base"
        else
            candidate="${base}-${i}"
        fi

        if [ -n "${REMOTE_TAGS[$candidate]}" ]; then
            if [ "${REMOTE_TAGS[$candidate]}" = "$GIT_HASH" ]; then
                TAG_NAME="$candidate"
                TAG_ACTION="exists"
                return 0
            fi
            continue
        fi

        if [ -n "${RUN_TAGS[$candidate]}" ]; then
            if [ "${RUN_TAGS[$candidate]}" = "$GIT_HASH" ]; then
                TAG_NAME="$candidate"
                TAG_ACTION="this_run"
                return 0
            fi
            continue
        fi

        local_hash=$(git rev-parse --verify --quiet "refs/tags/${candidate}^{commit}" || true)
        if [ -n "$local_hash" ]; then
            if [ "$local_hash" = "$GIT_HASH" ]; then
                TAG_NAME="$candidate"
                TAG_ACTION="push"
                return 0
            fi
            continue
        fi

        TAG_NAME="$candidate"
        TAG_ACTION="create"
        return 0
    done
    return 1
}

TAGS_TO_PUSH=()
ALREADY_TAGGED=()
SKIPPED=()
FAILED=()

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
        GIT_HASH=$(git reflog --grep-reflog=clone -n 1 --format='%H' 2>/dev/null || true)

        if [ -z "$GIT_HASH" ]; then
            # Reflog entries expire (90 days by default), after which the clone
            # commit can no longer be read here.
            echo -e "${YELLOW}  Warning: Could not determine clone commit from reflog${NC}"
            echo -e "${YELLOW}  Skipping tag creation for ${REPO_NAME}${NC}"
            SKIPPED+=("${REPO_NAME}: no clone entry in reflog")
            echo ""
            continue
        fi
        echo -e "${GREEN}  Clone Hash: ${GIT_HASH}${NC}"

        CLONE_TIME=$(git reflog --grep-reflog=clone -n 1 --date=unix 2>/dev/null | sed -n 's/.*HEAD@{\([0-9]*\)}:.*/\1/p')

        if [ -z "$CLONE_TIME" ]; then
            echo -e "${YELLOW}  Warning: Could not determine clone time${NC}"
            SKIPPED+=("${REPO_NAME}: no clone time in reflog")
            echo ""
            continue
        fi

        DATE_STR=$(date -d @"$CLONE_TIME" +%Y%m%d)

        # Switch to tag repository to create the tag
        cd "$TAG_REPO_PATH"

        # Check if commit exists in tag repo
        if ! git rev-parse --verify --quiet "${GIT_HASH}^{commit}" >/dev/null; then
            echo -e "${YELLOW}  Warning: Commit ${GIT_HASH} not found in tag repo. Skipping.${NC}"
            SKIPPED+=("${REPO_NAME}: commit ${GIT_HASH} not found in tag repo")
            echo ""
            continue
        fi

        if ! resolve_tag_name "${HUTCH_NAME}-${DATE_STR}"; then
            echo -e "${RED}  Error: No free tag name for ${HUTCH_NAME}-${DATE_STR} (tried ${MAX_SUFFIX})${NC}"
            FAILED+=("${REPO_NAME}: no free tag name")
            echo ""
            continue
        fi
        echo -e "${GREEN}  Tag Name: ${TAG_NAME}${NC}"

        case "$TAG_ACTION" in
            exists)
                echo -e "${YELLOW}  Tag '${TAG_NAME}' already exists on origin. Skipping creation.${NC}"
                ALREADY_TAGGED+=("$TAG_NAME")
                ;;
            this_run)
                echo -e "${YELLOW}  Tag '${TAG_NAME}' already covers this commit in this run. Skipping creation.${NC}"
                ALREADY_TAGGED+=("$TAG_NAME")
                ;;
            push)
                echo -e "${YELLOW}  Tag '${TAG_NAME}' exists locally but not on origin. Will push.${NC}"
                TAGS_TO_PUSH+=("$TAG_NAME")
                RUN_TAGS["$TAG_NAME"]="$GIT_HASH"
                ;;
            create)
                if $DRY_RUN; then
                    echo -e "${GREEN}  [dry-run] Would create tag ${TAG_NAME}${NC}"
                    TAGS_TO_PUSH+=("$TAG_NAME")
                    RUN_TAGS["$TAG_NAME"]="$GIT_HASH"
                else
                    echo -e "${GREEN}  Creating tag ${TAG_NAME} in tag repo${NC}"
                    if git tag -a "$TAG_NAME" "$GIT_HASH" -m "Tag for ${HUTCH_NAME} install
Commit: ${GIT_HASH}
Repo: ${REPO_NAME}
Path: ${item}"; then
                        TAGS_TO_PUSH+=("$TAG_NAME")
                        RUN_TAGS["$TAG_NAME"]="$GIT_HASH"
                    else
                        echo -e "${RED}  Error: Failed to create tag ${TAG_NAME}${NC}"
                        FAILED+=("${REPO_NAME}: git tag failed")
                    fi
                fi
                ;;
        esac
        echo ""
    fi
done

# Push only the tags from this run, all at once
cd "$TAG_REPO_PATH"
if [ ${#TAGS_TO_PUSH[@]} -eq 0 ]; then
    echo -e "${YELLOW}No tags to push${NC}"
elif $DRY_RUN; then
    echo -e "${BLUE}[dry-run] Would push ${#TAGS_TO_PUSH[@]} tag(s) to origin:${NC}"
    printf '  %s\n' "${TAGS_TO_PUSH[@]}"
else
    echo -e "${BLUE}Pushing ${#TAGS_TO_PUSH[@]} tag(s) to origin...${NC}"
    REFSPECS=()
    for tag in "${TAGS_TO_PUSH[@]}"; do
        REFSPECS+=("refs/tags/${tag}")
    done
    if git push origin "${REFSPECS[@]}"; then
        echo -e "${GREEN}All tags pushed successfully${NC}"
    else
        echo -e "${RED}Error: Tag push failed. The tags stay in the local tag repo and will be pushed on the next run.${NC}"
        FAILED+=("push of: ${TAGS_TO_PUSH[*]}")
    fi
fi

echo ""
echo "=== Summary ==="
echo "New tags:        ${#TAGS_TO_PUSH[@]}"
echo "Already tagged:  ${#ALREADY_TAGGED[@]}"
echo "Skipped:         ${#SKIPPED[@]}"
for s in "${SKIPPED[@]}"; do
    echo -e "${YELLOW}  - ${s}${NC}"
done
echo "Failed:          ${#FAILED[@]}"
for f in "${FAILED[@]}"; do
    echo -e "${RED}  - ${f}${NC}"
done

echo ""
echo "=== Single Push Collective Tag Complete ==="

if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
