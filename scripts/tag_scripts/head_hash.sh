#!/bin/bash
# Compare HEAD vs origin/master commit hashes

REPO_DIR="$1"

if [ -z "$1" ]; then
    echo "Usage: $0 <repo_path>"
    exit 1
fi

cd "$REPO_DIR" || exit 1

echo "HEAD:"
git rev-parse HEAD

echo ""
echo "origin/master:"
git rev-parse origin/master