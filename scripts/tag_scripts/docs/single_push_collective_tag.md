# `single_push_collective_tag.sh`

Tags the commit each production DAQ clone in a hutch was installed from, so there is a permanent record on GitHub of which version was deployed in which hutch, and when.

```
./single_push_collective_tag.sh [--dry-run] <hutch_name> <root_dir> <tag_repo_path> <prefix>
```

| Argument | Meaning |
|---|---|
| `hutch_name` | Hutch, e.g. `xpp`. Used as the start of the tag name. |
| `root_dir` | Directory holding that hutch's production clones, e.g. `rel/xpp`. |
| `tag_repo_path` | A clone of `lcls2` used only to create and push tags: `rel/tag_repo_lcls2/lcls2`. |
| `prefix` | `lcls`. Only directories whose names start with it are tagged. |
| `--dry-run` | Report what would be tagged; create and push nothing. |

Exit status is `0` if every matching clone is tagged (or already was), `1` if anything failed.

---

## Overview

For every production clone, the script answers two questions: *which commit was it installed from*, and *when*. It then makes sure GitHub has a tag `<hutch>-<YYYYMMDD>` pointing at that commit.

```
root_dir/lcls2_090826  ──(reflog: cloned at <commit A> on 2026-09-08)──►  tag xpp-20260908 → <commit A>
root_dir/lcls2_091826  ──(reflog: cloned at <commit B> on 2026-09-18)──►  tag xpp-20260918 → <commit B>
```

Tags are created in a separate **tag repo**, not in the production clones. The production clones are only read.

---

## Walkthrough

### 1. Arguments and `--dry-run`

```bash
DRY_RUN=false
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN=true
    else
        ARGS+=("$arg")
    fi
done
```

`--dry-run` can go anywhere on the command line. Everything else is treated as the four positional arguments. The script then checks that all four are present, that `root_dir` exists and isn't empty, and that `tag_repo_path` is a git repo.

### 2. Prepare the tag repo

```bash
cd "$TAG_REPO_PATH"
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${RED}Error: Tag repo has uncommitted changes. ..."
    exit 1
fi
git fetch --all
```

- **Clean check.** The tag repo should never have local edits. If it does, someone has been using it for something else, so the script stops rather than guess.
- **`git fetch --all`** downloads every commit from GitHub, so any commit a production clone was installed from exists locally and can be tagged. `pull` is not used: it would also merge into the local branch, which isn't needed. `fetch` is lighter.

### 3. Read the tags that already exist on GitHub

```bash
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
```

`git ls-remote --tags origin` asks GitHub for its list of tags. The result is stored in `REMOTE_TAGS`, a lookup table from tag name to the commit it points at.

**Why ask GitHub instead of the local repo:** the local tag repo can be missing tags. For example, it may have been re-cloned, or it may be out of date. If the script trusted only local tags, it would try to re-create a tag that already exists on GitHub, and the push would be rejected. GitHub is the source of truth, so the tag repo can be deleted and re-cloned at any time without breaking anything.

**The `^{}` lines:** these scripts create *annotated* tags. An annotated tag is its own git object, with a message, that points at a commit. `ls-remote` lists such a tag twice: once as the tag object, and once with `^{}` appended, showing the commit it points at. The script keeps the commit, because that's what it compares against.

### 4. Loop over the production clones

```bash
for item in "$ROOT_DIR"/*; do
    # Skip if not a directory
    if [ ! -d "$item" ]; then
        continue
    fi

    # Check if it's a git repository
    if [ -d "$item/.git" ]; then
        REPO_NAME=$(basename "$item")
        if [[ "$REPO_NAME" != ${PREFIX}* ]]; then
            continue
        fi
```

Every directory in `root_dir` that is a git repo **and** whose name starts with `prefix` is processed. A hutch directory holds clones of more than one repo, so the prefix picks out the `lcls2` ones (`lcls*`). Anything else is ignored, such as helper scripts or `old_build_scripts/`.

### 5. Find the clone commit and date from the reflog

```bash
cd "$item"
CLONE_ENTRY=$(git reflog --grep-reflog=clone -n 1 --date=unix --format='%H %gd' 2>/dev/null || true)
GIT_HASH="${CLONE_ENTRY%% *}"
...
CLONE_TIME=$(sed -n 's/.*HEAD@{\([0-9]*\)}.*/\1/p' <<< "$CLONE_ENTRY")
DATE_STR=$(date -d @"$CLONE_TIME" +%Y%m%d)
```

The **reflog** is git's local diary of where `HEAD` has been. The very first entry in a fresh clone is `clone: from https://github.com/...`, recorded with the commit that was checked out and the time it happened.

- `--grep-reflog=clone -n 1` finds that entry.
- `--format='%H %gd'` prints two things from it on one line, e.g. `c627e5e6… HEAD@{1790347793}`:
  - `%H` is the commit hash. This is the commit the clone was installed from, even if someone later pulled or checked out something else. `${CLONE_ENTRY%% *}` keeps the part before the space.
  - `%gd` is the entry's name, which `--date=unix` turns into `HEAD@{<seconds since 1970>}`, the moment of the clone. The `sed` pulls out that number and `date` turns it into `YYYYMMDD`.

Both come from a single `git reflog` call.

**Why not the commit's own date (`%ct`)?** That's when the commit was *written*, which can be weeks before it was installed. The reflog time is when the clone actually happened.

**Why not the date in the directory name?** Names like `lcls2_041726_reproduce` or `lcls2_slowscan` don't always contain it, and the install script accepts a custom date.

If there is no clone entry, the clone is **skipped** (reported in the summary, not a failure). This happens because git deletes reflog entries after 90 days by default (`gc.reflogExpire`). A clone that isn't tagged within 90 days of installation can't be tagged later, which is why this job must run regularly.

### 6. Check the commit exists in the tag repo

```bash
cd "$TAG_REPO_PATH"
if ! git rev-parse --verify --quiet "${GIT_HASH}^{commit}" >/dev/null; then
    ... SKIPPED+=("${REPO_NAME}: commit ${GIT_HASH} not found in tag repo")
    continue
fi
```

`--verify` with `^{commit}` makes git actually look the object up and confirm it's a commit. Without it, `git rev-parse <40-character hash>` succeeds for *any* well-formed hash, even one that doesn't exist. Earlier versions had that bug: the check always passed, `git tag` then failed with `fatal: bad object type`, and the whole run stopped before pushing.

A commit can be missing if the clone was made from a fork, or from a branch that was later deleted on GitHub. Such clones are skipped.

### 7. Choose the tag name

```bash
resolve_tag_name "${HUTCH_NAME}-${DATE_STR}"
```

`resolve_tag_name` tries `xpp-20260908`, then `xpp-20260908-2`, `-3` and so on (up to 20), and stops at the first name that fits:

| Where the name already exists | Points at our commit? | Result (`TAG_ACTION`) |
|---|---|---|
| on GitHub | yes | `exists`: already tagged, nothing to do |
| on GitHub | no | try the next suffix |
| created earlier in this run | yes | `this_run`: another clone at the same commit already got it |
| created earlier in this run | no | try the next suffix |
| only in the local tag repo | yes | `push`: created by an earlier run whose push failed; push it now |
| only in the local tag repo | no | try the next suffix |
| nowhere | – | `create` |

**Why suffixes:** two clones installed on the same day at *different* commits would both want `xpp-20260908`. Earlier versions skipped the second one with a warning, so that install was never recorded.

**Why track this run's tags (`RUN_TAGS`):** two clones can be at the *same* commit on the same day. The second should reuse the first's tag, not queue it again. A duplicate in the push list makes git reject the whole push (`dst ref ... receives from more than one src`).

Checks use `refs/tags/<name>`, so a *branch* that happens to share a tag's name is never mistaken for the tag.

### 8. Create the tag

```bash
git tag -a "$TAG_NAME" "$GIT_HASH" -m "Tag for ${HUTCH_NAME} install
Commit: ${GIT_HASH}
Repo: ${REPO_NAME}
Path: ${item}"
```

An annotated tag records who created it and when, plus a message saying which clone directory it came from. In a dry run this step is only reported.

### 9. Push once

```bash
REFSPECS=()
for tag in "${TAGS_TO_PUSH[@]}"; do
    REFSPECS+=("refs/tags/${tag}")
done
git push origin "${REFSPECS[@]}"
```

Only the tags created (or left unpushed) in this run are pushed, together in one `git push`. Earlier versions used `git push origin --tags`, which pushes every local tag, and pushed on every run even with nothing new.

If the push fails (for example, a network problem), the tags stay in the local tag repo. The next run finds them as "only in the local tag repo, same commit" (`push` in the table above) and pushes them then.

### 10. Summary and exit status

```
=== Summary ===
New tags:        2
Already tagged:  17
Skipped:         0
Failed:          0
```

Skipped clones are listed with the reason. The script exits `1` if anything is in **Failed** (a tag couldn't be created, no free name, or the push failed), so the caller (`run_monitor.sh`) can report it.

---

## Error handling

The script uses `set -e`, so a failure in the setup steps (fetching, reading GitHub's tags) stops the run immediately. Inside the loop, the commands that can fail for a single clone are checked with `if`, so one bad clone is recorded and the loop continues.

## Things to know

- **Tag names are per hutch** (`xpp-…`, `tmo-…`), so hutches never collide in the shared tag repo.
- **The tag repo is disposable.** Delete it and re-clone it (with an SSH remote) at any time.
- **A dry run still fetches** into the tag repo, because it needs the commits to check them. Fetching only updates the local copy of GitHub's state; nothing is created or pushed.
